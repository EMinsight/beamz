import os
import torch
import torch.nn.functional as F
import time
from beamz.simulation.backends.base import Backend

class TorchBackend(Backend):
    """High-performance PyTorch backend for FDTD (with torch.compile for CUDA/CPU and optimized MPS execution)."""

    def __init__(self, device="auto", **kwargs):
        super().__init__()
        self.device = self._select_device(device)
        self.dtype = kwargs.get("dtype", torch.float32)
        torch.set_default_dtype(self.dtype)
        
        # Enable fusion for better performance
        self.enable_fusion = kwargs.get("enable_fusion", True)
        
        # Batch size for MPS (process multiple steps at once)
        self.batch_size = kwargs.get("batch_size", 1)
        if self.device.type == "mps" and self.batch_size == 1:
            self.batch_size = 4  # Default to batch=4 on MPS for better performance
        
        # CUDA optimizations
        if self.device.type == "cuda":
            # Enable TF32 for faster matrix operations on CUDA
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set optimal CUDA memory allocation
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        # Performance tracking
        self._timers = {}
        self._timer_count = {}
        
        # CPU threading
        if self.device.type == "cpu":
            n = kwargs.get("num_threads", min(torch.get_num_threads(), os.cpu_count() or 1))
            torch.set_num_threads(n)
            torch.set_num_interop_threads(1)

        # MPS optimizations (Apple Silicon)
        if self.device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FUSION", "1")
            # Reduce memory fragmentation
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  
            # More aggressive memory reuse
            os.environ.setdefault("PYTORCH_MPS_ALLOCATOR_POLICY", "1")
            
            # Additional MPS-specific optimizations
            if self.enable_fusion:
                # This enables better kernel fusion 
                os.environ.setdefault("PYTORCH_MPS_ENFORCE_DETERMINISM", "0")
                # Enable tensor fusion to reduce kernel launches
                if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "_is_deterministic"):
                    torch.backends.mps._is_deterministic = False

        # Pre-allocate common constant placeholders
        self._cache = {}
        
        # Pre-allocate temporary buffers for field updates
        self._init_buffers()
        
        # Disable torch.compile for compatibility
        self.use_compile = kwargs.get("use_compile", False)
        if self.device.type == "mps":
            # Always use fused update for MPS
            self.kernel_fn = self._fused_update_fields
            self._fused_update = True
        else:
            # For CPU and CUDA
            self._fused_update = False
            
            # Use torch.compile if available and requested
            if torch.__version__ >= "2.1.0" and self.use_compile:
                # Mode options for torch.compile
                mode = kwargs.get("compile_mode", "reduce-overhead")
                self.compiled_update_h_fields = torch.compile(
                    self._update_h_fields_wrapper, 
                    backend="inductor", 
                    fullgraph=True,
                    mode=mode
                )
                self.compiled_update_e_field = torch.compile(
                    self._update_e_field_wrapper, 
                    backend="inductor", 
                    fullgraph=True,
                    mode=mode
                )
            else:
                # Use the non-compiled versions but still wrapped in no_grad
                self.compiled_update_h_fields = self._update_h_fields_wrapper
                self.compiled_update_e_field = self._update_e_field_wrapper
        
        # Precompute common buffer keys
        self._h_curl_x_key = "h_curl_x"
        self._h_curl_y_key = "h_curl_y"
        self._e_curl_key = "e_curl"
        
        # Batch processing state for MPS
        if self.device.type == "mps" and self.batch_size > 1:
            self._batch_counter = 0
            self._batch_buffers = {
                "Hx": None,
                "Hy": None,
                "Ez": None
            }

    def _init_buffers(self):
        """Initialize common buffers."""
        # We'll add buffers as they're needed with specific shapes
        # Initialize with empty dicts for different buffer categories
        self._field_buffers = {}
        self._temp_buffers = {}
        self._constants = {}

    def _get_temp_buffer(self, name, shape):
        """Get a temporary buffer with given name and shape."""
        key = f"{name}_{shape}"
        if key not in self._temp_buffers:
            self._temp_buffers[key] = torch.empty(shape, device=self.device, dtype=self.dtype)
        return self._temp_buffers[key]

    def _select_device(self, device_str):
        d = device_str.lower()
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if "cuda" in d and torch.cuda.is_available():
            return torch.device(device_str)
        if "mps" in d and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device(device_str)
        return torch.device("cpu")

    def _ensure(self, name, shape):
        """Get or create a buffer of given shape on self.device."""
        buf = self._cache.get(name)
        if buf is None or buf.shape != shape:
            self._cache[name] = torch.empty(shape, device=self.device, dtype=self.dtype)
        return self._cache[name]

    def _start_timer(self, name):
        """Start a timer for performance tracking."""
        self._timers[name] = time.time()
    
    def _stop_timer(self, name):
        """Stop a timer and record the elapsed time."""
        if name in self._timers:
            elapsed = time.time() - self._timers[name]
            self._timer_count[name] = self._timer_count.get(name, 0) + 1
            self._timers[name] = elapsed
            return elapsed
        return 0

    def zeros(self, shape):
        """Create an array of zeros with the given shape."""
        key = f"zeros_{shape}"
        if key not in self._cache:
            self._cache[key] = torch.zeros(shape, device=self.device, dtype=self.dtype)
        return self._cache[key].clone()
        
    def ones(self, shape):
        """Create an array of ones with the given shape."""
        key = f"ones_{shape}"
        if key not in self._cache:
            self._cache[key] = torch.ones(shape, device=self.device, dtype=self.dtype)
        return self._cache[key].clone()
        
    def copy(self, array):
        """Create a copy of the array."""
        return array.clone()

    def from_numpy(self, arr):
        """Efficiently convert NumPy array to tensor."""
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device)

    def to_numpy(self, t):
        """Convert tensor to NumPy array with minimal overhead."""
        if self.device.type == "mps":
            # On MPS, first transfer to CPU to avoid synchronization issues
            return t.detach().cpu().numpy()
        return t.detach().cpu().numpy()

    def update_h_fields(self, Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0):
        """Update magnetic field components."""
        if self._fused_update:
            # For MPS backend with fused updates, we just store parameters
            # and return tensors without actual calculation (will be done in E-field update)
            self._h_params = (Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)
            return Hx, Hy
        
        # No gradient version for better performance
        with torch.no_grad():
            return self.compiled_update_h_fields(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)

    def update_e_field(self, Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0):
        """Update electric field component."""
        if self._fused_update:
            # Batch processing for MPS (accumulate for batch_size steps before processing)
            if self.batch_size > 1:
                # Initialize batch buffers if needed
                if self._batch_buffers["Ez"] is None:
                    self._batch_buffers["Hx"] = Hx.clone()
                    self._batch_buffers["Hy"] = Hy.clone()
                    self._batch_buffers["Ez"] = Ez.clone()
                
                # Update batch counter
                self._batch_counter += 1
                
                # If we haven't reached batch size yet, just store state
                if self._batch_counter < self.batch_size:
                    # Store current state for later batch processing
                    self._batch_buffers["Hx"].copy_(Hx)
                    self._batch_buffers["Hy"].copy_(Hy)
                    self._batch_buffers["Ez"].copy_(Ez)
                    return Ez
                
                # Reset counter and process batch
                self._batch_counter = 0
                
                # Process accumulated batch
                Hx, Hy = self._h_params[:2]
                self.kernel_fn(Hx, Hy, Ez, self._h_params[3], eps_r, 
                              dx, dy, dt, self._h_params[7], eps0)
                return Ez
            else:
                # For MPS, perform both H and E updates together to minimize kernel launches
                Hx, Hy = self._h_params[:2]
                self.kernel_fn(Hx, Hy, Ez, self._h_params[3], eps_r, 
                            dx, dy, dt, self._h_params[7], eps0)
                return Ez
            
        # No gradient version for better performance
        with torch.no_grad():
            return self.compiled_update_e_field(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0)

    def _update_h_fields_wrapper(self, Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0):
        """Wrapper with no_grad for H field updates."""
        with torch.no_grad():
            return self._update_h_fields_optimized(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)

    def _update_e_field_wrapper(self, Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0):
        """Wrapper with no_grad for E field updates."""
        with torch.no_grad():
            return self._update_e_field_optimized(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0)
            
    def _fused_update_fields(self, Hx, Hy, Ez, sigma, eps_r, dx, dy, dt, mu0, eps0):
        """Fused implementation that updates both H and E fields together for MPS performance."""
        # Use no_grad for performance
        with torch.no_grad():
            # Store original shape
            orig_shape = Ez.shape
            
            # Get parameters for H-field update from stored values
            h_params = self._h_params
            
            # --- H-field update ---
            # Constants for H-field update
            dt_mu0 = dt / mu0
            half_h = dt_mu0 * 0.5
            mu0_eps0 = mu0 / eps0
            
            # Get curl buffers (reuse memory)
            ce_x = self._ensure(self._h_curl_x_key, (Ez.shape[0], Ez.shape[1]-1))
            ce_y = self._ensure(self._h_curl_y_key, (Ez.shape[0]-1, Ez.shape[1]))
            
            # Compute curl E for H-field update (reusing buffers)
            # Create new tensor instead of using 'out=' for PyTorch compatibility
            ce_x.copy_(Ez[:, 1:] - Ez[:, :-1])
            ce_x.div_(dy)
            
            ce_y.copy_(Ez[1:, :] - Ez[:-1, :])
            ce_y.div_(dx)
            
            # Precompute sx, sy (magnetic conductivity)
            sx_buf = self._ensure("sx_buf", sigma[:, :-1].shape)
            sy_buf = self._ensure("sy_buf", sigma[:-1, :].shape)
            
            sx_buf.copy_(sigma[:, :-1] * mu0_eps0)
            sy_buf.copy_(sigma[:-1, :] * mu0_eps0)
            
            # Calculate update factors
            fx_buf = self._ensure("fx_buf", sx_buf.shape)
            fy_buf = self._ensure("fy_buf", sy_buf.shape)
            
            # Denominators - create new tensors instead of using out=
            denom_x = 1.0 + sx_buf * half_h
            denom_y = 1.0 + sy_buf * half_h
            
            # Compute factors
            fx_buf.copy_((1.0 - sx_buf * half_h) / denom_x)
            fy_buf.copy_((1.0 - sy_buf * half_h) / denom_y)
            
            # Source terms
            sx_src = dt_mu0 / denom_x
            sy_src = dt_mu0 / denom_y
            
            # Update H fields in-place
            Hx.mul_(fx_buf)
            Hx.sub_(sx_src * ce_x)
            
            Hy.mul_(fy_buf)
            Hy.add_(sy_src * ce_y)
            
            # Force synchronization between H and E updates if needed
            if self.device.type == "mps":
                torch.mps.synchronize()
            
            # --- E-field update ---
            # Constants for E-field update
            dt_eps0 = dt / eps0
            half_e = dt_eps0 * 0.5
            
            # Get/Create curl buffer
            curl = self._ensure("e_curl", Ez.shape)
            curl.zero_()
            
            # Create views for interior regions
            curl_inner = curl[1:-1, 1:-1]
            inner_ez = Ez[1:-1, 1:-1]
            inner_sigma = sigma[1:-1, 1:-1]
            inner_eps_r = eps_r[1:-1, 1:-1]
            
            # Calculate curl H components and store directly
            # Create new tensors instead of using out=
            hx_diff = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
            hy_diff = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
            
            # Combine components
            curl_inner.copy_(hx_diff - hy_diff)
            
            # Calculate update coefficients
            s_half_e = inner_sigma * half_e / inner_eps_r
            denom = 1.0 + s_half_e
            f1 = (1.0 - s_half_e) / denom
            f2 = -dt_eps0 / (inner_eps_r * denom)
            
            # Update Ez in-place
            inner_ez.mul_(f1)
            inner_ez.add_(f2 * curl_inner)
            
            # Force synchronization if needed
            if self.device.type == "mps":
                torch.mps.synchronize()
            
            return

    def _update_h_fields_optimized(self, Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0):
        """Optimized update method for H fields (with tensor reuse and avoiding contiguity issues)."""
        # constants
        dt_mu0 = dt / mu0
        half = dt_mu0 * 0.5
        mu0_eps0 = mu0 / eps0

        # Get buffers for intermediate results
        ce_x = self._ensure("ce_x", (Ez.shape[0], Ez.shape[1]-1))
        ce_y = self._ensure("ce_y", (Ez.shape[0]-1, Ez.shape[1]))
        
        # Compute curl components (avoiding out= parameters)
        ce_x.copy_(Ez[:, 1:] - Ez[:, :-1])
        ce_x.div_(dy)
        
        ce_y.copy_(Ez[1:, :] - Ez[:-1, :])
        ce_y.div_(dx)

        # Magnetic conductivity (avoiding out= parameters)
        sx = self._ensure("sx", (sigma.shape[0], sigma.shape[1]-1))
        sy = self._ensure("sy", (sigma.shape[0]-1, sigma.shape[1]))
        
        sx.copy_(sigma[:, :-1] * mu0_eps0)
        sy.copy_(sigma[:-1, :] * mu0_eps0)

        # Update factors (compute directly instead of using out=)
        denom_x = 1.0 + sx * half
        fx = (1.0 - sx * half) / denom_x
        sx_src = dt_mu0 / denom_x
        
        denom_y = 1.0 + sy * half
        fy = (1.0 - sy * half) / denom_y
        sy_src = dt_mu0 / denom_y

        # Perform the updates (in-place)
        Hx.mul_(fx)
        Hx.sub_(sx_src * ce_x)

        Hy.mul_(fy)
        Hy.add_(sy_src * ce_y)

        return Hx, Hy

    def _update_e_field_optimized(self, Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0):
        """Optimized update method for E field (with tensor reuse and avoiding contiguity issues)."""
        # constants
        dt_eps0 = dt / eps0
        half = dt_eps0 * 0.5

        # Pre-allocated curl buffer
        curl = self._ensure("curl", Ez.shape)
        curl.zero_()
        
        # Create views/slices for interior regions
        curl_inner = curl[1:-1, 1:-1]
        inner_ez = Ez[1:-1, 1:-1]
        inner_sigma = sigma[1:-1, 1:-1]
        inner_eps_r = eps_r[1:-1, 1:-1]
        
        # Compute curl components directly (avoiding out= parameters on slices)
        hx_diff = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        hy_diff = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
        
        # Set curl value (avoiding out= on slices)
        curl_inner.copy_(hx_diff - hy_diff)
        
        # Calculate coefficients (compute directly instead of using out=)
        s_half_e = inner_sigma * half / inner_eps_r
        denom = 1.0 + s_half_e
        f1 = (1.0 - s_half_e) / denom
        f2 = -dt_eps0 / (inner_eps_r * denom)
        
        # Apply in-place update
        inner_ez.mul_(f1)
        inner_ez.add_(f2 * curl_inner)
        
        return Ez
