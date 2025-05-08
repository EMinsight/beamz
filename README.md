![HEADER](docs/header.png)

```bash
pip install beamz
```

Run an example:
```bash
python examples/0_resring.py
```

## Multi-Backend Support

Beamz now supports multiple computational backends for FDTD simulations:

- **NumPy**: Default CPU-based backend (no additional dependencies)
- **PyTorch**: Accelerated backend with support for:
  - NVIDIA GPUs via CUDA
  - Apple Silicon (M1/M2/M3/M4) via Metal Performance Shaders (MPS)
  - CPU fallback when no accelerator is available

### Using the PyTorch Backend

The PyTorch backend will automatically detect the best available acceleration:

```python
fdtd = FDTD(
    design=design,
    time=time,
    backend="torch",  # Use PyTorch backend
    backend_options={"device": "auto"}  # Automatically select best device
)
```

You can also explicitly select a specific device:

```python
# For NVIDIA GPU
fdtd = FDTD(design=design, time=time, backend="torch", backend_options={"device": "cuda"})

# For Apple Silicon via Metal
fdtd = FDTD(design=design, time=time, backend="torch", backend_options={"device": "mps"})

# For CPU processing
fdtd = FDTD(design=design, time=time, backend="torch", backend_options={"device": "cpu"})
```

To compare the performance of different backends, run:
```bash
python examples/backend_comparison.py
```