Clone the repository and install the package locally:
```bash
git clone https://github.com/QuentinWach/beamz
cd beamz
pip install .
```

For GPU acceleration, install with PyTorch support:
```bash
pip install ".[gpu]"
```

You can then run the first example:
```bash
python examples/0_resring.py
```

## Multi-Backend Support

Beamz now supports multiple computational backends for FDTD simulations:

- **NumPy**: Default CPU-based backend (no additional dependencies)
- **PyTorch**: GPU-accelerated backend (requires PyTorch installation)

To use the PyTorch backend for GPU acceleration:

```python
fdtd = FDTD(
    design=design,
    time=time,
    backend="torch",  # Use PyTorch backend
    backend_options={"device": "cuda"}  # Use GPU
)
```

To compare the performance of different backends, run:
```bash
python examples/backend_comparison.py
```