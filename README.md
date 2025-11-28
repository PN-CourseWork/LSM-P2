# MPI Poisson Solver

A modular framework for studying parallel performance of 3D Poisson equation solvers using MPI domain decomposition.

**Authors:** Alexander Elbæk Nielsen, Junriu Li, Philip Korsager Nickel
**Institution:** Technical University of Denmark, DTU Compute

## Documentation

📖 **[View Full Documentation](https://pn-coursework.github.io/LSM-P2/)** 
For local documentation, see [Building Documentation](#building-documentation) below.

## Quick Start

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```
## Building Documentation

Build the documentation locally:

```bash
uv run python main.py --build-docs
```

The documentation will be generated at `docs/build/html/index.html` and opened in your browser.


