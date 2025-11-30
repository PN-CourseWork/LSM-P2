"""HPC job management utilities.

Provides tools for:
- Generating job pack files for LSF/SLURM
- Submitting jobs to HPC schedulers
- Interactive job selection and preview
"""

from .jobgen import (
    load_config,
    generate_pack_lines,
    write_pack_file,
    get_job_output_dir,
)
from .submit import (
    interactive_generate,
    interactive_submit,
)

__all__ = [
    "load_config",
    "generate_pack_lines",
    "write_pack_file",
    "get_job_output_dir",
    "interactive_generate",
    "interactive_submit",
]
