"""HPC job management utilities.

Provides tools for:
- Generating job pack files for LSF/SLURM
- Submitting jobs to HPC schedulers
- Interactive job selection and preview
- Scaling experiment job generation
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
    tui_generate_pack,
    tui_submit_pack,
    get_available_groups,
    get_pack_files,
    submit_pack,
)
from .scaling import (
    interactive_scaling,
    generate_scaling_jobs,
)

__all__ = [
    "load_config",
    "generate_pack_lines",
    "write_pack_file",
    "get_job_output_dir",
    "interactive_generate",
    "interactive_submit",
    "tui_generate_pack",
    "tui_submit_pack",
    "get_available_groups",
    "get_pack_files",
    "submit_pack",
    "interactive_scaling",
    "generate_scaling_jobs",
]
