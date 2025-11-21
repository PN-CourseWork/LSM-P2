"""Sphinx configuration for numutils package documentation."""

import os
import sys

# Add src directory to path for module imports
src_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "src"))
sys.path.insert(0, src_path)

repo_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, repo_root)

# -- Project information -----------------------------------------------------

project = "MPI 3D Poisson Solver"
copyright = "2025, Alexander Elbæk Nielsen, Junriu Li, Philip Korsager Nickel, DTU Compute"
author = "Alexander Elbæk Nielsen, Junriu Li, Philip Korsager Nickel"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

root_doc = "index"
source_suffix = {".rst": "restructuredtext"}

# -- Autodoc configuration ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# Mock heavy runtime dependencies
# autodoc_mock_imports = ["numba", "pyarrow", "matplotlib"]  # Disabled - causing import issues

# -- Numpydoc configuration --------------------------------------------------

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "default", "of"}

# Use numpydoc's enhanced autosummary templates
templates_path = ["_templates"]
numpydoc_use_plots = False  # Don't auto-generate plots from Examples

# -- Sphinx Gallery configuration --------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "../../Experiments",  # Path to example scripts
    "gallery_dirs": "example_gallery",  # Output directory for gallery
    "filename_pattern": "/plot_",  # Pattern to match which scripts to execute
    "download_all_examples": False,  # No download buttons
    "remove_config_comments": True,  # Clean up notebook outputs
    "abort_on_example_error": False,  # Continue if examples fail
    "plot_gallery": True,  # Enable plot gallery generation
    "capture_repr": ("_repr_html_", "__repr__"),  # Capture output representations
    "matplotlib_animations": True,  # Support matplotlib animations
    # Remove Jupyter cell markers (# %%) from rendered output
    "first_notebook_cell": None,  # Don't add a first cell
    "last_notebook_cell": None,  # Don't add a last cell
    "notebook_images": False,  # Don't embed images in notebooks
    # Cross-referencing: Create "Examples using X" in API docs
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("Poisson",),  # Generate backreferences for our package
    "inspect_global_variables": True,  # Detect classes/functions used in examples
    # Make code clickable: Link to API docs when code mentions Poisson functions
    "reference_url": {
        "Poisson": None,  # None = use local docs (not external URL)
    },
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
}

# -- HTML output options -----------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "MPI Poisson Solver"
html_static_path = ["_static"]
html_show_sourcelink = False  # Hide "Show Source" link
html_css_files = ["custom.css"]  # Custom CSS for hiding download buttons

html_theme_options = {
    "github_url": "https://github.com/PhilipNickel-DTU-CourseWork/LSM-P2",
}
