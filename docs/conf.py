# docs/conf.py
"""Sphinx configuration for Aerosol3D documentation."""

import os
import sys

# Add src/ to path so autodoc can import the package
sys.path.insert(0, os.path.abspath("../src"))

project = "Aerosol3D"
copyright = "2026, Fan Zhang"
author = "Fan Zhang"
release = "0.5.0"
version = "0.5.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Aerosol3D Documentation"
html_short_title = "Aerosol3D"

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
# Mock imports not available in the docs build env: pyradtran (with the viz
# layer) is provided by the sibling pyRadtran repo, not a hard dependency here.
autodoc_mock_imports = ["pyradtran"]

# Autosummary
autosummary_generate = True

# Napoleon (Google/NumPy docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
