# Configuration file for the Sphinx documentation builder.
# This file controls the documentation properties of ngc-learn
# This file was created 4/19/2022
#
# @author Alexander Ororbia
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
#sys.path.insert(0, os.path.abspath('_build/html/'))

import sphinx_rtd_theme
#import jax
#import imageio
#import ngcsimlib
import ngclearn

# -- Project information -----------------------------------------------------

# general information about the project
project = "ngc-learn"
copyright = "The Neural Adaptive Computing Laboratory 2022"
author = 'Alexander Ororbia'

# The full version, including alpha/beta/rc tags
version = ngclearn.__version__
release = ngclearn.__version__ #'0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages'
]
# napoleon_google_docstring = False
# napoleon_use_param = False
# napoleon_use_ivar = True
myst_enable_extensions = [
    "amsmath",
    # "colon_fence",
    # "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    # "replacements",
    # "smartquotes",
    "strikethrough",
    # "substitution",
    #"tasklist",
]

pygments_style = "sphinx"

autodoc_mock_imports = ['jax', 'jaxlib', 'imageio', 'patchify', 'ngcsimlib']
#autodoc_mock_imports = ['scipy','sklearn','tensorflow','tensorflow_probability']
#autodoc_mock_imports = ['patchify', 'imageio', 'jax', 'scipy','sklearn','tensorflow','tensorflow_probability']

# sphinx api-doc variables
#apidoc_module_dir = "../ngclearn"
#apidoc_output_dir = "reference"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# suffixes of source filenames
source_suffix = [".rst", ".md"]

# the master toctree document
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
#exclude_patterns = ['Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_context = {
    "css_files": [
        "_static/css/theme.css",
        "_static/pygments.css",
        "_static/theme_overrides.css",  # override wide tables in RTD theme
    ],
}

html_logo = "images/ngc-learn-logo.png"
html_theme_options = {
    'logo_only': True,
    #'display_version': True,
}

# def setup(app):
#     RateCell.__name__ = 'RateCell'
#     #ngclearn.comopnents.neurons.rate_coded.RateCell.__name__ = 'RateCell'

# # Output file base name for HTML help builder.
# htmlhelp_basename = "ngclearndoc"
#
# # -- Options for LaTeX output ---------------------------------------------
# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     #
#     # 'papersize': 'letterpaper',
#     # The font size ('10pt', '11pt' or '12pt').
#     #
#     # 'pointsize': '10pt',
#     # Additional stuff for the LaTeX preamble.
#     #
#     # 'preamble': '',
#     # Latex figure (float) alignment
#     #
#     # 'figure_align': 'htbp',
# }
#
# # Grouping the document tree into LaTeX files. List of tuples
# # (source start file, target name, title,
# #  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     (master_doc, "ngclearn.tex", "ngc-learn Documentation", author, "manual"),
# ]
#
#
# # -- Options for manual page output ---------------------------------------
#
# # One entry per manual page. List of tuples
# # (source start file, name, description, authors, manual section).
# man_pages = [(master_doc, "ngclearn", "ngc-learn Documentation", [author], 1)]
