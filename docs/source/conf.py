"""Configuration file for the Sphinx documentation builder."""

import glob
import os
import shutil

project = "SimCATS-Datasets"
copyright = "2024 Forschungszentrum Jülich GmbH - Central Institute of Engineering, Electronics and Analytics (ZEA) - Electronic Systems (ZEA-2)"
author = "Fabian Hader, Fabian Fuchs, Karin Havemann, Sarah Fleitmann"
release = "2.5.0"  # also change in pyproject.toml

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # automatic documentation generation from docstrings
    "autoapi.extension",  # different automatic documentation generation from docstrings
    "sphinx_rtd_theme",  # readthedocs theme
    "sphinx.ext.napoleon",  # support google and numpy style docstrings
    "myst_nb",  # jupyter notebook support
]

exclude_patterns = []

# myst_nb
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# force notebook execution
nb_execution_mode = "off"

# autoapi
autoapi_dirs = ["../../simcats_datasets"]
autodoc_typehints = "description"
autodoc_typehints_format = "short"

python_use_unqualified_type_names = True
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    # "private-members",
    # "special-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "imported-members",
]

# We don't need the autoapi toctree entry, as we add it ourselves
autoapi_add_toctree_entry = False

# inherit python class parameter description from the __init__ method
autoapi_python_class_content = "both"

# set template folder
templates_path = ["_templates"]
autoapi_template_dir = "_templates"

# graphviz
graphviz_output_format = "svg"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

# for more options see https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    # Toc options
    "collapse_navigation": False,
    "navigation_depth": -1,
}

# copy notebooks and convert them
directory_path = "../../notebooks/*.ipynb"
notebook_paths = glob.glob(directory_path)

if not os.path.isdir("./notebooks"):
    os.mkdir("./notebooks")

for path in notebook_paths:
    new_path = path.replace("../..", "./")
    if os.path.isfile(new_path):
        os.remove(new_path)
    shutil.copyfile(path, new_path)

# copy readme
if not os.path.isdir("./misc"):
    os.mkdir("./misc")

shutil.copyfile("../../README.md", "./misc/README.md")
