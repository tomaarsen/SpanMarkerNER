# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import shutil
from datetime import datetime
from pathlib import Path

from span_marker import __version__

project = "SpanMarker"
copyright = f"{datetime.today().year}, Tom Aarsen"
author = "Tom Aarsen"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",  # <- For Jupyter Notebook support
    "sphinx.ext.napoleon",  # <- For Google style docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "m2r2",  # <- For loading CHANGELOG.md
    "sphinx.ext.intersphinx",  # <- For linking to e.g. Torch docs
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "nltk_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"navigation_depth": 2}
# Required for the theme, used for linking to a specific tag in the website footer
html_context = {"github_user": "tomaarsen", "github_repo": "SpanMarkerNER"}

# -- Options for Apidoc
# This can be uncommented to "refresh" the api .rst files.
# Until then, I like being able to manually edit them slightly.
"""
def run_apidoc(app) -> None:
    '''Generage API documentation'''
    import better_apidoc

    better_apidoc.APP = app
    better_apidoc.main([
        'better-apidoc',
        '-t',
        os.path.join('docs', '_templates'),
        '--force',
        '--separate',
        '-o',
        os.path.join('docs', 'api'),
        os.path.join('span_marker'),
    ])


def setup(app) -> None:
    app.connect('builder-inited', run_apidoc)
"""

# -- Setting up the "Usage" requirements


def copy_notebooks() -> None:
    for filename in Path("../notebooks").glob("*.ipynb"):
        shutil.copy2(str(filename), "notebooks")


copy_notebooks()

# -- NBSphinx options
# Do not execute the notebooks when building the docs
nbsphinx_execute = "never"

nbsphinx_prolog = """
.. raw:: html

    <div class="open-in-colab__wrapper">
    <a href="https://colab.research.google.com/github/tomaarsen/SpanMarkerNER/blob/main/{{ env.doc2path(env.docname, base=False) }}" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" style="display: inline; margin: 0" alt="Open In Colab"/>
    </a>
    </div>
"""

# -- Options for Autodoc
# Put the Python 3.5+ type hint in the parameters list
autodoc_typehints = "description"

# autodoc_class_signature = "separated"

autodoc_inherit_docstrings = False

# -- Options for Intersphinx
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable", None),  # <- For hyperparameter search method
}
