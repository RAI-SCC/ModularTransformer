# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ModularTransformer'
copyright = '2024, Arvid Weyrauch, Pavel Zwerschke, Asena Özdemir'
author = 'Arvid Weyrauch, Pavel Zwerschke, Asena Özdemir'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
]

templates_path = ['_templates']
exclude_patterns = []

autoapi_dirs = ["../../modular_transformer"]
autoapi_python_class_content = "init"
# add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']
