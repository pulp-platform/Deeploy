# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath('../'))

project = 'Deeploy'
copyright = '2024, Moritz Scherer, Philip Wiese, Luka Macan, Victor Jung'
author = 'Moritz Scherer, Philip Wiese, Luka Macan, Victor Jung'
release = '2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser', 'sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.autosummary',
    'sphinx_favicon'
]
autosummary_generate = True
napoleon_use_ivar = True
add_module_names = True
autodoc_member_order = "bysource"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "*flycheck_*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'style_nav_header_background': 'white',
    'logo_only': True,
}
html_static_path = ['_static']
html_logo = '_static/DeeployBannerGreen-640x-320.png'

# -- Options for HTML templates ------------------------------------------------

# Extract branch name from git

# Try to get branch name
branch = None
try:
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"],
                                     stderr = subprocess.DEVNULL).decode().strip()
except subprocess.CalledProcessError:
    pass  # Not a branch, maybe a tag?

# Try to get tag name if branch not available
tag = None
if not branch:
    try:
        tag = subprocess.check_output(["git", "describe", "--tags", "--exact-match"],
                                      stderr = subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        pass  # Not on a tag either

# Fallback
current = branch or tag or "unknown"

html_context = {
    'current_version':
        current,
    'versions': [
        ["main", "https://pulp-platform.github.io/Deeploy"],
        ["devel", "https://pulp-platform.github.io/Deeploy/branch/devel"],
        ["v0.2.0", "https://pulp-platform.github.io/Deeploy/tag/v0.2.0"],
        ["v0.2.1", "https://pulp-platform.github.io/Deeploy/tag/v0.2.1"],
    ],
}

# -- Options for myst_parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = ["html_image", "dollarmath", "linkify", "replacements"]

# -- Options for sphinx_favicon ------------------------------------------------
favicons = [
    {
        "href": "DeeployIconGreen.svg"
    },
    {
        "href": "DeeployIconGreen-32x32.png"
    },
    {
        "href": "DeeployIconGreen-64x64.png"
    },
]
