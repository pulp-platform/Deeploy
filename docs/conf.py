# ----------------------------------------------------------------------
#
# File: conf.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

project = 'Deeploy'
copyright = '2024, Moritz Scherer, Philip Wiese, Luka Macan, Victor Jung'
author = 'Moritz Scherer, Philip Wiese, Luka Macan, Victor Jung'
release = '2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
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
html_static_path = ['_static']

# -- Options for HTML templates ------------------------------------------------

# Extract branch name from git
branch = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()

html_context = {
    'current_version':
        f"{branch}",
    'versions': [["main", f"https://pulp-platform.github.io/Deeploy"],
                 ["devel", f"https://pulp-platform.github.io/Deeploy/branch/devel"]],
}
