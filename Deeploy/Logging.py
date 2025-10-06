# ----------------------------------------------------------------------
#
# File: Logging.py
#
# Last edited: 22.08.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Philip Wiese, ETH Zurich
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

# Setup logging
import logging
from enum import Enum

import coloredlogs


class AnsiColorCode(Enum):
    LigthBlue = "\033[94m"
    Green = "\033[92m"
    Yellow = "\033[93m"
    Red = "\033[91m"
    Magenta = "\033[95m"
    Reset = "\033[0m"

    def __str__(self) -> str:
        return self.value


def color(msg: str, color: AnsiColorCode) -> str:
    return f"{color}{msg}{AnsiColorCode.Reset}"


SUCCESS_MARK = color("✔", AnsiColorCode.Green)
FAILURE_MARK = color("✘", AnsiColorCode.Red)

CONSOLE_LOG_FORMAT = "[%(name)s] %(message)s"
FILE_LOG_FORMAT = "[%(name)s] [%(module)-15s] %(message)s"
DETAILED_FILE_LOG_FORMAT = "[%(levelname)s] [%(name)s] [%(pathname)s:%(lineno)d] %(message)s"

DEFAULT_LOGGER = logging.getLogger("Deeploy")
DEFAULT_FMT = CONSOLE_LOG_FORMAT

# Install default logging if not already installed
if not DEFAULT_LOGGER.handlers:
    coloredlogs.install(level = 'INFO', logger = DEFAULT_LOGGER, fmt = DEFAULT_FMT)
