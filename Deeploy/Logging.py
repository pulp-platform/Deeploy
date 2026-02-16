# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
