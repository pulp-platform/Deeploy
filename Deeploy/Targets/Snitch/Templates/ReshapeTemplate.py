# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Reshape on Snitch reduces to a pointer alias (data is reinterpreted, not
# copied). The Generic implementation already covers this and now also sets
# the legacy `_alias` attribute required by the tiling extension, so we just
# re-export its referenceTemplate verbatim.
from Deeploy.Targets.Generic.Templates.ReshapeTemplate import referenceTemplate  # noqa: F401
