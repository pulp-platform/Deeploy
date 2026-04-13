import sys

from testUtils.deeployRunner import main

if __name__ == "__main__":
    sys.exit(
        main(
            default_platform = "Spatz",
            default_simulator = "gvsoc",
            tiling_enabled = True,
        )
    )
