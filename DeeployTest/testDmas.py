# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Dma(Enum):
    Mchan = "Mchan"
    L3 = "L3"
    Snitch = "Snitch"

    def __str__(self) -> str:
        return self.value


testRunnerMap = {
    Dma.Mchan: "testRunner_siracusa_mchandma.py",
    Dma.L3: "testRunner_siracusa_l3dma.py",
    Dma.Snitch: "testRunner_snitch_dma.py",
}


@dataclass
class Config:
    dma: Dma
    inputShape: Tuple[int, ...]
    tileShape: Tuple[int, ...]
    nodeCount: int
    dataType: str
    doublebuffer: bool

    @property
    def cmd(self) -> str:
        assert dma in testRunnerMap, f"{self.dma} missing its own testRunner mapping"
        testRunner = testRunnerMap[self.dma]
        cmd = [f"python {testRunner}", f"-t test{self.dma}", "-DNUM_CORES=8"]
        cmd.append(f"--input-shape {' '.join(str(x) for x in self.inputShape)}")
        cmd.append(f"--tile-shape {' '.join(str(x) for x in self.tileShape)}")
        cmd.append(f"--node-count {self.nodeCount}")
        cmd.append(f"--type {self.dataType}")
        if self.doublebuffer:
            cmd.append("--doublebuffer")
        return " ".join(cmd)

    @property
    def short_repr(self) -> str:
        return f"{self.dma} - in:{self.inputShape}, " \
            f"tile:{self.tileShape}, " \
            f"n:{self.nodeCount}, " \
            f"ty:{self.dataType}" + \
            (", DB" if self.doublebuffer else "")

    def __repr__(self) -> str:
        return f"""
        - input shape: {inputShape}
        - tile shape: {tileShape}
        - node count: {nodeCount}
        - data type: {dataType}
        - doublebuffering: {doublebuffer}
        - dma: {dma}
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action = "store_true", default = False)
    parser.add_argument("--quiet", action = "store_true", default = False)
    parser.add_argument("--dma", action = 'append', type = Dma, choices = list(Dma), default = [])
    args = parser.parse_args()

    # input shape, tile shape, node count, data type
    test_shapes_and_more = [
        ((10, 10), (10, 10), 1, "uint8_t"),
        ((10, 10), (10, 4), 1, "uint8_t"),
        ((10, 10), (10, 4), 1, "uint16_t"),
        ((10, 10), (10, 4), 1, "uint32_t"),
        ((10, 10), (3, 4), 1, "uint32_t"),
        ((10, 10), (3, 4), 2, "uint32_t"),
        ((10, 10, 10), (2, 3, 4), 1, "uint8_t"),
        ((10, 10, 10, 10), (2, 3, 5, 4), 1, "uint8_t"),
        ((10, 10, 10, 10), (2, 3, 5, 4), 1, "uint32_t"),
        ((10, 10, 10, 10, 10), (2, 3, 5, 7, 4), 1, "uint8_t"),
    ]
    is_doublebuffers = [True, False]

    if len(args.dma) > 0:
        dmas = args.dma
    else:
        dmas = list(Dma)

    failures = []
    timeouts = []

    for testShape, doublebuffer, dma in itertools.product(test_shapes_and_more, is_doublebuffers, dmas):
        inputShape, tileShape, nodeCount, dataType = testShape
        cfg = Config(dma, inputShape, tileShape, nodeCount, dataType, doublebuffer)

        if args.verbose:
            print(f"Testing {cfg.dma} DMA with followig configuration:" + repr(cfg))
            print(f"Running command:\n{cfg.cmd}\n")

        try:
            if args.verbose:
                out, err = subprocess.STDOUT, subprocess.STDOUT
            else:
                out, err = subprocess.DEVNULL, subprocess.DEVNULL
            subprocess.run(cfg.cmd, shell = True, check = True, stdout = out, stderr = err,
                           timeout = 10)  # 10min timeout
            if not args.quiet:
                print(f"{cfg.short_repr} - OK")
        except subprocess.CalledProcessError:
            failures.append(cfg)
            if not args.quiet:
                print(f"{cfg.short_repr} - FAIL")
        except subprocess.TimeoutExpired:
            timeouts.append(cfg)
            # LMACAN: Not cross-platform but gvsoc keeps going after timeout
            subprocess.run("pkill gvsoc_launcher", shell = True)
            if not args.quiet:
                print(f"{cfg.short_repr} - TIMEOUT")

    if not args.quiet and len(failures) > 0:
        print(f"\nError: {len(failures)} tests failed.\nRerun with these commands:\n" +
              "\n".join(cfg.cmd for cfg in failures))
    if not args.quiet and len(timeouts) > 0:
        print(f"\nError: {len(timeouts)} tests timed out.\nRerun with these commands:\n" +
              "\n".join(cfg.cmd for cfg in timeouts))

    if len(failures) > 0 or len(timeouts) > 0:
        exit(1)
