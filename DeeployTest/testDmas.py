import itertools
import subprocess
from typing import Tuple


def test(dma: str, inputShape: Tuple[int, ...], tileShape: Tuple[int, ...], nodeCount: int, dataType: str,
         doublebuffer: bool):
    cfg_str = f"""
    - input shape: {inputShape}
    - tile shape: {tileShape}
    - node count: {nodeCount}
    - data type: {dataType}
    - doublebuffering: {doublebuffer}
    - dma: {dma}
    """

    print(f"test{dma}: Testing {dma} with followig configuration:" + cfg_str)

    cmd = [f"python testRunner_{dma}.py", f"-t test{dma}", "-DNUM_CORES=8"]
    cmd.append(f"--input-shape {' '.join(str(x) for x in inputShape)}")
    cmd.append(f"--tile-shape {' '.join(str(x) for x in tileShape)}")
    cmd.append(f"--node-count {nodeCount}")
    cmd.append(f"--type {dataType}")
    if doublebuffer:
        cmd.append("--doublebuffer")

    full_cmd = " ".join(cmd)

    print(f"Running command:\n{full_cmd}\n")

    try:
        subprocess.run(full_cmd, shell = True, check = True)
    except subprocess.CalledProcessError:
        print(f"test{dma}: Failed test:" + cfg_str)
        print(f"Rerun with command:\n{full_cmd}")
        exit(-1)


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
dmas = ["MchanDma", "L3Dma", "SnitchDma"]

for testShape, doublebuffer, dma in itertools.product(test_shapes_and_more, is_doublebuffers, dmas):
    inputShape, tileShape, nodeCount, dataType = testShape
    test(dma, inputShape, tileShape, nodeCount, dataType, doublebuffer)
