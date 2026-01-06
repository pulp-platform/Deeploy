

# How to use the DeeployTest PyTest Suite?

### Executing and Collecting Test Groups

The test suite is located in the `DeeployTest` folder, all commands below are assumed to be executed from the `DeeployTest` folder. The test suite is grouped with different markers, you can list the markers with `pytest --markers`. This will return something like:
```
@pytest.mark.generic: mark test as a Generic platform test
@pytest.mark.cortexm: mark test as a Cortex-M (QEMU-ARM) platform test
@pytest.mark.mempool: mark test as a MemPool platform test
```

You can run all test from a given mark group with `pytest -m <marker-name> -v`. Each platform has a given marker, if you want to run all tests from the generic platform, you can use `pytest -m generic -v`.

You can use boolean expressions on the markers to execute unions or intersections of markers. For instance, to execute only the kernel tests from the generic platform, one can use `pytest -m 'generic and kernels' -v`.

To display the tests captured by a given marker or expression, you can use the `--collect-only` flag. For instance, to list the kernel tests on the Siracusa with Neureka platform that are from L2 and single-buffered, I can use `pytest -m 'siracusa_neureka_tiled and kernels and l2 and singlebuffer' -v --collect-only`, which returns:

```
platform linux -- Python 3.10.0, pytest-9.0.2, pluggy-1.6.0 -- /usr/scratch/normandie/jungvi/micromamba/envs/deeploy/bin/python3.10
cachedir: .pytest_cache
rootdir: /scratch/jungvi/Deeploy/DeeployTest
configfile: pytest.ini
plugins: xdist-3.8.0
collected 378 items / 370 deselected / 8 selected

<Dir DeeployTest>
  <Module test_platforms.py>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer[testRequantizedLinear-16000-L2-singlebuffer]>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer[testPointwise-32000-L2-singlebuffer]>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer[testPointwiseConvBNReLU-32000-L2-singlebuffer]>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer[testPointwiseUnsignedWeights-32000-L2-singlebuffer]>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer_wmem[testRequantizedLinear-16000-L2-singlebuffer-wmem]>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer_wmem[testPointwise-32000-L2-singlebuffer-wmem]>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer_wmem[testPointwiseConvBNReLU-32000-L2-singlebuffer-wmem]>
    <Function test_siracusa_neureka_tiled_kernels_l2_singlebuffer_wmem[testPointwiseUnsignedWeights-32000-L2-singlebuffer-wmem]>
```

### Executing a Single Test

To run a single test, one can use the test identifier from the `--collect-only` output, for instance `pytest 'test_platforms.py::test_siracusa_neureka_tiled_kernels_l2_singlebuffer[testRequantizedLinear-16000-L2-singlebuffer]' -v`.

### Controlling Test Verbosity

By default, the test output is captured and displayed only if a test fails. If you want to see the captured output, use the `-s` flag. To increase the verbosity of the test, you can add more `v` to the `-v` flag, for instance, `-vvv` will display the commands executed during the test. You can filter the level of the messages from Python's built-in logging module with `--log-cli-level=<log-level>`. For instance, the following line captures only the commands executed by the tests:
```
pytest test_platforms.py -m "generic and kernels" -vvv --log-cli-level=DEBUG
```

### Parallelized Test Execution

You can run tests in parallel with the `-n` flag followed by the number of parallel threads. For instance, to run all generic tests with 16 threads, you can use:
```
pytest test_platforms.py -m generic -v -n 16
```

### Misc

When running `pytest -m <my-markers>` in a folder, PyTest will scan each file looking for tests. To speed up the detection you can specify the platform test file like `pytest test_platforms.py -m <my-markers>`.
