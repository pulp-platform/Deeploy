# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

# Mark all tests in this module as deeploy_internal
pytestmark = pytest.mark.deeploy_internal


@pytest.mark.parametrize("platform", ["QEMU-ARM", "Siracusa", "MemPool", "Generic"])
def test_deeploy_state_serialization(platform):
    """Test that Deeploy state can be serialized and deserialized correctly."""
    script_dir = Path(__file__).parent
    cmd = [
        "python",
        str(script_dir / "deeployStateEqualityTest.py"),
        "-t",
        "./Tests/Models/CNN_Linear2",
        "-p",
        platform,
    ]
    result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

    assert result.returncode == 0, (f"State serialization test failed for platform {platform}\n"
                                    f"stdout: {result.stdout}\n"
                                    f"stderr: {result.stderr}")


@pytest.mark.parametrize("platform", ["QEMU-ARM", "Siracusa", "MemPool", "Generic"])
def test_memory_level_extension(platform):
    """Test memory level extension functionality."""
    script_dir = Path(__file__).parent
    cmd = [
        "python",
        str(script_dir / "testMemoryLevelExtension.py"),
        "-t",
        "./Tests/Models/CNN_Linear2",
        "-p",
        platform,
    ]
    result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

    assert result.returncode == 0, (f"Memory level extension test failed for platform {platform}\n"
                                    f"stdout: {result.stdout}\n"
                                    f"stderr: {result.stderr}")


class TestMemoryAllocation:
    """Test memory allocation strategies and constraints."""

    def test_minimalloc_sufficient_memory(self):
        """Test MiniMalloc strategy with sufficient L2 memory."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testMVP.py"),
            "-t",
            "Tests/Models/CCT/FP32/CCT_1_16_16_8",
            "-p",
            "Siracusa",
            "--defaultMemLevel=L2",
            "--l1=64000",
            "--l2=75000",
            "--memAllocStrategy=MiniMalloc",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Memory allocation test (MiniMalloc, L2=75000) failed\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")

    def test_minimalloc_insufficient_memory(self):
        """Test that MiniMalloc correctly fails with insufficient L2 memory."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testMVP.py"),
            "-t",
            "Tests/Models/CCT/FP32/CCT_1_16_16_8",
            "-p",
            "Siracusa",
            "--defaultMemLevel=L2",
            "--l1=64000",
            "--l2=60000",
            "--memAllocStrategy=MiniMalloc",
            "--shouldFail",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (
            f"Memory allocation test (MiniMalloc should fail, L2=60000) did not behave as expected\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}")

    def test_tetrisrandom_sufficient_memory(self):
        """Test TetrisRandom strategy with sufficient L2 memory."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testMVP.py"),
            "-t",
            "Tests/Models/CCT/FP32/CCT_1_16_16_8",
            "-p",
            "Siracusa",
            "--defaultMemLevel=L2",
            "--l1=64000",
            "--l2=90000",
            "--memAllocStrategy=TetrisRandom",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Memory allocation test (TetrisRandom, L2=90000) failed\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")

    def test_tetrisrandom_insufficient_memory(self):
        """Test that TetrisRandom correctly fails with insufficient L2 memory."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testMVP.py"),
            "-t",
            "Tests/Models/CCT/FP32/CCT_1_16_16_8",
            "-p",
            "Siracusa",
            "--defaultMemLevel=L2",
            "--l1=64000",
            "--l2=75000",
            "--memAllocStrategy=TetrisRandom",
            "--shouldFail",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (
            f"Memory allocation test (TetrisRandom should fail, L2=75000) did not behave as expected\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}")


class TestTilerExtension:
    """Test tiling extension functionality."""

    @pytest.mark.parametrize("test_path", [
        "./Tests/Models/CNN_Linear2",
        "./Tests/Models/CNN_Linear1",
        "./Tests/Kernels/Integer/MatMul/Regular",
        "./Tests/Kernels/Integer/MaxPool/Regular_2D",
    ])
    def test_tiler_basic(self, test_path):
        """Test that tiler can process various networks without L1 constraints."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testTilerExtension.py"),
            "-p",
            "Siracusa",
            "-t",
            test_path,
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Tiler extension test failed for {test_path}\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")

    @pytest.mark.parametrize("test_path", [
        "./Tests/Models/CNN_Linear2",
        "./Tests/Models/CNN_Linear1",
        "./Tests/Kernels/Integer/MatMul/Regular",
        "./Tests/Kernels/Integer/MaxPool/Regular_2D",
    ])
    def test_tiler_constrained_should_fail(self, test_path):
        """Test that tiler correctly fails when L1 memory is too small."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testTilerExtension.py"),
            "-p",
            "Siracusa",
            "-t",
            test_path,
            "--l1",
            "2000",
            "--shouldFail",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (
            f"Tiler extension test (should fail) did not behave as expected for {test_path}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}")

    @pytest.mark.parametrize("test_path", [
        "./Tests/Models/CNN_Linear2",
        "./Tests/Models/CNN_Linear1",
        "./Tests/Models/miniMobileNet",
        "./Tests/Models/miniMobileNetv2",
        "./Tests/Kernels/Integer/MatMul/Regular",
        "./Tests/Kernels/Integer/MaxPool/Regular_2D",
    ])
    def test_tiler_double_buffer(self, test_path):
        """Test tiler with double buffering enabled."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testTilerExtension.py"),
            "-p",
            "Siracusa",
            "-t",
            test_path,
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Tiler extension test (double buffer) failed for {test_path}\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")


def test_types():
    """Test Deeploy type system (serialization, equivalence, promotion)."""
    script_dir = Path(__file__).parent
    cmd = [
        "python",
        str(script_dir / "testTypes.py"),
    ]
    result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

    assert result.returncode == 0, (f"Types test failed\n"
                                    f"stdout: {result.stdout}\n"
                                    f"stderr: {result.stderr}")


class TestDebugTransformations:
    """Test debug and diagnostic transformations."""

    @pytest.mark.parametrize("platform", ["Generic", "Siracusa"])
    def test_print_input_output_transformation(self, platform):
        """Test print input/output transformation for debugging."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testPrintInputOutputTransformation.py"),
            "-p",
            platform,
            "-t",
            "./Tests/Models/CNN_Linear2",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Print I/O transformation test failed for platform {platform}\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")

    def test_debug_print_pass(self):
        """Test debug print pass transformation."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "testDebugPrintPass.py"),
            "-p",
            "Generic",
            "-t",
            "./Tests/Models/CNN_Linear2",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Debug print pass test failed\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")


def test_regex_matching():
    """Test regex matching utilities."""
    script_dir = Path(__file__).parent
    cmd = [
        "python",
        str(script_dir / "testRegexMatching.py"),
    ]
    result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

    assert result.returncode == 0, (f"Regex matching test failed\n"
                                    f"stdout: {result.stdout}\n"
                                    f"stderr: {result.stderr}")


class TestTypeInference:
    """Test type inference functionality with different input type configurations."""

    def test_type_inference_fail_all_int8(self):
        """Test that type inference correctly fails when all inputs are int8."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "generateNetwork.py"),
            "-p",
            "Generic",
            "-t",
            "./Tests/Others/TypeInference",
            "-v",
            "--input-type-map",
            "A=int8_t",
            "B=int8_t",
            "C=int8_t",
            "--input-offset-map",
            "A=0",
            "B=0",
            "C=0",
            "--shouldFail",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Type inference test (should fail with all int8) did not behave as expected\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")

    def test_type_inference_fail_incompatible_output(self):
        """Test that type inference correctly fails with incompatible output type."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "generateNetwork.py"),
            "-p",
            "Generic",
            "-t",
            "./Tests/Others/TypeInference",
            "-v",
            "--input-type-map",
            "A=int16_t",
            "B=int8_t",
            "C=int16_t",
            "--input-offset-map",
            "A=0",
            "B=0",
            "C=0",
            "--shouldFail",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (
            f"Type inference test (should fail with incompatible output) did not behave as expected\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}")

    def test_type_inference_pass(self):
        """Test that type inference succeeds with correct type configuration."""
        script_dir = Path(__file__).parent
        cmd = [
            "python",
            str(script_dir / "generateNetwork.py"),
            "-p",
            "Generic",
            "-t",
            "./Tests/Others/TypeInference",
            "-v",
            "--input-type-map",
            "A=int16_t",
            "B=int8_t",
            "C=int32_t",
            "--input-offset-map",
            "A=0",
            "B=0",
            "C=0",
        ]
        result = subprocess.run(cmd, cwd = script_dir, capture_output = True, text = True)

        assert result.returncode == 0, (f"Type inference test (should pass) failed\n"
                                        f"stdout: {result.stdout}\n"
                                        f"stderr: {result.stderr}")
