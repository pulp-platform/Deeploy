# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import codecs
import os
import re
import shutil
import subprocess
from typing import Literal, Tuple

import coloredlogs

from Deeploy.Logging import DEFAULT_FMT
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Logging import DETAILED_FILE_LOG_FORMAT, FAILURE_MARK, SUCCESS_MARK


# Source: https://stackoverflow.com/a/38662876
def escapeAnsi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


def getPaths(path_test: str, gendir_name: str) -> Tuple[str, str]:

    dir_test = os.path.normpath(path_test)
    dir_abs = os.path.abspath(dir_test)
    test_name = dir_abs.split(os.sep)[-1]
    # Check if path is inside in some child folder of the script location

    # Get the absolute path of the script location
    scriptPath = os.path.realpath(__file__)

    # Get absolute path path to folder of the script location parent directory
    scriptDir = os.path.dirname(os.path.dirname(scriptPath))

    # Check if the path is inside the script location
    if scriptDir in dir_abs:
        dir_gen = os.path.join(scriptDir, gendir_name, dir_test)
        dir_gen = os.path.normpath(dir_gen)
    else:
        dir_gen = os.path.join(scriptDir, gendir_name, test_name)
        dir_gen = os.path.normpath(dir_gen)

        print(f"Path is not inside the script location. Using gendir path {dir_gen}")

    return dir_gen, dir_test, test_name


def cmake_str(arg_str):
    return "-D" + codecs.decode(str(arg_str), 'unicode_escape')


class _ArgumentDefaultMetavarTypeFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):

    def __init__(self, prog: str, indent_increment: int = 2, max_help_position: int = 100, width = None) -> None:
        super().__init__(prog, indent_increment, max_help_position, width)


class TestGeneratorArgumentParser(argparse.ArgumentParser):

    def __init__(self, description = None):

        formatter = _ArgumentDefaultMetavarTypeFormatter

        if description is None:
            super().__init__(description = "Test Utility.", formatter_class = formatter)
        else:
            super().__init__(description = description, formatter_class = formatter)

        self.add_argument('-t',
                          metavar = '<dir>',
                          dest = 'dir',
                          type = str,
                          required = True,
                          help = 'Set the regression test\n')
        self.add_argument('-p',
                          metavar = '<platform>',
                          dest = 'platform',
                          type = str,
                          required = True,
                          help = 'Choose the target Platform\n')
        self.add_argument('-d',
                          metavar = '<dir>',
                          dest = 'dumpdir',
                          type = str,
                          default = './TestFiles',
                          help = 'Set the output dump folder\n')
        self.add_argument('-v', action = 'count', dest = 'verbose', default = 0, help = 'Increase verbosity level\n')

        self.args = None

    def parse_args(self, args = None, namespace = None) -> argparse.Namespace:
        self.args = super().parse_args(args, namespace)

        # Install logger based on verbosity level
        if self.args.verbose > 2:
            coloredlogs.install(level = 'DEBUG', logger = log, fmt = DETAILED_FILE_LOG_FORMAT)
        elif self.args.verbose > 1:
            coloredlogs.install(level = 'DEBUG', logger = log, fmt = DEFAULT_FMT)
        elif self.args.verbose > 0:
            coloredlogs.install(level = 'INFO', logger = log, fmt = DEFAULT_FMT)
        else:
            coloredlogs.install(level = 'WARNING', logger = log, fmt = DEFAULT_FMT)
        return self.args


class TestRunnerArgumentParser(argparse.ArgumentParser):

    def __init__(self, tiling_arguments: bool, description = None):

        formatter = _ArgumentDefaultMetavarTypeFormatter

        if description is None:
            super().__init__(description = "Deeploy Code Generation Utility.", formatter_class = formatter)
        else:
            super().__init__(description = description, formatter_class = formatter)

        self.tiling_arguments = tiling_arguments

        self.add_argument('-t',
                          metavar = '<dir>',
                          dest = 'dir',
                          type = str,
                          required = True,
                          help = 'Set the regression test\n')
        self.add_argument('-v', action = 'count', dest = 'verbose', default = 0, help = 'Increase verbosity level\n')
        self.add_argument('-D',
                          dest = 'cmake',
                          action = 'extend',
                          nargs = "*",
                          type = cmake_str,
                          help = "Create or update a cmake cache entry\n")
        self.add_argument('--debug',
                          dest = 'debug',
                          action = 'store_true',
                          default = False,
                          help = 'Enable debugging mode.\n')
        self.add_argument('--skipgen',
                          dest = 'skipgen',
                          action = 'store_true',
                          default = False,
                          help = 'Skip network generation\n')
        self.add_argument('--skipsim',
                          dest = 'skipsim',
                          action = 'store_true',
                          default = False,
                          help = 'Skip network simulation\n')
        self.add_argument('--profileUntiled',
                          '--profile-untiled',
                          dest = 'profileUntiled',
                          action = 'store_true',
                          default = False,
                          help = 'Enable untiled profiling (Siracusa only)\n')
        self.add_argument('--toolchain',
                          metavar = '<LLVM|GCC>',
                          dest = 'toolchain',
                          type = str,
                          default = "LLVM",
                          help = 'Pick compiler toolchain\n')
        self.add_argument('--toolchain_install_dir',
                          metavar = '<dir>',
                          dest = 'toolchain_install_dir',
                          type = str,
                          default = os.environ.get('LLVM_INSTALL_DIR'),
                          help = 'Pick compiler install dir\n')
        self.add_argument('--input-type-map',
                          nargs = '*',
                          default = [],
                          type = str,
                          help = '(Optional) mapping of input names to data types. '
                          'If not specified, types are inferred from the input data. '
                          'Example: --input-type-map input_0=int8_t input_1=float32_t ...')
        self.add_argument('--input-offset-map',
                          nargs = '*',
                          default = [],
                          type = str,
                          help = '(Optional) mapping of input names to offsets. '
                          'If not specified, offsets are set to 0. '
                          'Example: --input-offset-map input_0=0 input_1=128 ...')

        if self.tiling_arguments:
            self.add_argument('--defaultMemLevel',
                              metavar = '<level>',
                              dest = 'defaultMemLevel',
                              type = str,
                              default = "L2",
                              help = 'Set default memory level\n')
            self.add_argument('--doublebuffer', action = 'store_true', help = 'Enable double buffering\n')
            self.add_argument('--l1',
                              metavar = '<size>',
                              dest = 'l1',
                              type = int,
                              default = 64000,
                              help = 'Set L1 size in bytes.\n')
            self.add_argument('--l2',
                              metavar = '<size>',
                              dest = 'l2',
                              type = int,
                              default = 1024000,
                              help = 'Set L2 size in bytes.\n')
            self.add_argument('--randomizedMemoryScheduler',
                              action = "store_true",
                              help = 'Enable randomized memory scheduler\n')
            self.add_argument('--profileTiling', action = 'store_true', help = 'Enable tiling profiling\n')
            self.add_argument('--memAllocStrategy',
                              metavar = 'memAllocStrategy',
                              dest = 'memAllocStrategy',
                              type = str,
                              default = "MiniMalloc",
                              help = """Choose the memory allocation strategy, possible values are:
                            - TetrisRandom: Randomly sample an placement schedule (order) for the Tetris Memory Allocation.
                            - TetrisCo-Opt: Co-optimize the placement schedule with the tiling solver (works best with random-max solver strategy).
                            - MiniMalloc: Use SotA static memory allocator from https://dl.acm.org/doi/10.1145/3623278.3624752
                        """)
            self.add_argument('--searchStrategy',
                              metavar = 'searchStrategy',
                              dest = 'searchStrategy',
                              type = str,
                              default = "random-max",
                              help = """Choose the search strategy for the CP solver:
                            - random-max: Initalize the permutation matrix variables randomly and initalize all other variables at their maximal value. This is recommended and lead to better solutions.
                            - max: Initalize all variables at their maximal value.
                            - min: Initalize all variables at their minimal value.
                        """)
            self.add_argument(
                '--plotMemAlloc',
                action = 'store_true',
                help = 'Turn on plotting of the memory allocation and save it in the deeployState folder\n')

        self.args = None

    def parse_args(self, args = None, namespace = None) -> argparse.Namespace:
        self.args = super().parse_args(args, namespace)
        return self.args

    def generate_cmd_args(self) -> str:
        if self.args is None:
            self.args = super().parse_args()

        command = ""
        if self.args.verbose:
            command += " -" + "v" * self.args.verbose
        if self.args.debug:
            command += " --debug"
        if hasattr(self.args, 'profileUntiled') and self.args.profileUntiled:
            command += " --profileUntiled"
        if self.args.input_type_map:
            command += " --input-type-map " + " ".join(self.args.input_type_map)
        if self.args.input_offset_map:
            command += " --input-offset-map " + " ".join(self.args.input_offset_map)

        if self.tiling_arguments:
            if self.args.defaultMemLevel:
                command += f" --defaultMemLevel={self.args.defaultMemLevel}"
            if self.args.doublebuffer:
                command += " --doublebuffer"
            if self.args.l1:
                command += f" --l1={self.args.l1}"
            if self.args.l2:
                command += f" --l2={self.args.l2}"
            if self.args.randomizedMemoryScheduler:
                command += " --randomizedMemoryScheduler"
            if self.args.profileTiling:
                command += f" --profileTiling"
            if self.args.memAllocStrategy:
                command += f" --memAllocStrategy={self.args.memAllocStrategy}"
            if self.args.plotMemAlloc:
                command += f" --plotMemAlloc"
            if self.args.searchStrategy:
                command += f" --searchStrategy={self.args.searchStrategy}"

        return command

    def cmake_args(self) -> str:
        if self.args is None:
            self.args = super().parse_args()

        cmake_args = " ".join(self.args.cmake) if self.args.cmake is not None else ""
        return cmake_args


class TestRunner():

    def __init__(self,
                 platform: str,
                 simulator: Literal['gvsoc', 'banshee', 'qemu', 'vsim', 'vsim.gui', 'host', 'none'],
                 tiling: bool,
                 argument_parser: TestRunnerArgumentParser,
                 gen_args: str = "",
                 cmake_args: str = ""):

        if simulator not in ['gvsoc', 'banshee', 'qemu', 'vsim', 'vsim.gui', 'host', 'none']:
            raise ValueError(
                f"Invalid emulator {simulator} (valid options are 'gvsoc', 'banshee', 'qemu', 'vsim', 'vsim.gui', 'host', 'none')!"
            )

        if tiling is not argument_parser.tiling_arguments:
            raise ValueError("Specified argument parser without tile arguments for tiling test or vice versa!")

        self._platform = platform
        self._simulator = simulator
        self._tiling = tiling

        self._argument_parser = argument_parser
        self._args = self._argument_parser.parse_args()

        self.cmake_args = cmake_args
        self.gen_args = gen_args

        self._dir_gen_root = f'TEST_{platform.upper()}'
        assert self._args.toolchain_install_dir is not None, f"Environment variable LLVM_INSTALL_DIR is not set"
        self._dir_toolchain = os.path.normpath(self._args.toolchain_install_dir)
        self._dir_build = f"{self._dir_gen_root}/build"
        self._dir_gen, self._dir_test, self._name_test = getPaths(self._args.dir, self._dir_gen_root)

        if "CMAKE" not in os.environ:
            if self._args.verbose >= 1:
                log.error(f"[TestRunner] CMAKE environment variable not set. Falling back to cmake")
            assert shutil.which(
                "cmake"
            ) is not None, "CMake not found. Please check that CMake is installed and available in your system’s PATH, or set the CMAKE environment variable to the full path of your preferred CMake executable."
            os.environ["CMAKE"] = "cmake"

        print("Generation Directory: ", self._dir_gen)
        print("Test Directory      : ", self._dir_test)
        print("Test Name           : ", self._name_test)

    def run(self,):
        log.info(f"################## Testing {self._dir_test} on {self._platform} Platform ##################")

        if self._args.skipgen is False:
            self.generate_test()

        self.configure_cmake_project()

        self.build_binary()

        if self._args.skipsim is False:
            self.run_simulation()

    def generate_test(self):
        if self._tiling is True:
            generation_script = "testMVP.py"
        else:
            generation_script = "generateNetwork.py"

        command = f"python {generation_script} -d {self._dir_gen} -t {self._dir_test} -p {self._platform} {self.gen_args}"

        if self._platform in ["Siracusa", "Siracusa_w_neureka"]:
            command += f" --cores={self._args.cores}"

        command += self._argument_parser.generate_cmd_args()

        log.debug(f"[TestRunner] Generation Command: {command}")

        err = os.system(command)
        if err != 0:
            raise RuntimeError(f"generate Network failed on {self._args.dir}")

    def configure_cmake_project(self):
        self.cmake_args += self._argument_parser.cmake_args()

        if self._simulator == 'banshee':
            self.cmake_args += " -D banshee_simulation=ON"
        else:
            self.cmake_args += " -D banshee_simulation=OFF"

        if self._simulator == 'gvsoc':
            self.cmake_args += " -D gvsoc_simulation=ON"
        else:
            self.cmake_args += " -D gvsoc_simulation=OFF"

        command = f"$CMAKE -D TOOLCHAIN={self._args.toolchain} -D TOOLCHAIN_INSTALL_DIR={self._dir_toolchain} -D GENERATED_SOURCE={self._dir_gen} -D platform={self._platform} {self.cmake_args} -B {self._dir_build} -D TESTNAME={self._name_test} .."

        if self._args.verbose >= 3:
            command = "VERBOSE=1 " + command

        log.debug(f"[TestRunner] Cmake Command: {command}")

        err = os.system(command)
        if err != 0:
            raise RuntimeError(f"Configuring cMake project failed on {self._dir_test}")

    def build_binary(self):
        command = f"$CMAKE --build {self._dir_build} --target {self._name_test}"

        if self._args.verbose >= 3:
            command = "VERBOSE=1 " + command

        log.debug(f"[TestRunner] Building Command: {command}")

        err = os.system(command)
        if err != 0:
            raise RuntimeError(f"Building cMake project failed on {self._dir_test}")

    def run_simulation(self, out_file = 'out.txt'):
        if self._simulator == 'none':
            raise RuntimeError("No simulator specified!")

        if self._simulator == 'host':
            command = f"{self._dir_build}/bin/{self._name_test}"
        else:
            command = f"$CMAKE --build {self._dir_build} --target {self._simulator}_{self._name_test}"

        if self._args.verbose >= 3:
            command = "VERBOSE=1 " + command

        if self._simulator == 'banshee':
            if self._args.verbose == 1:
                command = "BANSHEE_LOG=warn " + command
            if self._args.verbose == 2:
                command = "BANSHEE_LOG=info " + command
            if self._args.verbose >= 3:
                command = "BANSHEE_LOG=debug " + command

        log.debug(f"[TestRunner] Simulation Command: {command}")

        process = subprocess.Popen([command],
                                   stdout = subprocess.PIPE,
                                   stderr = subprocess.STDOUT,
                                   shell = True,
                                   encoding = 'utf-8')

        fileHandle = open(out_file, 'a', encoding = 'utf-8')
        fileHandle.write(
            f"################## Testing {self._dir_test} on {self._platform} Platform ##################\n")

        result = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                result += output
                fileHandle.write(f"{escapeAnsi(output)}")

        fileHandle.write("")
        fileHandle.close()

        if "Errors: 0 out of " not in result:
            log.error(f"{FAILURE_MARK} Found errors in {self._dir_test}")
            raise RuntimeError(f"Found an error in {self._dir_test}")
        else:
            log.info(f"{SUCCESS_MARK} No errors found in in {self._dir_test}")
