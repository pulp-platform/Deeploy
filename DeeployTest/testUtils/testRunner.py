# ----------------------------------------------------------------------
#
# File: testRunner.py
#
# Last edited: 17.03.2023
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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

import argparse
import codecs
import os
import re
import subprocess
from typing import Literal, Tuple


# Source: https://stackoverflow.com/a/38662876
def escapeAnsi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


def prRed(skk):
    print("\033[91m{}\033[00m".format(skk))


def prGreen(skk):
    print("\033[92m{}\033[00m".format(skk))


def prBlue(skk):
    print("\033[94m{}\033[00m".format(skk))


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
        self.add_argument('--overwriteRecentState',
                          action = 'store_true',
                          help = 'Copy the recent state to the ./deeployStates folder\n')

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
                              help = 'Set L1 size\n')
            self.add_argument('--randomizedMemoryScheduler',
                              action = "store_true",
                              help = 'Enable randomized memory scheduler\n')
            self.add_argument('--profileTiling',
                              metavar = '<level>',
                              dest = 'profileTiling',
                              type = str,
                              default = None,
                              help = 'Profile tiling for a given memory level (eg. "L2")\n')

        self.args = None

    def parse_args(self, args = None, namespace = None) -> argparse.Namespace:
        self.args = super().parse_args(args, namespace)
        return self.args

    def generate_cmd_args(self) -> str:
        if self.args is None:
            self.args = super().parse_args()

        command = ""
        if self.args.verbose:
            command += " -v"
        if self.args.overwriteRecentState:
            command += " --overwriteRecentState"
        if self.args.debug:
            command += " --debug"

        if self.tiling_arguments:
            if self.args.defaultMemLevel:
                command += f" --defaultMemLevel={self.args.defaultMemLevel}"
            if self.args.doublebuffer:
                command += " --doublebuffer"
            if self.args.l1:
                command += f" --l1={self.args.l1}"
            if self.args.randomizedMemoryScheduler:
                command += " --randomizedMemoryScheduler"
            if self.args.profileTiling is not None:
                command += f" --profileTiling {self.args.profileTiling}"

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
        self._dir_toolchain = os.path.normpath(self._args.toolchain_install_dir)
        self._dir_build = f"{self._dir_gen_root}/build"
        self._dir_gen, self._dir_test, self._name_test = getPaths(self._args.dir, self._dir_gen_root)

        print("Generation Directory: ", self._dir_gen)
        print("Test Directory      : ", self._dir_test)
        print("Test Name           : ", self._name_test)

    def run(self,):
        prRed(f"################## Testing {self._dir_test} on {self._platform} Platform ##################")

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
        command += self._argument_parser.generate_cmd_args()

        if self._args.verbose >= 2:
            prBlue(f"[TestRunner] Generation Command: {command}")

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
        if self._args.verbose >= 2:
            prBlue(f"[TestRunner] Cmake Command: {command}")

        err = os.system(command)
        if err != 0:
            raise RuntimeError(f"Configuring cMake project failed on {self._dir_test}")

    def build_binary(self):
        command = f"$CMAKE --build {self._dir_build} --target {self._name_test}"

        if self._args.verbose >= 3:
            command = "VERBOSE=1 " + command
        if self._args.verbose >= 2:
            prBlue(f"[TestRunner] Building Command: {command}")

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

        if self._args.verbose >= 2:
            prBlue(f"[TestRunner] Simulation Command: {command}")

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
            prRed(f"❌ Found errors in {self._dir_test}")
            raise RuntimeError(f"Found an error in {self._dir_test}")
        else:
            prGreen(f"✅ No errors found in in {self._dir_test}")
