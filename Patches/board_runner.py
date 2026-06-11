"""
Helper for running on gap9_v2 boards
"""

#
# Copyright (C) 2022 GreenWaves Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Authors: Germain Haugou, GreenWaves Technologies (germain.haugou@greenwaves-technologies.com)
#

import random
import sys
import time
import os.path
import argparse
import threading
import pexpect
import gapylib.target
from elftools.elf.elffile import ELFFile
import ppk2_api.ppk2_api
try:
    from pyhwtestbench import pyHwTestBench
    import pandas as pd
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass


class Testbench():
    """
    Helper class for using Gap HW testbench.

    This is based on pyhwtestbench package providede with HW testbench firmware.
    """

    def __init__(self, power_plot=False, power_all=False):
        self.power_plot = power_plot
        self.power_all = power_all
        self.hwtestbench = pyHwTestBench()
        self.hwtestbench.powerOffBoard()
        self.hwtestbench.resetTester()
        self.hwtestbench.mapPad(40,"GPIO")
        self.hwtestbench.gpioDir(40,"IN")

        self.memid = 5
        samplerate = 20000
        nb_sample_per_in = samplerate * 2

        self.buffsize = self.hwtestbench.powerMeasurmentConfig(memid=self.memid,
            nb_sample_per_in=nb_sample_per_in, v1v8main=True, v1v8=True, v0v65=True,
            imain=True, idcdc=True, ivddio=True, imems=False, icore=True, gpio_capture=True,
            samplerate=samplerate)

        self.hwtestbench.bootTested("JTAG")

    def start(self):
        """Start power sampling.

        This will notify the testbench that power sampling should start.
        """
        self.hwtestbench.powerMeasurmentStart(self.memid)

    def stop(self):
        """Stop power sampling.

        This currently waits for the HW testbench to finish the power measurement which has
        a fixed duration.
        This should soon notify the testbench to stop.
        Then this is parsing the samples, check measurement periods delimited by GPIO being at 1
        and print the results to the terminal.
        """
        self.hwtestbench.waitPowerMeasurmentEnd()

        samples = self.hwtestbench.powerMeasurmentGetDataFrame(self.memid, self.buffsize,
            filterwindow=0, gpio_list=[40])

        self.__process_samples(samples)


    def __process_samples(self, samples):
        samples_gpio = samples['gpio 40']

        powers = ['Pdcdc', 'Pmain', 'Pcore', 'Pvddio', 'Ptotal']

        # When Pmems is enabled, this is messing-up the gpio
        # 'Pmems'

        # read measured values in a for loop like this:
        prev_gpio = None
        sample_buffer = {}
        for power_name in powers:
            sample_buffer[power_name] = []

        measure_id = 0
        sampling_enabled = False

        for i, gpio in enumerate(samples_gpio):

            if prev_gpio is not None:
                if prev_gpio == 0 and gpio == 1:
                    # Raising GPIO edge, reset the buffer to start a new measurement
                    sampling_enabled = True
                    for name in powers:
                        sample_buffer[name] = []
                elif prev_gpio == 1 and gpio == 0:
                    # Falling GPIO edge, if we collected something, compute average and display
                    if len(sample_buffer[powers[0]]) != 0:
                        for name in powers:
                            power = sum(sample_buffer[name])/len(sample_buffer[name])
                            self.__dump_power_measure(measure_id, name, power)
                        measure_id += 1
                    # Period is over, set buffer to None so that we stop collecting sample
                    sampling_enabled = False

                else:
                    # Otherwise collect the sample only if sampling is enabled
                    if sampling_enabled:
                        for name in powers:
                            sample_buffer[name].append(samples[name][i])

            prev_gpio = gpio

        if self.power_plot:
            self.__plot_power(samples)

    def __dump_power_measure(self, measure_id, name, power):
        if self.power_all:
            print(f"@power_{name}.measure_{measure_id}@{power}@")
        else:
            if name == 'Pdcdc':
                print(f"@power.measure_{measure_id}@{power}@")

    def __plot_power(self, samples):
        samples_power = samples.copy()
        samples_gpio = samples.copy()

        samples_power = samples_power[
            samples.columns[pd.Series(samples.columns).str.startswith('P')]]
        if not samples_power.empty:
            samples_power['time'] = samples['time']

        samples_gpio = samples_gpio[
            samples.columns[pd.Series(samples.columns).str.startswith('gpio')]]
        if not samples_gpio.empty:
            samples_gpio['time'] = samples['time']

        if not samples_gpio.empty:
            if not samples_power.empty:
                __, values_x = plt.subplots()
                samples_power.plot(title="Power",x='time',ax=values_x)
                samples_gpio.plot(title="gpio",x='time',ax=values_x,secondary_y = True)

        plt.show()



class Ppk2():
    """
    Helper class for using Power profiler kit II.

    This is based on ppk2-api package with a synchronization with GPIO to compute average power
    """

    def __init__(self):
        self.ppk2_test = ppk2_api.ppk2_api.PPK2_API("/dev/ttyACM0")
        self.ppk2_test.get_modifiers()
        self.ppk2_test.use_source_meter()
        self.ppk2_test.set_source_voltage(1800)
        self.ppk2_test.toggle_DUT_power("ON")
        self.stop_measure = False
        self.thread = None


    def start(self):
        """Start sampling.

        This will launch a background thread which will collect samples and check for
        measurement periods delimited by GPIO being at 1.
        Measurements are printed on the terminal.
        """
        self.ppk2_test.start_measuring()
        self.stop_measure = False
        self.thread = threading.Thread(target=self.__sample_loop)
        self.thread.start()


    def stop(self):
        """Stop sampling.
        """
        self.stop_measure = True
        self.ppk2_test.stop_measuring()
        self.thread.join()
        del self.ppk2_test
        # PPk2 takes some time to close which can be a problem when chaining tests
        time.sleep(1)


    def __sample_loop(self):
        # read measured values in a for loop like this:
        prev_gpio = None
        sample_buffer = None
        measure_id = 0

        while not self.stop_measure:

            # Get from uart all what was received since last call
            samples, samples_logic = self.__read_next_chunk()
            if samples is None:
                continue

            # Go through samples and append them only if we are inside a phase where GPIO is 1
            for i, sample in enumerate(samples):
                gpio = samples_logic[i] & 1

                if prev_gpio is not None:
                    if prev_gpio == 0 and gpio == 1:
                        # Raising GPIO edge, reset the buffer to start a new measurement
                        sample_buffer = []
                    elif prev_gpio == 1 and gpio == 0:
                        # Falling GPIO edge, if we collected something, compute average and display
                        if sample_buffer is not None and len(sample_buffer) != 0:
                            power = sum(sample_buffer)/len(sample_buffer)*1.8/1000000
                            print(f"@power.measure_{measure_id}@{power}@")
                            measure_id += 1
                        # Period is over, set buffer to None so that we stop collecting sample
                        sample_buffer = None

                    else:
                        # Otherwise collect the sample only if a buffer is ready which means GPIO
                        # is 1
                        if sample_buffer is not None:
                            sample_buffer.append(sample)

                prev_gpio = gpio


    def __read_next_chunk(self):
        # Wait a bit to collect enough data. Sampling is 100kSamples / s
        # so we should get 100 samples
        time.sleep(0.001)
        read_data = self.ppk2_test.get_data()
        if read_data != b'':
            return self.ppk2_test.get_samples(read_data)

        return None, None





class Openocd():
    """
    Helper class for managing gap9 targets through Openocd.

    This is using Openocd telnet proxy to send read/write commands to the target

    Attributes
    ----------
    args : argparse.ArgumentParser
        The command line arguments. This class will use to get Openocd options
    """

    def __init__(self, args: argparse.ArgumentParser):
        self.args = args
        self.telnet = None
        self.run = None

        if self.args.openocd_cable is None:
            raise RuntimeError("Argument --openocd-cable is missing")

        if self.args.openocd_script is None:
            raise RuntimeError("Argument --openocd-script is missing")

        if self.args.openocd_tools is None:
            raise RuntimeError("Argument --openocd-tools is missing")

    def connect(self):
        """Connect to target.

        This will launch OpenOCD with the telnet proxy and connect to it so that this class
        is ready to interact with it.
        """

        # To allow the execution of several gapy in parallel, we cannot use a fixed port.
        # Iterate until we find an available one.
        retry_nb = 0
        while retry_nb < 30:

            port = random.randint(4000, 20000)

            cmd = (
                f"{self.args.openocd} "
                f'-c "gdb_port disabled; telnet_port {port}; tcl_port disabled" '
                f'-f "{self.args.openocd_cable}" -c "{self.args.openocd_precmd}" '
                f'-f "{self.args.openocd_script}"'
            )

            # And finally execute it
            print ('Flashing image with command:')
            print (cmd)

            self.run = pexpect.spawn(cmd, encoding='utf-8', logfile=sys.stdout)
            try:
                match = self.run.expect(['Listening on', 'could not open port', 'Handshake fail abort'], timeout=None)
            except pexpect.exceptions.EOF:
                match = 1
            if match == 0:
                success = True
                break

            # An error was detected, wait until OpenOCD exits
            # try:
            #     self.run.expect(pexpect.EOF, timeout=None)
            # except pexpect.exceptions.EOF:
            #     pass

            # An error was detected, force OpenOCD to exit
            print ('Error detected. Forcing OpenOCD close.')
            self.run.close(True)

            if match == 2:
                print ('Handshake error was detected')
                raise RuntimeError('Handshake error')

            print ('Telnet port already in use, restarting with another port')
            retry_nb+=1
            time.sleep(1)
        else:
            raise RuntimeError('Failed to connect to openocd after 30 retries')

        self.telnet = pexpect.spawn(f'telnet localhost {port}', encoding='utf-8', echo=False)
        match = self.telnet.expect(['Open On-Chip Debugger'], timeout=None)

    def write(self, addr: int, value: int):
        """Write a 4-bytes word to target.

        This generates a mww command to the telnet proxy and wait until the command is done.

        Parameters
        ----------
        addr : int
            Target address of the access.
        value : int
            Value to be written at the specified address.
        """
        self.telnet.sendline(f'mww 0x{addr:x} 0x{value:x}')
        while True:
            # Wait on OCD with no timeout just to display flasher printf
            #self.run.expect([pexpect.TIMEOUT], timeout=0)
            match = self.telnet.expect(['> ', pexpect.TIMEOUT], timeout=0)
            if match == 0:
                return


    def read(self, addr: int):
        """Read a 4-bytes word from the target.

        This generates a mdw command to the telnet proxy and wait until the command is done.

        Parameters
        ----------
        addr : int
            Target address of the access.

        Returns
        -------
        int
            Value read at the specified address.
        """
        self.telnet.sendline(f'mdw 0x{addr:x}')
        while True:
            # Wait on OCD with no timeout just to display flasher printf
            # self.run.expect([pexpect.TIMEOUT], timeout=0)
            match = self.telnet.expect([f'0x{addr:x}: [0-9a-fA-F]+', pexpect.TIMEOUT], timeout=0)
            if match == 0:
                value = int(self.telnet.after.split()[1], 16)
                return value

    def load_and_start(self, binary: str):
        """Load a binary into the target and starts it.

        Parameters
        ----------
        binary : str
            Path to the binary to be loaded.
        """
        with open(binary, 'rb') as file_desc:
            elffile = ELFFile(file_desc)
            entry = elffile.header['e_entry']

        self.telnet.sendline(f'load_and_start_binary {binary} 0x{entry:x}')

    def load_from_file(self, path: str, file_offset: int, target_address: int, size: int):
        """Load the content of a file into the target memory.

        Parameters
        ----------
        path : str
            Path to the file containing the data to be loaded into the target.
        target_address : int
            Target address where the file content should be loaded.
        file_offset : int
            Offset in the file where the data should be taken.
        size : int
            Size of the data which should be laoded to the target
        """
        self.telnet.sendline(f'load_image {path} {target_address - file_offset} bin'
            f' {target_address} {size}')



class Flasher():
    """
    Utility class for interacting with gap9 flasher.

    Attributes
    ----------
    ocl : Openocd
        Openocd utility class for interacting with the target.
    binary : str
        Path to the flasher binary to be used on the target.
    size : int
        Size of the flash
    block_size : int
        BLock size to be used for the interactions with the flasher.
    """

    def __init__(self, ocd: Openocd, flash_type: str, binary: str, block_size: int):
        self.ocd = ocd
        self.binary = binary
        self.block_size = block_size
        self.flash_type = flash_type

        # Load and start the flasher
        self.ocd.load_and_start(binary)

        # Retrive some information like the debug struct
        debug_struct_symbol = '__rt_debug_struct_ptr'
        debug_struct_addr = self.__get_symbol_from_file(binary, debug_struct_symbol)

        if debug_struct_addr is None:
            raise RuntimeError(f'Flasher does not contain debug struct (binary: {binary}, '
                f'symbol: {debug_struct_symbol})')

        count = 0
        while True:
            device_struct_ptr = ocd.read(debug_struct_addr)

            if device_struct_ptr not in [0xdeadbeef, 0]:
                break

            count += 1
            if count == 0x80:
                raise RuntimeError('flasher script could not connect to board, check your cables')

        self.device_struct = device_struct_ptr

        self.host_rdy        = device_struct_ptr + 0
        self.gap_rdy         = device_struct_ptr + 4
        self.buff_ptr_addr   = device_struct_ptr + 8
        self.buff_size_addr  = device_struct_ptr + 12
        self.flash_run       = device_struct_ptr + 16
        self.flash_addr      = device_struct_ptr + 20
        self.flash_size      = device_struct_ptr + 24
        self.flash_type_addr = device_struct_ptr + 28

    def upload_from_file(self, path: str, base_addr: int, size: int):
        """Upload file content into the flash.

        Parameters
        ----------
        path : str
            Path to the file containing the data to be uploaded to the flash.
        size : int
            Size of the data to be uploaded.
        """
        total_size = size
        offset = 0

        # GAP RDY  <--- 0
        self.ocd.write(self.gap_rdy, 0x0)
        self.ocd.write(self.flash_type_addr, 0x0)

        # tell the chip we are going to flash
        self.ocd.write(self.flash_run, 0x1)

        # HOST RDY <--- 1 / signal to begin app
        self.ocd.write(self.host_rdy, 0x1)
        buff_ptr = self.ocd.read(self.buff_ptr_addr)

        while size > 0:

            if size > self.block_size:
                iter_size = self.block_size
            else:
                iter_size = size

            size -= iter_size

            # spin on gap rdy: wait for current flash write to finish
            while self.ocd.read(self.gap_rdy) != 1:
                time.sleep(0.01)

            self.ocd.write(self.host_rdy, 0)

            if size == 0:
                self.ocd.write(self.flash_run, 0)

            self.ocd.write(self.flash_addr, base_addr + offset)
            self.ocd.write(self.flash_size, iter_size)

            size_done = total_size - size - iter_size
            percentage_done = (size_done) / total_size * 100
            print (f'\rloading {os.path.basename(path)}'
                   f' to {self.flash_type} - copied {size_done} / {total_size} Bytes'
                f' - {percentage_done:.2f} %', end='')

            self.ocd.load_from_file(path, offset, buff_ptr, iter_size)

            #signal app we wrote our buff
            # ACK the gap rdy now that we wrote sector (flasher may work)
            self.ocd.write(self.gap_rdy, 0)

            #signal we are rdy when flasher is
            self.ocd.write(self.host_rdy, 1)

            offset += iter_size

        # Wait for the flasher to finish erasing+programming+verifying the LAST
        # chunk. Without this, the host returns while the final 16 KB SPI write
        # is still in flight; the next operation (e.g. loading the MRAM flasher)
        # halts the chip and cuts it off, leaving the last chunk uncommitted.
        # The flasher restores flash_run to 1 once it is fully done.
        while self.ocd.read(self.flash_run) != 1:
            time.sleep(0.01)

        print (f'\rloading {os.path.basename(path)} '
               f'to {self.flash_type} - copied {total_size} / {total_size} Bytes'
            f' - {100.:.2f} %')

    def __get_symbol_from_file(self, binary: str, symbol_name: str):
        with open(binary, 'rb') as file:
            elf = ELFFile(file)
            for section in elf.iter_sections():
                if section.header['sh_type'] == 'SHT_SYMTAB':
                    for symbol in section.iter_symbols():
                        if symbol.name == symbol_name:
                            t_vaddr=symbol.entry['st_value']
                            return t_vaddr

        return None


class Runner():
    """
    Helper class for running execution on gap9_v2 boards.

    Attributes
    ----------
    target : gapylib.target.Target
        The target on which the execution is run.
    """

    def __init__(self, target: gapylib.target.Target):
        self.target = target


    @staticmethod
    def append_args(parser: argparse.ArgumentParser):
        """Append runner specific arguments to gapy command-line.

        This is used to append arguments only if the RTL platform is selected.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The parser where to add arguments.
        """

        parser.add_argument("--openocd", dest="openocd", type=str, default="openocd",
            help="path to openocd executable")

        parser.add_argument("--openocd-cable", dest="openocd_cable", type=str, default=None,
            help="Openocd cable")

        parser.add_argument("--openocd-script", dest="openocd_script", type=str, default=None,
            help="Openocd script")

        parser.add_argument("--openocd-tools", dest="openocd_tools", type=str, default=None,
            help="Openocd tools")

        parser.add_argument(
            "--openocd-precmd",
            dest="openocd_precmd",
            type=str,
            default="",
            help="Openocd pre-command",
        )

        parser.add_argument("--gdb", dest="gdb", action="store_true",
            help="launch with gdb support enabled")

        parser.add_argument("--py-flasher", dest="py_flasher", action="store_true",
            help="interract with the flasher directly from Python")

        parser.add_argument("--gdb-port", dest="gdb_port", default=3333, type=int,
            help="GDB port")

        parser.add_argument("--wsl", dest = "wsl", type=str, default=None,
            help = "Launch command in wsl environment")

        parser.add_argument("--power", dest = "power", action="store_true", default=None,
            help = "Launch command with power extraction")

        parser.add_argument("--power-plot", dest = "power_plot", action="store_true", default=None,
            help = "Plot power")

        parser.add_argument("--power-all", dest = "power_all", action="store_true", default=None,
            help = "Dump all available power")

        parser.add_argument("--power-tool", dest = "power_tool", default="testbench", choices=[
            'ppk2', 'testbench'], help = "Select tool to extract power")



    def parse_args(self, args: any):
        """Handle arguments.

        This will mostly use the arguments to prepare the platform command.

        Parameters
        ----------
        args :
            The arguments.
        """


    def __flash_image(self, args: argparse.ArgumentParser, ocd: Openocd, flash: gapylib.flash.Flash,
                      base_addr: int, first_section: int, last_section: int):

        # And get image for selected sections
        image = flash.get_image(first=first_section, last=last_section)

        flasher_attribute = flash.get_flash_attribute('flasher')
        if os.path.basename(flasher_attribute) != flasher_attribute:
            flasher = flasher_attribute
        else:
            flasher = args.openocd_tools + '/gap_bins/' + flasher_attribute
        block_size = flash.get_flash_attribute('flasher_block_size')
        if isinstance(block_size, str):
            try:
                block_size = int(block_size)
            except ValueError:
                block_size = int(block_size,16)
        flash_type = flash.get_flash_attribute('flash_type')

        # Be careful, Openocd only accept images of multiple of 4 bytes, pad with
        # few bytes if needed
        image_size = len(image)
        padded_image_size = (image_size + 3) & ~0x3
        for _ in range(0, padded_image_size - image_size):
            image.append(0)

        image_path = flash.get_image_path()+f"_{base_addr}"
        with open(image_path, 'wb') as out_desc:
            out_desc.write(image)

        # Generate the right Openocd command
        if args.wsl is not None:
            path_header = '\\"//wsl$/' + args.wsl
            path_footer = '\\"'
            flash_img = path_header + image_path + path_footer
        else:
            flash_img = image_path

        flasher = Flasher(ocd, flash_type, flasher, block_size)

        flasher.upload_from_file(flash_img, base_addr, padded_image_size)

    def flash(self):
        """Handle the gapy command 'flash' to upload the flash images.
        """
        args = self.target.get_args()

        if args.openocd_cable is None:
            raise RuntimeError("Argument --openocd-cable is missing")

        if args.openocd_script is None:
            raise RuntimeError("Argument --openocd-script is missing")

        if args.openocd_tools is None:
            raise RuntimeError("Argument --openocd-tools is missing")

        ocd = None

        # Go through all the flashes and upload their content to the target
        for flash in self.target.flashes.values():

            # Always flash if we are not in auto mode, or if the flash is not empty
            if not args.flash_auto or not flash.is_empty():

                # Openocd just needs to be given the image and few information and will take care
                # of everything.

                # To not pollute flashing with empty sections, we bypass the last empty sections.
                # We could do the same for empty sections in the middle by executing
                # several times the flasher in case it becomes a bottleneck.

                # First drop last empty sections
                sections = flash.get_sections()

                # Loop until we have no more sections to load.
                # Each iteration is taking a set of contiguous and non-empty sections
                # so that we waste a little time as possible on connection.
                # We stop as the first empty section in order to avoid flashing it for nothing.
                first_index = None
                base_addr = None
                for index, section in enumerate(sections):

                    # Skip leading empty sections
                    if first_index is None and section.is_empty():
                        continue

                    if first_index is None:
                        base_addr = section.get_offset()
                        first_index = index

                    # We flash the sections either if we are at the last section or we find an empty
                    # section. We don't skip empty sections with size 0 to have more chance to
                    # with a non-empty sections which comes after
                    if index == len(sections) - 1 or section.is_empty():
                        if section.is_empty():
                            last = index - 1
                        else:
                            last = index

                        if ocd is None:
                            ocd = Openocd(args)
                            ocd.connect()

                        self.__flash_image(args, ocd, flash, base_addr, first_index, last)

                        # Reinit the set of sections
                        first_index = None



    def run(self):
        """Handle the gapy command 'run' to start execution on the platform.
        """
        args = self.target.get_args()

        if args.openocd_cable is None:
            raise RuntimeError("Argument --openocd-cable is missing")

        if args.openocd_script is None:
            raise RuntimeError("Argument --openocd-script is missing")

        if args.openocd_tools is None:
            raise RuntimeError("Argument --openocd-tools is missing")

        # Since we are in JTAG mode, we just need to pass the binary to Openocd and it will take
        # care to upload it to the target and start the target on the entry point

        # Generate the Openocd command
        gdb_port = 'disabled'
        if args.gdb:
            gdb_port = args.gdb_port

        cmd = (
            f'{args.openocd} -d0 -c "gdb_port {gdb_port}; telnet_port disabled; '
            f'tcl_port disabled" -f "{args.openocd_cable}" -c "{args.openocd_precmd}" '
            f'-f "{args.openocd_script}" '
        )

        if not args.gdb and args.binary is not None:
            with open(args.binary, 'rb') as file_desc:
                elffile = ELFFile(file_desc)
                entry = elffile.header['e_entry']

            if args.wsl is None:
                cmd += f' -c "load_and_start_binary {args.binary} 0x{entry:x}"'
            else:
                path_header = '\\"//wsl$/' + args.wsl
                path_footer = '\\"'
                binary = path_header + args.binary + path_footer
                cmd += f' -c "load_and_start_binary {binary} 0x{entry:x}"'

        os.chdir(self.target.get_working_dir())

        print ('Launching execution with command:')
        print (cmd)

        # If power extraction is enabled, start it to create the background thread for extraction
        if args.power:
            if args.power_tool == 'testbench':
                testbench = Testbench(args.power_plot, args.power_all)
                testbench.start()
            else:
                ppk2 = Ppk2()
                ppk2.start()

        # And execute it
        error = os.system(cmd)

        if args.power:
            if args.power_tool == 'testbench':
                testbench.stop()
            else:
                ppk2.stop()

        if error != 0:
            raise RuntimeError(f'The board returned an error: {error:d}')
