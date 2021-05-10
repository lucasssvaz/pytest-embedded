import functools
import os
import sys
from typing import Optional, Tuple

import esptool as esptool
from pytest_embedded.app import App
from pytest_embedded_serial.serial_dut import SerialDut
from serial.tools.list_ports_posix import comports


class EspSerialDut(SerialDut):
    def __init__(self, app: Optional[App] = None, port: Optional[str] = None, *args, **kwargs) -> None:
        self.target = getattr(self, 'target', 'auto')
        self.target, port = detect_target_port(self.target, port)

        # put at last to make sure the forward_io_proc is forwarding output after the hard reset
        super().__init__(app, port, *args, **kwargs)

    def _uses_esptool(func):
        @functools.wraps(func)
        def handler(self, *args, **kwargs):
            settings = self.port_inst.get_settings()

            # esptool use print, we need to redirect that to our pexpect
            origin_stdout = sys.stdout
            sys.stdout = self.pexpect_proc

            if getattr(self, 'rom_inst', None) is None:
                self.rom_inst = esptool.ESPLoader.detect_chip(self.port_inst)

            try:
                self.rom_inst.connect('hard_reset')
                stub_inst = self.rom_inst.run_stub()

                ret = func(self, stub_inst, *args, **kwargs)
                # do hard reset after use esptool
                stub_inst.hard_reset()
            finally:
                self.port_inst.apply_settings(settings)
                sys.stdout = origin_stdout
            return ret

        return handler

    @_uses_esptool
    def hard_reset(self, stub_inst: esptool.ESPLoader):
        pass

    def start(self):
        self.hard_reset()


def detect_target_port(target=None, port=None) -> Tuple[str, str]:
    available_ports = _list_available_ports()

    if target:
        if port:
            return target, port
        else:
            return _judge_by_target(available_ports, target)
    elif port:
        if port not in available_ports:
            raise ValueError(f'Port "{port}" unreachable')
        return _judge_by_port(port)
    else:  # pick the first available port then...
        return _judge_by_target(available_ports, 'auto')


def _list_available_ports():
    def _sort_usb_ports(_ports):
        # we only use usb ports
        usb_ports = []
        for port in _ports:
            if 'usb' in port.lower():
                usb_ports.append(port)
        return usb_ports

    ports = _sort_usb_ports([x.device for x in comports()])
    espport = os.getenv('ESPPORT')
    if not espport:
        return ports

    # If $ESPPORT is a valid port, make it appear first in the list
    if espport in ports:
        ports.remove(espport)
        return [espport] + ports

    # On macOS, user may set ESPPORT to /dev/tty.xxx while
    # pySerial lists only the corresponding /dev/cu.xxx port
    if sys.platform == 'darwin' and 'tty.' in espport:
        espport = espport.replace('tty.', 'cu.')
        if espport in ports:
            ports.remove(espport)
            return [espport] + ports
    return ports


def _rom_target_name(rom: esptool.ESPLoader) -> str:
    return rom.__class__.CHIP_NAME.lower().replace('-', '')


def _judge_by_target(ports: list[str], target: str = 'auto') -> Tuple[str, str]:
    for port in ports:
        rom_inst = None
        try:
            rom_inst = esptool.ESPLoader.detect_chip(port)
            inst_target = _rom_target_name(rom_inst)
            if target == 'auto' or inst_target == target:
                return inst_target, port
        except Exception:  # noqa
            continue
        finally:
            if rom_inst is not None:
                rom_inst._port.close()
    raise ValueError(f'Target "{target}" port not found')


def _judge_by_port(port: str) -> Tuple[str, str]:
    rom_inst = None
    try:
        rom_inst = esptool.ESPLoader.detect_chip(port)
    except Exception:  # noqa
        raise
    else:
        return _rom_target_name(rom_inst), port
    finally:
        if rom_inst is not None:
            rom_inst._port.close()