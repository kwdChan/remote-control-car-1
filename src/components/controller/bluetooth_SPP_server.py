import os
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib # type: ignore
from typing import Callable, Tuple, TypeVar, get_origin, get_args, Union, Any, Optional, Dict, List
from multiprocessing.managers import BaseProxy, BaseManager, DictProxy, SyncManager
from multiprocessing import Process



def start_bluetooth_server(manager: SyncManager):

    data = manager.dict()

    def update(message:str):
        "a -> a down, a0 = a up"
        up = message[-1] != '0'
        key = message if up else message[:-1]
        data[key] = up

    p = Process(
        target=main,
        args=(update, )
    )
    p.start()

    return [data], p
    
    
    



# the code below is adapted from  https://raspberrypi.stackexchange.com/questions/140934/cant-use-bluetooth-com-ports-from-windows-10
# this is needed to connect to most bluetooth app on android google playstore 
# that expected the UUID '00001101-0000-1000-8000-00805f9b34fb'

class Profile(dbus.service.Object):
    fd = -1
    
    callbacks = [] # MODIFIED

    @dbus.service.method('org.bluez.Profile1',
                         in_signature='',
                         out_signature='')
    def Release(self):
        print('Release')

    @dbus.service.method('org.bluez.Profile1',
                         in_signature='oha{sv}',
                         out_signature='')
    def NewConnection(self, path, fd, properties):
        self.fd = fd.take()
        print('NewConnection(%s, %d)' % (path, self.fd))
        for key in properties.keys():
            if key == 'Version' or key == 'Features':
                print('  %s = 0x%04x' % (key, properties[key]))
            else:
                print('  %s = %s' % (key, properties[key]))
        io_id = GLib.io_add_watch(self.fd,
                                  GLib.PRIORITY_DEFAULT,
                                  GLib.IO_IN | GLib.IO_PRI,
                                  self.io_cb)
    # MODIFIED
    def io_cb(self, fd, conditions):
        data = os.read(fd, 1024)
        message = data.decode('utf-8')

        for cb in self.callbacks:
            cb(message)

        os.write(fd, bytes(list(reversed(data.rstrip()))) + b'\n')
        return True

    # MODIFIED
    def register_callback(self, func: Callable[[str], None]) -> None:
        self.callbacks.append(func)

    @dbus.service.method('org.bluez.Profile1',
                         in_signature='o',
                         out_signature='')
    def RequestDisconnection(self, path):
        print('RequestDisconnection(%s)' % (path))

        if self.fd > 0:
            os.close(self.fd)
            self.fd = -1


def main(callback: Callable[[str], None]):
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SystemBus()

    manager = dbus.Interface(bus.get_object('org.bluez',
                                            '/org/bluez'),
                             'org.bluez.ProfileManager1')

    mainloop = GLib.MainLoop()

    adapter = dbus.Interface(bus.get_object('org.bluez',
                                            '/org/bluez/hci0'),
                            dbus.PROPERTIES_IFACE)
    discoverable = adapter.Get('org.bluez.Adapter1', 'Discoverable')
    if not discoverable:
        print('Making discoverable...')
        adapter.Set('org.bluez.Adapter1', 'Discoverable', True)

    #TODO: what is this profile path doing? 
    # I can't run this function twice without restarting python 
    # KeyError: "Can't register the object-path handler for '/foo/baz/profile': there is already a handler"

    profile_path = '/foo/baz/profile'  # what????
    server_uuid = '00001101-0000-1000-8000-00805f9b34fb'
    opts = {
        'Version': dbus.UInt16(0x0102),
        'AutoConnect': dbus.Boolean(True),
        'Role': 'server',
        'Name': 'SerialPort',
        'Service': server_uuid, 
        'RequireAuthentication': dbus.Boolean(False),
        'RequireAuthorization': dbus.Boolean(False),
        'Channel': dbus.UInt16(1),
    }

    print('Starting Serial Port Profile...')

    profile = Profile(bus, profile_path)
    profile.register_callback(callback) # MODIFIED

    manager.RegisterProfile(profile_path, server_uuid, opts)

    try:
        mainloop.run()
    except KeyboardInterrupt:
        mainloop.quit()
