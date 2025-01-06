import socket
import fcntl
import struct

def get_ip_address(ifname):
    '''get IP Address of given interface'''
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack(bytes('256s', encoding="utf-8"), bytes(ifname[:15], encoding="utf-8"))
    )[20:24])
