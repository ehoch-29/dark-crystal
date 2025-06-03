import serial
import argparse

parser = argparse.ArgumentParser('Parse text in the file')
parser.add_argument( '-on', '--on', action = 'store_true', help = 'include -o flag to rewrite the save data?')
parser.add_argument( '-off', '--off', action = 'store_true', help = 'include -o flag to rewrite the save data?')

args = parser.parse_args()

if args.on:
    command = b'1'
if args.off:
    command = b'0'
    
arduino = serial.Serial('/dev/tty.usbmodem14201', 9600, timeout = 1)
arduino.write(command) 
