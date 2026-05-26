#!/usr/bin/env python
# filter and crystal fluoresce scan
import os, sys, time, datetime
import glob
import logging
from subprocess import call
import numpy as np

sys.path.append('/home/qis/soft/astroskip_10/python/device')

from shutter import Shutter
from monochromator import Monochromator
from lakeshore import Lakeshore
from powermeter import Powermeter
from lta import Lta
from lamp import Lamp

WSTART = 250
WSTOP = 750
WSTEP = 10
source = 'LED'
Filter = 6
WAVES =  np.arange(WSTART,WSTOP,WSTEP).tolist()
REF_EXPTIME = 30  # exposure time (s)
wait = 0 # TIME for power meter to take measurment

nmax = 1

def source_file(file):
    """ runs bash scripts 

    Parameters:
    -----------
    file: name of the ".sh" file/path to the file 

    Returns:
    --------
    None 
    """
    call('bash'+ " " + file, shell=True)
    
def make_dir():
    """ Makes directory to save images 

    Parameters:
    -----------
    None

    Returns:
    --------
    direcotry path 
    """

    today = datetime.datetime.now().strftime("%Y%m%d")
    OutDir = './images/%s_CLEAN/'%today
    if not os.path.exists(OutDir): os.makedirs(OutDir)

    return OutDir


def clean(fast_clean_file=None,cleaning_voltage_file=None,voltage_file=None):
    fast_clean_file="fast_clean_v5.sh"
    cleaning_voltage_file="manu_fastclear_voltages_v2.sh"
    voltage_file="low_voltages.sh"
    source_file(cleaning_voltage_file)
    source_file(fast_clean_file)
    source_file(voltage_file)

    

def get_exptime(wave, exptime_orig):
    """Increase the exposure time for wavengths shorter than 550nm.

    Parameters
    ----------
    wave     : wavelength [nm]
    exptime_orig : original exposure time [seconds]

    Returns
    -------
    exptime : scaled exposure time [seconds]
    """
    WAVE = 550.0 # reference wavelength
    diff = WAVE - wave
    exptime = exptime_orig
    if diff >= 0: exptime *= (1 + diff / 50.0)
    logging.debug("Scaled exptime for %s nm : %s s"%(wave,exptime))
    return int(exptime)

def take_exposure(exptime=10.0,nrow=None,nsamp=None, source='lamp'):
    """ Take a skipper exposure

    Parameters:
    -----------
    exptime : exposure time (seconds)
    nrow : number of rows to read ('lta NROW')
    nsamp : number of samples to read ('lta NSAMP')

    Returns:
    --------
    None
    """
    if source.upper() == 'LED':
        print("turning LED on") # !!
        turn_led('on')

    # pwr = Shutter().expose(exptime,measure=True,wait=wait)
    time.sleep(exptime + wait) # to just leave Shutter open
    pwr = 0.
    if source.upper() == 'LED':
        print("turning LED off") # !!
        turn_led('off')
    time.sleep(1)

    status = Lta().read(nsamp=nsamp,nrow=nrow)
    time.sleep(1)
    return status, pwr


def turn_led(cmd):
    assert cmd == 'on' or cmd == 'off', "led command must be 'on' or 'off'"
    call(["python3", "../astroskip_10/bin/relay.py", f"-{cmd}"])


def scan(name='scintillation_',waves=None, exptime=REF_EXPTIME, nrow=20, nsamp=150, source='lamp'):
    """ Take data for scan measurements.
    
    Parameters
    ----------
    name  : str
        output file name prefix
    waves : list, float
        list of wavelengths (nm)
    exptime : float
        exposure time (seconds)
    nrow    : int
        number of rows to read
    nsamp   : int
        number of samples per pixel

    Returns
    -------
    None
    """
    POWER=list()
    if waves is None or not len(waves):
        waves = WAVES

    logging.info("waves (nm) : %s"%waves)
    logging.info("exptime (s): %s"%exptime)
    logging.info("nrow       : %s"%nrow)
    logging.info("nsamp      : %s"%nsamp)

    dirbase = '/home/qis/soft/ltaDaemon_Astroskipper/images/%(today)s_%(idx)02d'
    basename = '%(name)s_t%(exptime).1f_w%(wave)i_n%(nsamp)i_i%(idx)i_'

    i = 0
    today = datetime.datetime.now().strftime("%Y%m%d")
    dirname = dirbase%dict(today=today,idx=i)
    while os.path.exists(dirname):
        i+=1
        dirname = dirbase%dict(today=today,idx=i)

    os.makedirs(dirname)
    
    # create the devices
    mono = Monochromator()
    #shutter = Shutter()
    #power = Powermeter()
    #lake = Lakeshore()
    
    # Change the filter
    #flt = mono.set_filter(1)
    #flt = mono.get_filter()
    if Filter is not None:
        mono.set_filter(Filter)
        flt = int(mono.get_filter())
    else:
        flt = 'None'
    logging.info(f"filter     : {flt}")
    
    # turn the lamp on if we're using it as the source
    if source.upper() == 'LAMP':
        print("turning lamp on")
        Lamp().start()

    # Change the grating
    #grt = mono.set_grating(2)
    #grt = mono.get_grating()
    grt=1
    
    for wave in waves:
        for idx in range(nmax):
            # Send the monochromator to a specific grating, filter and wavelength
            #grt,flt,wav = mono.sendto(wave)
            #time.sleep(30)
            wav = mono.set_wave(wave)

            # Set the powermeter wavelength (this is already done by monochromator)
            #power.set_wave(wave)
            #time.sleep(15)
             
            # Clean the CCD
            logging.info("Cleaning CCD...")
            clean()
            clean()

            # Get the scaled expsoure time
            #etime = get_exptime(wave,exptime)
            #etime = exptime
            # Take the exposure
            params =dict(name=name,exptime=exptime,nsamp=nsamp,idx=idx,wave=wave)
            filename = os.path.join(dirname,basename%params)
            Lta().name(filename)
            logging.info("Taking exposure %s*.fz..."%filename)
            status,pwr = take_exposure(exptime, nrow, nsamp, source)
            POWER.append(0) # !!
            logging.info('Done')

            # Get the temperature
            #temp = lake.get_temp()
            temp = -1

            values = {'EXPTIME':(float(exptime)+wait,'Exposure time (s)'),
                      'WAVE':   (float(wave),   'Wavelength (nm)'),
                      'POWER':  (float(0),    'Powermeter reading (W)'),
                      'TEMP':   (float(temp),   'Temperature (K)'),
                      'GRATING':(int(grt),      'Grating'),
                      'SOURCE':(str(source), 'Light source'),
                      'FILTER': (flt,      'Filter'),
                      'IMAGE_NUM': (int(idx),   'Image Number (of {})'.format(nmax))
            }

            # Update the header with values
            logging.info("Updating headers...")
            for f in glob.glob(filename+'*.fz'):
                Lta.modhead(f,values)
            msg = '\n'+'\n'.join("  %-7s = %-9s / %s"%(k,v[0],v[1]) for k,v in values.items())
            logging.info(msg)
            logging.info('Done with %s\n'%(filename+'*.fz'))
            time.sleep(3)
    if source.upper() == 'LED':
        turn_led('off')
    elif source.upper() == 'LAMP':
        print("turning lamp off")
        Lamp().stop()
    np.savetxt(dirname+"/new_filter_scan.txt",np.array(POWER))
                
if __name__ == '__main__':
    from device import Parser
    parser = Parser(description=__doc__)
    parser.add_argument('-r', '--nrow',default=100,type=int,
                        help='number of rows to read')
    parser.add_argument('-s', '--nsamp',default=1,type=int,
                        help='number of samples per exposure')
    parser.add_argument('-t', '--exptime',default=5.0,type=float,
                        help='"exposure" time (seconds)')
    parser.add_argument('-w', '--waves',nargs='+',default=None, type=int,
                        help='wavelengths (nm)')
    parser.add_argument('-S', '--source', default='lamp', type=str,
                        help='light source (must be "lamp", "led", or "None")')
    parser.add_verbose()
    parser.add_version()
    args = parser.parse_args()

    scan(waves=args.waves,nsamp=args.nsamp,exptime=args.exptime,nrow=args.nrow, source=args.source)
