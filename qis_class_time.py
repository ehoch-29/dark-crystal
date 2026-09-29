from CCDClass import QISCCDFactory
import argparse

parser = argparse.ArgumentParser('Parse text in the file')
parser.add_argument('filename', help = 'folder you want to process')
parser.add_argument('-n', '--filenumber', help = 'image number you want to process')
args = parser.parse_args()


directory = "/home/daq_user/ltaDaemon-master/images/2026-09-28/"
filename  = args.filename
ccd_data = QISCCDFactory(directory+filename, args.filenumber, ".fits")

ccd_data.load_images(4)
ccd_data.read_header()
ccd_data.process_overscan()
ccd_data.plot_active_area()
#ccd_data.plot_fits()
ccd_data.measure_row_drift()
ccd_data.subtract_overscan()
ccd_data.stitch_image()
#ccd_data.plot_active_area()
#ccd_data.fit_multi_gaussian()

