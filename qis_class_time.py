from CCDClass import QISCCDFactory
import argparse

parser = argparse.ArgumentParser('Parse text in the file')
parser.add_argument('filename', help = 'folder you want to process')
args = parser.parse_args()

ccd_data = QISCCDFactory(args.filename, "8", 1)

ccd_data.load_images(4)
ccd_data.read_header()
ccd_data.process_overscan()
#ccd_data.plot_fits()
#ccd_data.measure_row_drift()
ccd_data.subtract_overscan()

