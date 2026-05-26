from CCDClass import QISCCDFactory

directory = "/home/daq_user/ltaDaemon-master/images/2026-05-20/"
filename  = "proc_500samp_200rows_3"
ccd_data = QISCCDFactory(directory+filename, "8")

ccd_data.load_images(4)
ccd_data.read_header()
ccd_data.process_overscan()
#ccd_data.plot_fits()
ccd_data.measure_row_drift()

