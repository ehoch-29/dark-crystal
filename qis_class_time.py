from CCDClass import QISCCDFactory

directory = "/home/daq_user/ltaDaemon-master/images/2026-09-04/"
filename  = "proc_1samp_full_image"
ccd_data = QISCCDFactory(directory+filename, "46")

ccd_data.load_images(4)
ccd_data.read_header()
ccd_data.process_overscan()
ccd_data.plot_active_area()
#ccd_data.plot_fits()
ccd_data.measure_row_drift()
ccd_data.subtract_overscan()
#ccd_data.plot_active_area()
ccd_data.fit_multi_gaussian()

