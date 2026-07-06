from CCDClass import QISCCDFactory

directory = "/home/daq_user/ltaDaemon-master/images/2026-07-02/"
filename  = "proc_400samp_150rows"
ccd_data = QISCCDFactory(directory+filename, "9")

ccd_data.load_images(4)
ccd_data.read_header()
ccd_data.process_overscan()
ccd_data.plot_active_area()
#ccd_data.plot_fits()
ccd_data.measure_row_drift()
ccd_data.subtract_overscan()
#ccd_data.plot_active_area()
ccd_data.fit_multi_gaussian()

