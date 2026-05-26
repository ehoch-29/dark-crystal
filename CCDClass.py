import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.stats import sigma_clip


def gaussian(x, amp, mean, std):
    y = amp * np.exp(-((x-mean) ** 2) / (2 * std ** 2))
    return y

class QISCCDFactory:
    """
    Factory class for processing CCD images

    Workflow:
    tbd
    """

    def __init__(self, image_name, image_num):
        """
        parameters
        _________
        filename: string
            CCD file you want processed
        """
        self.image_name = image_name
        self.image_nums = image_num
        ltanums = [str(1), str(2), str(3), str(4)]
        self.filenames = {}
        for i, lta in enumerate(ltanums):
            self.filenames[i] = self.image_name + "_" + lta + "_" + image_num + ".fits"

        
    def load_images(self, nAmp):
        """
        Loads in the hdul files for each amplifier and takes the
        header from the first file since all the CCD values should be the same
        across the image
        """
        self.hduls = {}
        self.nAmp  = 16
        hdul_dummy, self.header = fits.getdata(self.filenames[0], header = True)
        for i,f in enumerate(self.filenames.values()):
            print(f)
            for n in range(nAmp):
                self.hduls[4*i+n] = fits.getdata(f, n)
        
    def read_header(self):
        """
        Exports various parameters from the CCD header
        """
        self.nrow = int(self.header['NROW'])
        self.samples = float(self.header['NSAMP'])
        self.nCCDcol = int(self.header["CCDNCOL"])
        self.overscan_start = int(self.nCCDcol + 10)
        
    def process_overscan(self):
        """
        Pulls out the overscan and does things
        """
        fig, axes = plt.subplots(4, 4, figsize = (8, 8))
        axes = axes.flatten()
        
        self.overscans = {}
        for n in range(self.nAmp):
            ax = axes[n]
            self.overscans[n] = self.hduls[n][:5,self.overscan_start:]
            self.overscans[n] = sigma_clip(self.overscans[n], sigma=3)
            values = self.overscans[n].flatten().tolist()
            values = np.array(values, dtype=float)
            bins = np.arange(np.nanmin(values), np.nanmin(values)+1500, 20)
            ax.hist(values, bins=bins, density=False, histtype = 'step')

        plt.show()
            
    def measure_row_drift(self):
        """
        Looks at the single electron peak across the different rows of the overscan
        to show whether there is a drift

        TODO: make graph nice
        """
        
        fig, axes = plt.subplots(4, 4, figsize =(8,8))
        axes = axes.flatten()
        for n in range(self.nAmp):
            ax = axes[n]
            zero_peak = []
            for r in range(self.nrow):
                hdul_slice = self.overscans[n][r:r+1, :]
                slice_list = hdul_slice.flatten().tolist()
                if not slice_list:
                    max_bin = 0
                    #print("row is empty")
                else:
                    slice_list = np.array(slice_list, dtype=float)
                    #print(r, slice_list)
                    bins = np.arange(np.nanmin(slice_list), np.nanmax(slice_list), 20)

                    slice_counts, bin_edges = np.histogram(slice_list, bins=bins, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    peak = np.argmax(slice_counts)
                    popt, pcov = curve_fit(gaussian, slice_counts, bin_centers,  p0=(slice_counts[peak], bin_centers[peak], 25))
                    print(slice_counts[peak], bin_centers[peak], popt)
                    max_bin = bin_centers[np.argmax(slice_counts)]
                    zero_peak.append(max_bin)
            ax.plot(np.arange(self.nrow), zero_peak)
        plt.show()

    def plot_fits(self):
        """
        Plot the fits image to look at it
        """
        fig, axes = plt.subplots(4, 4, figsize =(8,8))
        axes = axes.flatten()
        for n in range(self.nAmp):
            ax = axes[n]
            im = ax.imshow(self.hduls[n], cmap='viridis', origin='upper', norm = 'log')  # or 'gray' or 'plasma'
            fig.colorbar(im, label='ADU', ax = ax)  # optional color bar
        plt.show()
        
  
    def subtract_overscan(self):
        "todo"
        
