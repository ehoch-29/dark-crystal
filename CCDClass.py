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

    def __init__(self, image_name, image_num, lta_num):
        """
        parameters
        _________
        filename: string
            CCD file you want processed
        """
        self.image_name = image_name
        self.image_nums = image_num
        self.lta_num = lta_num
        ltanums = []
        for n in range(lta_num):
            ltanums.append(str(n+1))
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
        self.nAmp  = self.lta_num*4
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
            self.overscans[n] = self.hduls[n][:,self.overscan_start:]
            self.overscans[n] = sigma_clip(self.overscans[n], sigma=3)
            values = self.overscans[n].flatten().tolist()
            values = np.array(values, dtype=float)
            print(np.nanmin(values), np.nanmax(values))
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
            for r in range(self.nrows):
                hdul_slice = self.overscans[n][r:r+1, :]
                slice_list = hdul_slice.flatten().tolist()
                if not slice_list:
                    max_bin = 0
                    #print("row is empty")
                else:
                    slice_list = np.array(slice_list, dtype=float)
                    #print(r, slice_list)
                    bins = np.arange(np.nanmin(slice_list), np.nanmax(slice_list), 5)

                    slice_counts, bin_edges = np.histogram(slice_list, bins=bins, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    peak = np.argmax(slice_counts)
                    popt, pcov = curve_fit(gaussian, slice_counts, bin_centers,  p0=(slice_counts[peak], bin_centers[peak], 30))
                    print(slice_counts[peak], bin_centers[peak], popt)
                    max_bin = bin_centers[np.argmax(slice_counts)]
                    zero_peak.append(max_bin)
                    y = gaussian(bin_centers, popt[0], popt[1], popt[2])

                    ax.hist(slice_list, bins=bins, density=False, histtype = 'step')
                    ax.plot(bin_centers, y)
                #ax.plot(np.arange(self.nrow), zero_peak)
            self.zero_peaks[n] = zero_peak
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
        self.active_area = {}
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
      
        for n in range(self.nAmp):
            ax = axes[n]
            values = []
            
            for r in range(self.nrow):
                # --- Find the overscan zero peak for this row ---
                hdul_slice = self.overscans[n][r:r+1, :]
                slice_arr = hdul_slice.flatten().astype(float)

                if slice_arr.size == 0 or np.all(np.isnan(slice_arr)):
                    max_bin = 0.0
                else:
                    vmin, vmax = np.nanmin(slice_arr), np.nanmax(slice_arr)
                    if vmax - vmin < 5:
                        # Degenerate row — use the mean instead
                        max_bin = np.nanmean(slice_arr)
                    else:
                        bins = np.arange(vmin, vmax, 5)
                        counts, bin_edges = np.histogram(slice_arr, bins=bins)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        max_bin = bin_centers[np.argmax(counts)]
                        print(max_bin)
                # --- Extract active area, subtract overscan, then sigma clip ---
                raw_row = self.hduls[n][r:r+1, :self.overscan_start].astype(float)
                print("active_area: ", np.nanmin(raw_row))
                corrected_row = raw_row - max_bin
                print("active area after subtraction: ", np.nanmin(corrected_row), np.nanmax(corrected_row))
                # Sigma clip after subtraction so clipping is on calibrated values
                clipped = sigma_clip(corrected_row, sigma=3)
                clipped = np.ma.masked_invalid(clipped)
                self.active_area[r, n] = clipped
                values.extend(clipped.compressed())  # compressed() drops masked values

            # --- Plot histogram for this amplifier ---
            values = np.array(values, dtype=float)
            if values.size > 0:
                vmin = np.nanmin(values)
                print(vmin)
                bins = np.arange(-150, 1500, 20)
                ax.hist(values, bins=bins, density=False, histtype='step')
            ax.set_title(f'Amp {n}')

        plt.tight_layout()
        plt.show()
        
"""
    def subtract_overscan(self):
        self.active_area = {}
        fig, axes = plt.subplots(4, 4, figsize = (8, 8))
        axes = axes.flatten()
        for n in range(self.nAmp):
            zero_peak = []
            ax = axes[n]
            values = []
            for r in range(self.nrow):
                self.active_area[r, n] = self.hduls[n][r:r+1, :self.overscan_start]
                self.active_area[r, n] = sigma_clip(self.active_area[r, n], sigma=2)
                hdul_slice = self.overscans[n][r:r+1, :]
                slice_list = hdul_slice.flatten().tolist()
                if not slice_list:
                    max_bin = 0
                    #print("row is empty")
                else:
                    slice_list = np.array(slice_list, dtype=float)
                    #print(r, slice_list)
                    bins = np.arange(np.nanmin(slice_list), np.nanmax(slice_list), 5)

                    slice_counts, bin_edges = np.histogram(slice_list, bins=bins, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    max_bin = bin_centers[np.argmax(slice_counts)]
                    for pixel in self.active_area[r, n]:
                        print(np.nanmax(pixel), np.nanmin(pixel))
                        pixel = pixel - max_bin
                        self.active_area[r , n] = pixel
                        print(np.nanmax(pixel), np.nanmin(pixel))

                    value = self.active_area[r, n].flatten().tolist()
                    values = values + value
            values = np.array(values, dtype=float)
            print(np.nanmin(values), np.nanmax(values))
            bins = np.arange(np.nanmin(values), np.nanmin(values)+1500, 20)
            ax.hist(values, bins=bins, density=False, histtype = 'step')
        plt.show()
"""
