import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval, ImageNormalize
from scipy.signal import find_peaks

def gaussian(x, amp, mean, std):
    y = amp * np.exp(-((x-mean) ** 2) / (2 * std ** 2))
    return y

def multi_gaussian(x, *params):
        """Sum of N gaussians. Params: [std1, ampl1, mean1, amp2, mean2, , ...]"""
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = float(params[i])
            mean = float(params[i+1])
            std = float(params[i+2])
            #print(i, "amp: ", amp, "mean: ",  mean, "std: ",  std)                                                                                                      
            y += amp * np.exp(-((x - mean) ** 2) / (2 * std **2))
        return y

class QISCCDFactory:
    """
    Factory class for processing CCD images

    Workflow:
    tbd
    """

    def __init__(self, image_name, image_num, file_type):
        """
        parameters
        _________
        filename: string
            CCD file you want processed
        """
        self.image_name = image_name
        self.image_nums = image_num
        self.ltanums = [str(1), str(2), str(3), str(4)]
        self.filenames = {}
        for i, lta in enumerate(self.ltanums):
            self.filenames[i] = self.image_name + "_" + lta + "_" + image_num + file_type
        self.box_style = dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='blue')
        self.good_hdus = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        
    def load_images(self, nAmp):
        """
        Loads in the hdul files for each amplifier and takes the
        header from the first file since all the CCD values should be the same
        across the image
        """
        self.hduls = {}
        self.nAmp  = len(self.ltanums)*4
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
        self.ncol = int(self.header['NCOL'])
        self.samples = float(self.header['NSAMP'])
        self.nCCDcol = int(self.header["CCDNCOL"])
        self.nCCDrow = int(self.header["CCDNROW"])
        self.overscan_start = int(self.nCCDcol)
        self.npixels = self.nrow*self.nCCDcol
        self.exptime = 49.61*60 #to-do, figure out how to make this not hardcoded, for 150 rows and 400 samples
        print(self.header)

        
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
            #self.overscans[n] = sigma_clip(self.overscans[n], sigma=6)
            values = self.overscans[n].flatten().tolist()
            values = np.array(values, dtype=float)
            bins = np.arange(np.nanmin(values), np.nanmax(values), 20)
            ax.hist(values, bins=bins, density=False, histtype = 'step')
            #ax.set_yscale('log')
        plt.suptitle("Histogram of Overscan values")
        #plt.show()

                
    def plot_active_area(self):
        """
        Pulls out the overscan and does things
        """
        fig, axes = plt.subplots(4, 4, figsize = (8, 8))
        axes = axes.flatten()
        self.override_row = 1035
        self.override_col = 515
        try:
            self.active_areas
        except:
            self.active_areas = {}
            for n in range(self.nAmp):
                self.active_areas[n] = self.hduls[n][:self.override_row,7:self.override_col]
                self.active_areas[n] = sigma_clip(self.active_areas[n], sigma=5)

        for n in range(self.nAmp):
            ax = axes[n]
            values = self.active_areas[n].flatten().tolist()
            values = np.array(values, dtype=float)
            bins = np.arange(0, 100, 20)
            #bins = np.arange(np.nanmin(values), np.nanmax(values), 20)
            ax.hist(values, bins=bins, density=False, histtype = 'step')
            #ax.set_yscale('log')
        plt.suptitle("Histogram of Active Area Values")
        plt.show()

    def measure_row_drift(self):
        """
        Looks at the single electron peak across the different rows of the overscan
        to show whether there is a drift

        TODO: make graph nice
        """
        print("calculating the overscan single electron peak")
        fig, axes = plt.subplots(4, 4, figsize =(8,8))
        axes = axes.flatten()
        self.gains = {}
        self.overscan_e_peak = {}
        rows = np.arange(self.nrow)
        for n in range(self.nAmp):
            ax = axes[n]
            zero_peak = []
            for r in range(self.nrow):
                hdul_slice = self.overscans[n][r:r+5, :]
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

                    if n == 18:
                        peaks, _  = find_peaks(slice_counts, distance = 5, prominence = (0.0001, None), height = 5)
                        ax.plot(bin_centers[peaks], slice_counts[peaks], 'x')
                        initial_guess = []
                        width = 35
                        for p in peaks:
                            initial_guess.append(slice_counts[p])
                            initial_guess.append(bin_centers[p])
                            initial_guess.append(width)
                        popt, pcov = curve_fit(multi_gaussian, bin_centers, slice_counts,  p0=initial_guess)
                        y = multi_gaussian(bin_centers, *initial_guess)
                        ax.plot(bin_centers, y)
                    else:
                        peak = np.argmax(slice_counts)
                        popt, pcov = curve_fit(gaussian, bin_centers, slice_counts,  p0=(slice_counts[peak], bin_centers[peak], 25))
                        residuals = slice_counts - gaussian(bin_centers, *popt)
                        mask = slice_counts > 0
                        chi2 = np.sum((residuals[mask] ** 2) / slice_counts[mask])
                        n_dof = np.sum(mask) - len(popt)
                        chi2_reduced = chi2 / n_dof
                        max_bin = bin_centers[np.argmax(slice_counts)]
                        zero_peak.append(popt[1])
                        y = gaussian(bin_centers, popt[0], popt[1], popt[2])
                        #ax.plot(bin_centers, y)
            ax.plot(rows, zero_peak)
                    #ax.hist(slice_list, bins=bins, density=False, histtype = 'step')
                
            self.overscan_e_peak[n] = zero_peak
        #print(self.gains)
        #print(self.single_e_peak)
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
        

    def stitch_image(self):
        """
        Stitching together the 16 amplifiers into one image and plotting it

        Mapping:
        4.3 4.1 4.2 4.0 2.3 2.1 2.2 2.0
        3.0 3.2 3.3 3.1 1.0 1.2 1.3 1.1

        15  13  14  12   7  5   6   4
         8  10  11   9   0  2   3   1
        """
        Mapping = [8, 10, 11, 9, 0, 2, 3, 1, 15, 13, 14, 12, 7, 5, 6, 4]
        xdim = min(self.ncol, self.nCCDcol, self.override_col)-7
        ydim = min(self.nrow, self.nCCDrow, self.override_row)
        print(xdim, ydim)
        full_image = np.full((ydim*2, xdim*8), np.nan)
                
        for x in np.arange(1, 3):
            for y in np.arange(1,9):
                n = (y-1)+(x-1)*8
                print(n, x, y)
                print(ydim*(x-1),ydim*x, xdim*(y-1),xdim*y)
                if x == 1:
                    full_image[ydim*(x-1):ydim*x, xdim*(y-1):xdim*y] = self.active_areas[Mapping[n]][:,:]
                else:
                    full_image[ydim*(x-1):ydim*x, xdim*(y-1):xdim*y] = self.active_areas[Mapping[n]][::-1,:]
        plt.imshow(full_image, origin="lower", cmap="gray", interpolation="nearest",
           norm=ImageNormalize(full_image, interval=ZScaleInterval()))
        plt.title(self.image_name + self.image_nums)
        plt.colorbar()
        plt.show()
    def subtract_overscan(self):
        """
        Take the peak from the overscan and use that as the baseline to subtract off the active area
        """
        print("subtracting the overscan")
        for n in range(self.nAmp):
            for r in range(self.override_row):
                overscan_single_e = self.overscan_e_peak[n][r]
                if self.good_hdus[n] == 1:
                    self.active_areas[n][r] = self.active_areas[n][r] - overscan_single_e
                else:
                    self.active_areas[n] = np.zeros((self.override_row, self.override_col-7))
    def fit_multi_gaussian(self):
        """
        fit a multi_gaussian to the active area
        """
        print("fitting multi gaussian")
        n_amps_to_show = 4
        fig, axes = plt.subplots(1, 4, figsize =(8,8))
        axes = axes.flatten()
        self.noise = {}
        self.darkcounts = {}
        for n in range(n_amps_to_show):
            print(n)
            ax = axes[n]
            slice_list = self.active_areas[n].flatten().tolist()
            slice_list = np.array(slice_list, dtype=float)
            bins = np.arange(np.nanmin(slice_list), np.nanmax(slice_list), 20)
            slice_counts, bin_edges = np.histogram(slice_list, bins=bins, density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            peaks, _  = find_peaks(slice_counts, distance = 5, prominence = (0.0001, None), height = 5)
            ax.plot(bin_centers[peaks], slice_counts[peaks], 'x')
            initial_guess = []
            width = 35
            for p in peaks:
                initial_guess.append(slice_counts[p])
                initial_guess.append(bin_centers[p])
                initial_guess.append(width)
            try:
                popt, pcov = curve_fit(multi_gaussian, bin_centers, slice_counts,  p0=initial_guess)
            except:
                y = multi_gaussian(bin_centers, *initial_guess)
                print("fit did not converge")
            else:
                y = multi_gaussian(bin_centers, *popt)
                print(popt)
                self.gains[n] = popt[4]-popt[1]
                self.noise[n] = (popt[2] + popt[5])/2
                lower_limit = popt[4]-3*self.noise[n]
                upper_limit = popt[4]+3*self.noise[n]
                dark_count = np.count_nonzero((slice_list >= lower_limit))# & (slice_counts <= upper_limit))
                print(lower_limit, upper_limit)
                print(dark_count)
                
                self.darkcounts[n] = dark_count/self.npixels/self.exptime
            line1 =ax.hist(slice_list, bins=bins, density=False, histtype='step', 
                    linewidth=2, color='navy', label='Data')
            line2 =ax.plot(bin_centers, y, linewidth=2.5, color='red', label='Multi-Gaussian Fit')
            #line3 =ax.axvline(popt[4], color='green', linestyle='--', alpha=0.7, label='Single e⁻ peak')
    
            #ax.set_title(f'Amplifier {n}', fontsize=14, fontweight='bold')
            #fig.legend([line1, line2, line3], ['Data', 'Multi-Gaussian Fit', 'Single e- peak'])
            ax.set_xlabel('ADU', fontsize=14)
            ax.set_ylabel('Counts', fontsize=14)
            #ax.plot(bin_centers, y)
            #ax.hist(slice_list, bins=bins, density=False, histtype = 'step')
        plt.tight_layout()    
        plt.show()
        print(self.gains)
        print(self.noise)
        print(self.darkcounts)
