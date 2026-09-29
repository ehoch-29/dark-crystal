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

    # Usable region in physical (unbinned) CCD pixels, [start, stop). Empirically
    # smaller than the nominal active area; set on the instance to change it.
    crop_rows = (0, 1035)
    crop_cols = (7, 515)

    def __init__(self, image_name, image_num, file_type):
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
        self.ncol = int(self.header['NCOL'])
        self.samples = float(self.header['NSAMP'])
        self.nCCDcol = int(self.header["CCDNCOL"])
        self.nCCDrow = int(self.header["CCDNROW"])
        self.nprescan = int(self.header["CCDNPRES"])

        # Binning and skipped pixels (absent from unbinned headers -> defaults).
        # NROW/NCOL are already binned counts; SKIP* are in physical pixels.
        self.rbin = int(self.header.get("NBINROW", 1))
        self.cbin = int(self.header.get("NBINCOL", 1))
        self.skiprow = int(self.header.get("SKIPROW", 0))
        self.skipcol = int(self.header.get("SKIPCOL", 0))

        # Physical pixel bounds -> indices in the (binned) image.
        # The usable region is smaller than the nominal active area, so it is cropped
        # to crop_rows/crop_cols (physical, unbinned CCD pixels, half-open), which
        # scale automatically with binning and skipping.
        def to_index(phys, skip, nbin, up):
            x = (phys - skip) / nbin
            return max(0, int(np.ceil(x) if up else np.floor(x)))

        first_row = to_index(self.crop_rows[0], self.skiprow, self.rbin, up=True)
        last_row = to_index(min(self.crop_rows[1], self.nCCDrow), self.skiprow, self.rbin, up=False)
        first_col = to_index(max(self.crop_cols[0], self.nprescan), self.skipcol, self.cbin, up=True)
        last_col = to_index(min(self.crop_cols[1], self.nCCDcol), self.skipcol, self.cbin, up=False)
        self.active_rows = last_row - first_row
        self.active_cols = last_col - first_col
        self.row_slice = slice(first_row, last_row)
        self.col_slice = slice(first_col, last_col)
        # the overscan starts after the full physical CCD, not after the crop
        self.overscan_slice = slice(to_index(self.nCCDcol, self.skipcol, self.cbin, up=False), None)
        self.npixels = self.active_rows*self.active_cols
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
            self.overscans[n] = self.hduls[n][:,self.overscan_slice]
            self.overscans[n] = sigma_clip(self.overscans[n], sigma=3)
            values = self.overscans[n].flatten().tolist()
            values = np.array(values, dtype=float)
            print(np.nanmin(values), np.nanmax(values))
            bins = np.arange(np.nanmin(values), np.nanmin(values)+1500, 20)
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
        try:
            self.active_areas
        except:
            self.active_areas = {}
            for n in range(self.nAmp):
                self.active_areas[n] = self.hduls[n][self.row_slice,self.col_slice]
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
        ydim, xdim = self.active_areas[0].shape
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
            for r in range(self.active_rows):
                overscan_single_e = self.overscan_e_peak[n][r]
                if self.good_hdus[n] == 1:
                    self.active_areas[n][r] = self.active_areas[n][r] - overscan_single_e
                else:
                    self.active_areas[n] = np.zeros_like(self.active_areas[n])
    def fit_multi_gaussian(self):
        """
        fit a multi_gaussian to the active area
        """
        print("fitting multi gaussian")
        n_amps_to_show = 4
        fig, axes = plt.subplots(1, 4, figsize =(8,8))
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
                raw_row = self.hduls[n][r:r+1, self.col_slice].astype(float)
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
