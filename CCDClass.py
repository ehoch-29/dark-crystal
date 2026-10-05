import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval, ImageNormalize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, label

def gaussian(x, amp, mean, std):
    y = amp * np.exp(-((x-mean) ** 2) / (2 * std ** 2))
    return y

def gaussian_2d(coords, amp, x0, y0, sx, sy, theta, offset):
    """Rotated elliptical 2-D gaussian on a constant background, flattened for curve_fit.
    coords = (x, y) grids; sx, sy are the widths along the rotated axes; theta in radians."""
    x, y = coords
    dx, dy = x - x0, y - y0
    xr = dx*np.cos(theta) + dy*np.sin(theta)
    yr = -dx*np.sin(theta) + dy*np.cos(theta)
    return (offset + amp*np.exp(-(xr**2/(2*sx**2) + yr**2/(2*sy**2)))).ravel()

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
    # Active-area pixels above this many ADU (after baseline subtraction) are masked
    max_adu = 2500

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
                # raw values; the high-value cut is applied in subtract_overscan(), after
                # the baseline is removed (the threshold is in baseline-subtracted ADU)
                self.active_areas[n] = np.ma.masked_invalid(self.hduls[n][self.row_slice,self.col_slice])

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

    @staticmethod
    def _zero_peak(values, bin_width=5, min_fit_pixels=100):
        """
        Zero-electron peak position of one row of overscan pixels.

        With enough pixels, histogram the values and fit a gaussian starting from
        the tallest bin. With too few (e.g. binned images have a narrow overscan)
        a histogram is too coarse, so use the mean of the sigma-clipped values.
        Falls back to the same mean if the fit fails.
        Returns (peak, sigma, (bin_centers, counts, fit)) -- the last is None if
        no histogram was fit.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan, np.nan, None
        mean_fallback = (np.mean(values), np.std(values), None)
        vmin, vmax = values.min(), values.max()
        if values.size < min_fit_pixels or vmax - vmin < 2*bin_width:
            return mean_fallback

        bins = np.arange(vmin, vmax + bin_width, bin_width)
        counts, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        peak = np.argmax(counts)
        try:
            # x = ADU bin centres, y = counts (the fit variable order matters)
            popt, _ = curve_fit(gaussian, centers, counts,
                                p0=(counts[peak], centers[peak], np.std(values)))
        except (RuntimeError, ValueError):
            return mean_fallback
        if not (vmin <= popt[1] <= vmax):
            return mean_fallback
        return popt[1], abs(popt[2]), (centers, counts, gaussian(centers, *popt))

    def measure_row_drift(self, plot=True):
        """
        Finds the zero-electron peak of the overscan for every row of every
        amplifier, to show whether the baseline drifts from row to row and so
        it can be subtracted from the active area by subtract_overscan.

        Needs process_overscan() to have been run first.

        Sets
            self.overscan_e_peak[n][r]  baseline for row r of the active area
                                        (indexed like self.active_areas[n])
            self.overscan_sigma[n][r]   width of the peak in that row
        """
        print("calculating the overscan zero peak")
        if not hasattr(self, "overscans"):
            self.process_overscan()

        self.overscan_e_peak = {}
        self.overscan_sigma = {}
        first_row = self.row_slice.start
        rows = np.arange(self.active_rows)

        if plot:
            fig, axes = plt.subplots(4, 4, figsize=(8, 8), sharex=True)
            axes = axes.flatten()

        for n in range(self.nAmp):
            # overscans[n] is a masked array (sigma clipped) with the full image height;
            # active row r sits at image row first_row + r
            peaks = np.full(self.active_rows, np.nan)
            sigmas = np.full(self.active_rows, np.nan)
            for r in rows:
                row = np.ma.filled(self.overscans[n][first_row + r].astype(float), np.nan)
                peaks[r], sigmas[r], _ = self._zero_peak(row)

            # rows with no usable overscan pixels: borrow the amplifier's median baseline
            bad = ~np.isfinite(peaks)
            if bad.all():
                peaks[:] = 0.0
            elif bad.any():
                print(f"amp {n}: {bad.sum()} rows without overscan data, using median")
                peaks[bad] = np.nanmedian(peaks)
            self.overscan_e_peak[n] = peaks
            self.overscan_sigma[n] = sigmas

            print(f"amp {n}: zero peak mean {np.mean(peaks):.1f} ADU, "
                  f"drift (max-min) {np.ptp(peaks):.1f} ADU")
            if plot:
                ax = axes[n]
                ax.plot(rows, peaks, lw=0.8)
                ax.set_title(f"Amp {n}", fontsize=8)
                ax.tick_params(labelsize=6)
        if plot:
            fig.supxlabel("Row")
            fig.supylabel("Zero peak (ADU)")
            plt.suptitle("Overscan zero peak vs row")
            plt.tight_layout()
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
        valid = np.zeros(full_image.shape, dtype=bool)   # False for dead amps
                
        for x in np.arange(1, 3):
            for y in np.arange(1,9):
                n = (y-1)+(x-1)*8
                print(n, x, y)
                print(ydim*(x-1),ydim*x, xdim*(y-1),xdim*y)
                # getdata: keep the sigma-clipped pixels (bright LED light gets clipped as outliers)
                tile = np.ma.getdata(self.active_areas[Mapping[n]])
                if x != 1:
                    tile = tile[::-1,:]
                full_image[ydim*(x-1):ydim*x, xdim*(y-1):xdim*y] = tile
                valid[ydim*(x-1):ydim*x, xdim*(y-1):xdim*y] = self.good_hdus[Mapping[n]] == 1
        self.full_image = full_image
        self.stitch_valid = valid
        plt.imshow(full_image, origin="lower", cmap="gray", interpolation="nearest",
           norm=ImageNormalize(full_image, interval=ZScaleInterval()))
        plt.title(self.image_name + self.image_nums)
        plt.colorbar()
        plt.show()

    def fit_led_spots(self, n_spots=1, bin_factor=None, plot=True, equalize_amps=True):
        """
        Fit n_spots rotated elliptical 2-D gaussians (sharing one constant background)
        to the stitched image, and overlay them on it with the residual underneath.

        Needs stitch_image() to have been run. The fit is done on a block-averaged copy
        of the image (bin_factor x bin_factor); by default the block size is chosen so the
        fit sees roughly 40k pixels (1 for images already binned in hardware). Dead amps
        and NaNs are excluded, and a robust loss keeps hot pixels from dragging the fit.

        Spots are located automatically: the brightest smoothed peak first, then the
        next brightest after removing the first, and so on. Use n_spots = number of
        circles you can see.

        equalize_amps: subtract each amplifier's 25th-percentile level first, so amp-to-amp
            offsets don't dominate the fit (the fitted background is relative to that).

        Returns a dict {"offset": ..., "spots": [ {x0, y0, sigma_1, sigma_2, theta, amp,
        fwhm, err} ... ]} in full-resolution stitched-image pixels; also in self.led_fit.
        """
        img = np.where(self.stitch_valid, self.full_image, np.nan)
        if equalize_amps:
            ydim, xdim = self.active_areas[0].shape
            for i in range(2):
                for j in range(8):
                    tile = img[i*ydim:(i+1)*ydim, j*xdim:(j+1)*xdim]   # a view
                    if np.isfinite(tile).any():
                        tile -= np.nanpercentile(tile, 25)
        b = bin_factor or max(1, int(np.ceil(np.sqrt(img.size/40000))))
        ny, nx = (img.shape[0]//b)*b, (img.shape[1]//b)*b
        blocks = img[:ny, :nx].reshape(ny//b, b, nx//b, b)
        with np.errstate(all="ignore"):
            small = np.nanmean(blocks, axis=(1, 3))
        yy, xx = np.mgrid[:small.shape[0], :small.shape[1]]
        good = np.isfinite(small)

        # --- initial guesses: peak of the smoothed image, width from the half-maximum blob ---
        filled = np.where(good, small, np.nanmedian(small))
        smooth = gaussian_filter(filled, 2)
        offset0 = np.nanpercentile(small, 10)
        guess, lower, upper = [offset0], [-np.inf], [np.inf]
        remaining = smooth - offset0
        for _ in range(n_spots):
            iy, ix = np.unravel_index(np.argmax(np.where(good, remaining, -np.inf)), remaining.shape)
            amp0 = remaining[iy, ix]
            blob, _ = label(remaining > amp0/2)
            w0 = max(np.sqrt(np.count_nonzero(blob == blob[iy, ix])/np.pi)/1.177, 1.0)
            guess += [amp0, ix, iy, w0, w0, 0.0]
            lower += [0, 0, 0, 0.5, 0.5, -np.pi]
            upper += [np.inf, small.shape[1], small.shape[0], small.shape[1], small.shape[0], np.pi]
            remaining = remaining - gaussian_2d((xx, yy), amp0, ix, iy, w0, w0, 0.0, 0.0).reshape(small.shape)

        def model_flat(coords, offset, *spots):
            y = np.full(coords[0].size, offset, dtype=float)
            for k in range(0, len(spots), 6):
                y += gaussian_2d(coords, *spots[k:k+6], 0.0)
            return y

        # robust noise scale so hot pixels are down-weighted (soft L1 beyond ~1 noise sigma)
        noise = 1.4826*np.nanmedian(np.abs((small - smooth)[good]))
        popt, pcov = curve_fit(model_flat, (xx[good], yy[good]), small[good], p0=guess,
                               bounds=(lower, upper), loss="soft_l1", f_scale=max(noise, 1.0))
        perr = np.sqrt(np.diag(pcov))

        # --- binned -> full-resolution pixel coordinates (centre of block i is i*b + (b-1)/2) ---
        scale = np.array([1, b, b, b, b, 1])
        spots = []
        for k in range(1, len(popt), 6):
            amp, x0, y0, sx, sy, th = popt[k:k+6]
            spots.append(dict(amp=amp, x0=x0*b + (b-1)/2, y0=y0*b + (b-1)/2,
                              sigma_1=sx*b, sigma_2=sy*b, theta=th,
                              fwhm=2*np.sqrt(2*np.log(2))*np.array([sx*b, sy*b]),
                              err=perr[k:k+6]*scale))
        self.led_fit = dict(offset=popt[0], spots=spots, bin_factor=b)
        print(f"background offset {popt[0]:.1f} ADU (block size {b})")
        for i, sp in enumerate(spots):
            print(f"spot {i}: centre ({sp['x0']:.1f} +/- {sp['err'][1]:.1f}, "
                  f"{sp['y0']:.1f} +/- {sp['err'][2]:.1f}) px, amp {sp['amp']:.3g}, "
                  f"FWHM {sp['fwhm'][0]:.1f} x {sp['fwhm'][1]:.1f} px, theta {sp['theta']:.2f}")

        if plot:
            model = model_flat((xx, yy), *popt).reshape(small.shape)
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            extent = (0, nx, 0, ny)
            norm = ImageNormalize(small[good], interval=ZScaleInterval())
            axes[0].imshow(small, origin="lower", cmap="gray", norm=norm, extent=extent,
                           interpolation="nearest")
            # one FWHM contour per spot (its own gaussian only), in full-resolution pixels
            xc, yc = (xx + 0.5)*b, (yy + 0.5)*b
            for k in range(1, len(popt), 6):
                one = gaussian_2d((xx, yy), *popt[k:k+6], 0.0).reshape(small.shape)
                axes[0].contour(xc, yc, one, levels=[popt[k]/2], colors="red", linewidths=1.2)
                axes[0].plot(popt[k+1]*b + (b-1)/2, popt[k+2]*b + (b-1)/2, "+", color="red", ms=12)
            axes[0].set_title(f"{self.image_name}{self.image_nums}: {n_spots}-spot 2-D gaussian "
                              "fit (contours = FWHM)")
            resid = np.where(good, small - model, np.nan)
            rlim = np.nanpercentile(np.abs(resid), 99)
            im = axes[1].imshow(resid, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim,
                                extent=extent, interpolation="nearest")
            fig.colorbar(im, ax=axes[1], label="data - fit (ADU)")
            axes[1].set_title("Residual")
            plt.tight_layout()
            plt.show()
        return self.led_fit

    def subtract_overscan(self):
        """
        Take the peak from the overscan and use that as the baseline to subtract off the active area,
        then mask pixels above self.max_adu (set max_adu = None to keep everything)
        """
        print("subtracting the overscan")
        for n in range(self.nAmp):
            for r in range(self.active_rows):
                overscan_single_e = self.overscan_e_peak[n][r]
                if self.good_hdus[n] == 1:
                    self.active_areas[n][r] = self.active_areas[n][r] - overscan_single_e
                else:
                    self.active_areas[n] = np.zeros_like(self.active_areas[n])
            if self.good_hdus[n] == 1 and self.max_adu is not None:
                # mask (not delete) so the values stay available, e.g. for fit_led_spots
                self.active_areas[n] = np.ma.masked_greater(self.active_areas[n], self.max_adu)
                print(f"amp {n}: masked {np.ma.count_masked(self.active_areas[n])} pixels "
                      f"above {self.max_adu} ADU")
                
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
