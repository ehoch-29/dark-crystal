import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import csv
import argparse

#various stat packages to import                                                                                                  
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import quad


#Define gaussians for fitting 
def multi_gaussian(x, *params):
        """Sum of N gaussians. Params: [std1, ampl1, mean1, amp2, mean2, , ...]"""
        y = np.zeros_like(x)
        #print(params)
        #print(len(params))
        for i in range(1, len(params), 2):
                amp = float(params[i])
                mean = float(params[i+1])
                std = float(params[0])
                #print(i, "amp: ", amp, "mean: ",  mean, "std: ",  std)
                y += amp * np.exp(-((x - mean) ** 2) / (2 * std **2))
        return y

def gaussian(x, amp, mean, std):
        y = amp * np.exp(-((x-mean) ** 2) / (2 * std ** 2))
        return y

class MultiSampleData:
        def __init__(self, nsamp, gain, width, darkrate, time, hdu, unmasked, single_e, no_neighbors):
                self.nsamp = nsamp
                self.gain = gain
                self.width = width
                self.darkrate = darkrate
                self.time = time
                self.hdu = hdu
                self.unmasked = unmasked
                self.single_e = single_e
                self.no_neighbors = no_neighbors

        def __str__(self):
                return f"The darkrate for data with {self.nsamp} is {self.darkrate}"

hdus = [0,1,2,3]

def check_neighbors(hdul, index, neighbors, lower_bound):
        i, j = index
        neighbor_offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)
                ]
        check_indices = [(i + di, j + dj) for di, dj in neighbor_offsets]
        rows, col = hdul.shape
        valid_indices = [
                (ni, nj)
                for ni, nj in check_indices
                if 0 <= ni < rows and 0 <= nj < col
             ]

        zero_count = 0
        event = False
        for j in valid_indices:
                if hdul[j] < lower_bound:
                      zero_count += 1
        if zero_count == 8:
                event = True
        neighbors.append(zero_count)
        return event

def cosmic_masks(hdul, index, mask):
        """
        Take any pixel with >100 e- and make a mask that goes out 40 pixels in each direction
        Inputs: data from one amplifier, the index of a pixel that is >100 e-
        """
        radius = 1
        i, j = index
        rows, cols = hdul.shape

        # Create a grid of relative indices
        # Create a grid of relative indices
        dy = np.arange(-radius, radius)
        dx = np.arange(-radius, radius)
        dx_grid, dy_grid = np.meshgrid(dx, dy)

        mask_area = dx_grid**2 + dy_grid**2 < radius**2  # boolean circle mask

        # Get the absolute indices
        x_indices = dx_grid[mask_area] + j
        y_indices = dy_grid[mask_area] + i

        # Clip to stay within array bounds
        valid = (0 <= x_indices) & (x_indices < cols) & (0 <= y_indices) & (y_indices < rows)
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]

        # Apply the mask
        mask[y_indices, x_indices] = False

def plot_fits(hdul, title, axis, fig):
        im = axis.imshow(hdul, cmap='viridis', origin='upper', norm = 'log')  # or 'gray' or 'plasma'
        fig.colorbar(im, label='ADU', ax = axis)  # optional color bar
        axis.set_title(title)
        axis.set_xlabel("Column Index")
        axis.set_ylabel("Row Index")
        plt.show()

def fit_multi_gaussian(counts, bin_centers):
        #don't find peaks less than 0
        mask = bin_centers >= -500
        filtered_counts = counts[mask]
        filtered_centers = bin_centers[mask]
        #use find peaks to get initial guesses at location and heights of gaussians
        peaks, _  = find_peaks(filtered_counts, distance = 5, prominence = (0.0001, None), height = 100)
        
        #convert the result from find_peaks into an initial guess that can be used in the function
        amps  = []
        initial_guess = [25]
        width = 25
        offset = sum(~mask)
        offset_peaks = [item + offset for item in peaks]
        for p in offset_peaks:
                initial_guess.append(counts[p])
                initial_guess.append(bin_centers[p])
                #initial_guess.append(width)

        if len(peaks) == 0:
                initial_guess = [0.005, 0, 200]
        try:
                print("Amplitudes of Gaussians: ", amps)
                popt, pcov = curve_fit(multi_gaussian, filtered_centers, filtered_counts,  p0=initial_guess)
                print(pcov)
        except:
                print("data could not be fit with multigaussian")
                popt = [1, 1, 1, 2, 3]
                pcov = [0, 0, 0]
        print("Optimal fit: ", popt)
        heights = []
        centers = []
        widths = [popt[0]]
        for i in range(1, len(popt), 2):
                heights.append(float(popt[i]))
                centers.append(float(popt[i+1]))
                #widths.append(float(popt[i+2]))
        #print("heights are: ", heights)
        #print("centers are: ", centers)
        #print("widths  are: ", widths)
        return heights, centers, widths, offset_peaks

def one_e_parameters(amp, mean, std, time):
        """
        integrate under the 1e- guassian and calculate the 2 sigma bounds for what is 1e-
        """
        #amp = popt[3]
        #mean = popt[4]                
        #std = popt[5]
        area, err = quad(lambda x: gaussian(x, amp, mean, std), -np.inf, np.inf)
        print("area under 1e: ", area)
        print("first peak: ", mean)
        print("per pixel 1e noise: ", area/time)
        one_e_min = mean - 2*std
        one_e_max = mean + 2*std
        return one_e_min, one_e_max

def check_gauss_fit(heights, centers, widths, bin_centers, ax, data, offset_peaks, counts, file_path):
        #adjust the data such that the 0 e- peak is at 0 and plot the multi-gaussian fit
        offset_centers = [item - centers[0] for item in centers]
        plot_params = []
        bins = np.arange(np.nanmin(data), np.nanmax(data), 20)
        plot_params.append(widths)
        for p in range(len(offset_centers)):
                plot_params.append(heights[p])
                plot_params.append(offset_centers[p])
                #plot_params.append(widths)
        x_fit = np.arange(min(bin_centers), max(bin_centers), 20)
        y_fit = multi_gaussian(x_fit, *plot_params)
        #ax.plot(x_fit, y_fit, label='Fitted Sum of Gaussians', color='red')
        
        #Plot individual Gaussians
        for i in range(1, len(plot_params), 2):

                amp, mean = plot_params[i:i+2]
                std = plot_params[0]
                y_component = amp * np.exp(-((x_fit - mean) ** 2) / (2 * std ** 2))
                
                #ax.plot(x_fit, y_component, '--', label=f'Gaussian {i//3 + 1}')
        
        #set the x ticks to say the number of e- that it corresponds to
        #custom_xtick_loc    = np.arange(0, 1500, int(sum(gain)/len(gain)))
        #custom_xtick_values = np.arange(0, len(custom_xtick_loc), 1)
        #ax.set_xticks(custom_xtick_loc)
        #ax.set_xticklabels(custom_xtick_values)


        #plot the data, peaks, and gaussians
        
        offset_list = [item - centers[0] for item in data]
        offset_bins = np.array([item - centers[0] for item in bin_centers])

        ax.hist(offset_list, bins=bins, density=False, histtype = 'step', label=file_path)
        ax.plot(offset_bins[offset_peaks], counts[offset_peaks], 'x', label = 'Peaks', color = 'red')
        ax.set_xlim(-400, 2000)
        ax.set_xlabel('Pixel Value (e-)')


def update_evolFile(filename, newdata, evolfile):
    if os.path.exists(evolfile):                                                # Check if the file exists                        
        df = pd.read_csv(evolfile, sep="\t")                                    # Load existing data                              
    else:
        df = pd.DataFrame(columns = ["ANSAMP", "f0", "EXP"])
        #df = pd.DataFrame(columns=["RUNID", "Date start", "Date end", "ANSAMP", "MCMID", "Bond OK", "Data OK", "Median per row", "Median per col", "Noise", "Noise error", "Gain", "Gain error", "SER", "SER error"])   # Create an empty DataFrame with the corrct columns                                                                                                                        
    #df = df[~((df["Date start"] == datestart) & (df["MCMID"] == mcmid))]        # Remove existing rows for the given MCMID and dateStart                                                                                                                          
    newdf = pd.DataFrame(newdata)                                               # Create a DataFrame for the new data
    updateddf = pd.concat([df, newdf], ignore_index=True)                       # Append the new data
    updateddf.to_csv(evolfile, sep="\t", index=False) 
