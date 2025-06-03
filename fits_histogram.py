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

#needed constants
h = 6.626e-34 #plank's constant
c = 3e8 #speed of light
scale = (0.01)**2  #size of the photodiode


def multi_gaussian(x, *params):
        """Sum of N gaussians. Params: [ampl1, mean1. std1, amp2, mean2, std2, ...]"""
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
                amp = params[i]
                mean = params[i+1]
                std = params[i+2]
                y += amp * np.exp(-((x - mean) ** 2) / (2 * std **2))
        return y

def get_counts(file_path):
        for n in range(4):
                #take the data from the fits file and turn it into a list of pixel values
                hdul = fits.getdata(file_path, n)
                print(np.shape(hdul))
                filter_data = sigma_clip(hdul[:, :3072])
                count_list = hdul[:,:3072][~filter_data.mask].flatten().tolist()
                print(count_list)
                bins = np.arange(-1000, 1000, 20)

                plt.title(n)
                plt.hist(count_list, bins = 100)
                plt.show()
                

                #make it into a histogram for fitting
                counts, bin_edges = np.histogram(count_list, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                #use find peaks to get initial guesses at location and heights of gaussians
                peaks, _  = find_peaks(counts, distance = 5)

                #convert the result from find_peaks into an initial guess that can be used in the function
                initial_guess = []
                width = 25
                for n in peaks:
                        initial_guess.append(counts[n])
                        initial_guess.append(bin_centers[n])
                        initial_guess.append(width)

                #fit the data with a multi-gaussian
                #popt, pcov = curve_fit(multi_gaussian, bin_centers, counts, p0=initial_guess)
                """
                heights = []
                centers = []
                widths = []
                for i in range(0, len(popt), 3):
                        heights.append(float(popt[i]))
                        centers.append(float(popt[i+1]))
                        widths.append(float(popt[i+2]))
                print("Heights are: ", heights)
                print("Centers are: ", centers)
                print("Widths  are: ", widths)
                """
                
                #plot the data, peaks, and gaussians
                plt.hist(count_list, bins=bins, density=True, alpha=0.5, label='Histogram')
                plt.plot(bin_centers[peaks], counts[peaks], 'x', label = 'Peaks', color = 'red')
                """
                x_fit = np.linspace(-1000, 1000, 1000)
                y_fit = multi_gaussian(x_fit, *popt)
                plt.plot(x_fit, y_fit, label='Fitted Sum of Gaussians', color='red')

                # Plot individual Gaussians
                for i in range(0, len(popt), 3):
                        amp, mean, std = popt[i:i+3]
                        y_component = amp * np.exp(-((x_fit - mean) ** 2) / (2 * std ** 2))
                        plt.plot(x_fit, y_component, '--', label=f'Gaussian {i//3 + 1}')
                """
                plt.legend()
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.title('Fit Multiple Gaussians to Histogram')
                plt.grid(True)
                plt.show()

                
parser = argparse.ArgumentParser('Parse text in the file')
parser.add_argument('filename', help = 'folder you want to process')
parser.add_argument( '-o', '--override', action = 'store_true', help = 'include -o flag to rewrite the save data?')

args = parser.parse_args()

#input the qe data and make a dataset that has a value for every 5 nm using interpolation
qe = pd.read_csv("qe.csv")
full_wavelengths = np.arange(200, qe.iloc[:, 0].max() + 1, 1)
interpolated_qe  = np.interp(full_wavelengths, qe.iloc[:, 0], qe.iloc[:, 1] )
new_qe = pd.DataFrame({"wavelengths": full_wavelengths, "qe": interpolated_qe})
new_qe.set_index('wavelengths', inplace=True)

get_counts(args.filename)
