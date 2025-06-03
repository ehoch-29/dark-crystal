import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import sys
import pandas as pd
import argparse


def func(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))


def double_gauss(x, a, b, x1, sigma2):
        return a*np.exp(-(x-370)**2/(2*13.5**2)) + b*np.exp(-(x-x1)**2/(2*sigma2**2))


def match_baseline(cube, no_cube):
        print(np.median(cube))
        print(np.median(no_cube))
        min_to_match = cube[0]
        higher_min = no_cube[0]
        #diff = higher_min - min_to_match
        diff = np.median(no_cube) - np.median(cube)
        return no_cube - diff

def extract_data_and_plot(folder, label):
        save_file = os.path.join(folder, out_suffix)
        if os.path.isfile(save_file):
                print(save_file)
                df = pd.read_csv(save_file)
                sorted_power = df.iloc[0, :].to_numpy()
                sorted_power_norm = sorted_power/max(sorted_power)
                sorted_photo_diode = df.iloc[1, :].to_numpy()
                sorted_counts = df.iloc[2, :].to_numpy()
                sorted_error_cts = df.iloc[3, :].to_numpy()
                sorted_error_pwr = df.iloc[4, :].to_numpy()
                waves = df.columns.values
                sorted_waves = np.array(list(map(float, waves)))
                peak = np.argmax(sorted_power)
        if args.avgcts: ax.errorbar(sorted_waves, sorted_counts, yerr = sorted_error_cts, label = label + " CCD counts")
        if args.ccd:    ax.errorbar(sorted_waves[sorted_power > 1], sorted_power[sorted_power > 1], yerr=sorted_error_pwr[sorted_power > 1]*1000, label = label + ", CCD")
        if args.pd :    plt.semilogy(sorted_waves[sorted_photo_diode != 0], sorted_photo_diode[sorted_photo_diode != 0], label = label + ", PD")
        
"""
All the folders that we currently have:



#filter scans
2025_01_15_filterscan
2025_01_17_filter4test
2025_01_29_filter3scan					
2025_01_29_filter4scan 

#Ones with puc
2025_01_24_puck
2025_01_27_puck_filter4
2025_02_03_broadband_puck

#Broadband/Dark
2025_01_21_broadband
2025_01_28_lampon_shutteropen			
2025_01_29_lampon_shutteroff
2025_01_29_lampon_shutteropen_take2
2025_01_29_lampon_shutteroff_take2
2025_02_04_broadband_nopuck

#Cube + filter3
2025_01_16_cube
2025_01_17_cube_60sec
2025_01_31_cube_filter3


#Cube + filter 4
2025_01_21_filter4_cube
2025_01_31_cube_filter4



#other ones
2025_01_27_full_filter4
2025_01_21_dark
2025_01_27_full_image
2025_01_22_800nmfilter
"""

parser = argparse.ArgumentParser('Parse text in the file')
parser.add_argument( '-c', '--ccd'   , action = 'store_true', help = 'include -c flag to plot ccd power')
parser.add_argument( '-p', '--pd'    , action = 'store_true', help = 'include -p flag to plot photodiode power')
parser.add_argument( '-a', '--avgcts', action = 'store_true', help = 'include -a flag to plot average counts')

args = parser.parse_args()

#get the folder we are processing in this run of the code
#folders = ["2025_03_19_led_60sec", "2025_03_18_led_cube_60sec", "2025_03_19_led_cube_30sec", "2025_03_19_led_cube_10sec", "2025_03_20_dark"]
#folders = ["2025_03_20_led_epoxy", "2025_03_19_led_60sec"]
folders = ["2025_03_26_led_cube_30sec"]
#folders = ["2025_02_07_cube_5sec", "2025_02_06_cube_filter4"]
out_suffix = "save_data.csv"

#labels = ["60 sec", "60 sec cube", "30 sec cube", "10 sec cube", "dark"]
labels = ["cube data"]
Filter = "Filter 4 "
title_name = "Comparison of Fermilab Data with Literature Values"
pl = pd.read_csv("~/Downloads/lit_pl.csv")
#pl.plot(x='wavelength', y='normalization', style='o', label = "Literature data")

fig, ax = plt.subplots()
for i, folder in enumerate(folders):
        print(labels[i])
        extract_data_and_plot(folder, labels[i])

#if args.ccd or args.pd: plt.ylabel("Power per area (watts/m^3)")
if args.ccd or args.pd: plt.ylabel("Relative units")
#if args.ccd or args.pd: plt.ylabel("rel units")
if args.avgcts: plt.ylabel("Average Counts (ADU)")
ax.set_yscale("log")
plt.xlabel("Wavelength (nm)")
plt.title(title_name)
plt.xlim(250,700)
plt.legend()
plt.show()

#if we have already run the processing code on this folder just load in the saved .csv and skip to plotting it
plt.show()
sys.exit()
