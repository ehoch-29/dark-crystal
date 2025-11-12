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

def exponential(x, a, b):
        return b*(a**x)


def match_baseline(cube, no_cube):
        print(np.median(cube))
        print(np.median(no_cube))
        min_to_match = cube[0]
        higher_min = no_cube[0]
        #diff = higher_min - min_to_match
        diff = np.median(no_cube) - np.median(cube)
        return no_cube - diff

def extract_data(folder):
        """
        function to extract data from the save_data.csv
        folder: the file folder that contains the data you want
        """
        save_file = os.path.join(folder, out_suffix)
        if os.path.isfile(save_file):
                print(save_file)
                df = pd.read_csv(save_file)
                ccd_power     = df.iloc[0, :].to_numpy()
                pd_power      = df.iloc[1, :].to_numpy()
                ccd_counts    = df.iloc[2, :].to_numpy()
                ccd_pwr_error = df.iloc[3, :].to_numpy()
                ccd_cts_error = df.iloc[4, :].to_numpy()
                waves = df.columns.values
                sorted_waves = np.array(list(map(float, waves)))

        return ccd_power, pd_power, ccd_counts, ccd_pwr_error, ccd_cts_error, sorted_waves



def subtract_dark(ccd_power, dark_power, ccd_pwr_error, dark_pwr_error):
        """
        function to subtract the dark rate from the optical spectrum and combine the errors
        ccd_power, ccd_pwr_error: the power extracted from extract_data from light data
        dark_power, dark_pwr_error: the power extracted from extract_data from dark data
        """
        minlen = min(len(ccd_power), len(dark_power))
        subtracted_pwr = ccd_power[:minlen] - dark_power[:minlen]
        combined_error = np.zeros(len(ccd_pwr_error))
        for i in range(len(ccd_pwr_error[:minlen])):
                combined_error[i] = np.sqrt((ccd_pwr_error[i])**2 + (dark_pwr_error[i])**2)
        return subtracted_pwr, combined_error


def calculate_qe(ccd_power, pd_power):
        """
        function to generate a relative qe using the pd power to account for systematics
        ccd_power: the power after the dark rate subtraction
        pd_power: the photodiode power
        """
        norm_pd_power = pd_power/max(pd_power)

        qe = np.zeros(len(ccd_power))
        for i, w in enumerate(ccd_power):
                qe[i] = w/norm_pd_power[i]

        return qe
        
        
def plot_qe(folder, label, scale, dark_file):
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
                print(sorted_waves)
                peak = np.argmax(sorted_power)
                df_dark = pd.read_csv(dark_file)
                dark_power = df_dark.iloc[0, :].to_numpy()
                dark_counts = df_dark.iloc[2, :].to_numpy()
                dark_error_cts = df_dark.iloc[3, :].to_numpy()
                dark_error_pwr = df_dark.iloc[4, :].to_numpy()
                minlen = min(len(sorted_power), len(dark_power))
                power_plot = sorted_power[:minlen] - dark_power[:minlen]


                #line = fit_baseline(sorted_waves[sorted_waves > 600], sorted_power[sorted_waves > 600])
                x = np.arange(250, 700, 5)
                #y = line[1]*line[0]**x
                #plt.plot(x, y/max(sorted_power), 'x')
                #print(line)
        plt.plot(sorted_waves, power_plot/max(sorted_power), label = label)

        #if args.avgcts: ax.errorbar(sorted_waves, sorted_counts, yerr = sorted_error_cts, label = label + " CCD counts")
        #if args.ccd:    ax.errorbar(sorted_waves[sorted_power > 1], sorted_power[sorted_power > 1], yerr=sorted_error_pwr[sorted_power > 1], label = label + ", CCD")
        #if args.ccd:    ax.errorbar(sorted_waves[sorted_power > 1], sorted_power[sorted_power > 1]*scale/max(sorted_power[:200]), yerr=sorted_error_pwr[sorted_power > 1]*scale/max(sorted_power), label = label + ", CCD")
        #if args.ccd:    plt.semilogy(sorted_waves[sorted_power > 1], sorted_power[sorted_power > 1], label = label + ", CCD")
        #if args.pd :    plt.semilogy(sorted_waves[sorted_photo_diode != 0], sorted_photo_diode[sorted_photo_diode != 0], label = label + ", PD")

def extract_data_subtract_and_plot(folder, label, scale):
        save_file = os.path.join(folder, out_suffix)
        dark_file = os.path.join(dark, out_suffix)
        if os.path.isfile(save_file):
                print(save_file)
                df = pd.read_csv(save_file)
                df_dark = pd.read_csv(dark_file)
                dark_power = df_dark.iloc[0, :].to_numpy()
                dark_counts = df_dark.iloc[2, :].to_numpy()
                dark_error_cts = df_dark.iloc[3, :].to_numpy()
                dark_error_pwr = df_dark.iloc[4, :].to_numpy()

                
                sorted_power = df.iloc[0, :].to_numpy()
                sorted_power_norm = sorted_power/max(sorted_power)
                sorted_photo_diode = df.iloc[1, :].to_numpy()
                sorted_counts = df.iloc[2, :].to_numpy()
                sorted_error_cts = df.iloc[3, :].to_numpy()
                sorted_error_pwr = df.iloc[4, :].to_numpy()
                waves = df.columns.values
                sorted_waves = np.array(list(map(float, waves)))
                
                minlen = min(len(sorted_power), len(dark_power))
                peak = np.argmax(sorted_power)
                print(sorted_power)
                print(dark_power)
                power_plot = sorted_power[:minlen] - dark_power[:minlen]
                print(power_plot)
                error_plot = np.zeros(len(sorted_error_cts))
                for i in range(len(sorted_error_pwr[:minlen])):
                        error_plot[i] = np.sqrt((sorted_error_pwr[i])**2 + (dark_error_pwr[i])**2)
                print(len(power_plot), len(sorted_waves))
        if args.avgcts: ax.errorbar(sorted_waves, sorted_counts, yerr = sorted_error_cts, label = label + " CCD counts")
        if args.ccd:    ax.errorbar(sorted_waves[sorted_power > 1], power_plot/max(power_plot), yerr=error_plot/max(power_plot), label = label + ", CCD")
        #if args.ccd:    ax.errorbar(sorted_waves, power_plot*scale, yerr = error_plot*scale, label = label + ", CCD")
        #if args.pd :    plt.semilogy(sorted_waves[sorted_photo_diode != 0], sorted_photo_diode[sorted_photo_diode != 0], label = label + ", PD")
        

def generate_qe(folder ):
        save_file = os.path.join(folder, out_suffix)
        pl = pd.read_csv("~/Downloads/lit_pl.csv")
        pl['wavelength'] = pl['wavelength'].astype(int)
        df = pd.read_csv(save_file)
        df_trans = df.T
        df_trans = df_trans.reset_index()
        full_wavelengths = np.arange(int(pl["wavelength"].min()), int(pl["wavelength"].max())+1, 1)
        interpolated_data  = np.interp(full_wavelengths, df_trans["index"], df_trans[2] )
        new_df = pd.DataFrame({"wavelength": full_wavelengths, "power": interpolated_data})
        #new_df.set_index('wavelengths', inplace=True)

        scale = pl["normalization"].max()/new_df["power"].max()
        new_df["power_scaled"] = new_df["power"]*scale
        #new_df.plot(x = "wavelength", y = "power_scaled", style = 'o')
        #pl.plot(x = "wavelength", y = "normalization")
        #plt.show()
        merged = pl.merge(new_df, on = "wavelength")
        qe = pd.read_csv("qe.csv")
        print(qe)
        merged['difference'] = merged["normalization"]/merged["power_scaled"]
        plt.plot(merged['wavelength'],  merged['power_scaled'], label = "lamp data")
        plt.plot(merged['wavelength'],  merged['normalization'], label = "PL from ")
        plt.plot(merged['wavelength'],  merged['difference'], label = "qe?")
        plt.plot(qe.iloc[:,0], qe.iloc[:,1], 'o', label = "Edgar's qe")
        plt.xlim(350, 450)
        plt.ylim(-0.25, 3)
        plt.legend()
        plt.show()

def extract_data_add_and_plot(folder1, folder2, scale):
        save_file = os.path.join(folder1, out_suffix)
        save_file2 = os.path.join(folder2, out_suffix)
        
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
                print(sorted_waves)
                peak = np.argmax(sorted_power)

        if os.path.isfile(save_file2):
                print(save_file)
                df2 = pd.read_csv(save_file2)
                sorted_power2 = df2.iloc[0, :].to_numpy()
                sorted_power_norm2 = sorted_power2/max(sorted_power2)
                sorted_photo_diode2 = df2.iloc[1, :].to_numpy()
                sorted_counts2 = df2.iloc[2, :].to_numpy()
                sorted_error_cts2 = df2.iloc[3, :].to_numpy()
                sorted_error_pwr2 = df2.iloc[4, :].to_numpy()
                waves2 = df2.columns.values
                sorted_waves2 = np.array(list(map(float, waves2)))

        added_power = sorted_power + sorted_power2[1:]
        added_error = np.zeros(len(sorted_error_pwr))
        for i in range(len(sorted_error_pwr)):
                added_error[i] = np.sqrt((sorted_error_pwr[i])**2 + (sorted_error_pwr2[i])**2)
                
        if args.ccd:    ax.errorbar(sorted_waves, added_power*scale/max(added_power), yerr = added_error*scale/max(added_power), label = "convolution")
        
def fit_baseline(wavelengths, power):
        initial_guess = (0.5, 1)
        popt_os, pcov_os = curve_fit(exponential, wavelengths, power)
        return popt_os
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
folders = ["Astroskipper_qe"]

dark = "20251104_dark"
#folders = ["2025_02_07_cube_5sec", "2025_02_06_cube_filter4"]
out_suffix = "save_data.csv"

#labels = ["60 sec", "60 sec cube", "30 sec cube", "10 sec cube", "dark"]
labels = ["10 sec", "5 sec"]
Filter = "Filter 4 "
title_name = "Scintillation of Trans-stilbene with Different Stimulation Sources"
#pl = pd.read_csv("~/Downloads/lit_pl.csv")
ab = pd.read_csv("absorption.csv")


qe = pd.read_csv("qe.csv")
full_wavelengths = np.arange(270, qe.iloc[:, 0].max() + 1, 1)
interpolated_qe  = np.interp(full_wavelengths, qe.iloc[:, 0], qe.iloc[:, 1] )
new_qe = pd.DataFrame({"wavelengths": full_wavelengths, "qe": interpolated_qe})
new_qe.set_index('wavelengths', inplace=True)

fig, ax = plt.subplots()
for i, folder in enumerate(folders):
        print(labels[i])
        ccd_power, pd_power, ccd_counts, ccd_pwr_error, ccd_cts_error, sorted_waves = extract_data(folder)
        #dark_ccd_power, dark_pd_power, dark_ccd_counts, dark_ccd_pwr_error, dark_ccd_cts_error, dark_sorted_waves = extract_data(dark)
        data = np.loadtxt('ABS_QE_Calibration.txt')
        pd_wavelengths = data[:,0]
        pd_power       = data[:,1]
        
        print(*pd_wavelengths)
        print(*pd_power)
        #subtracted_power, combined_error = subtract_dark(ccd_power, dark_ccd_power, ccd_pwr_error, dark_ccd_pwr_error)
        qe = calculate_qe(ccd_power[sorted_waves > 300], pd_power[pd_wavelengths < 450])
        plt.plot(sorted_waves[sorted_waves > 300], qe, label = labels[i] + " qe?")
        if args.avgcts: ax.errorbar(sorted_waves, ccd_counts, ccd_cts_error, label = labels[i] + " CCD counts")
        if args.ccd:    plt.semilogy(sorted_waves[ccd_power > 1], ccd_power[ccd_power > 1], label = labels[i] + ", CCD")
        if args.pd :    plt.semilogy(sorted_waves[pd_power != 0], pd_power[pd_power != 0], label = labels[i] + ", PD")


        #extract_data_and_plot(folder, labels[i], 1)
        #if i == 0: extract_data_subtract_and_plot(folder, labels[i], 1)
        #if i == 1: extract_data_and_plot(folder, labels[i], 1)
        #if i == 2: extract_data_subtract_and_plot(folder, labels[i], 4)
        #plot_qe(folder,labels[i], 1, dark)
plt.yscale("log")
#plt.xlim(200, 400)
plt.xlabel("wavelength")
plt.ylabel("Power")
plt.title("Comparing Dark and lamp data")
plt.plot(new_qe['qe'], label = "Edgar's QE")
plt.legend()
plt.show()
#extract_data_add_and_plot("2025_03_25_cube_60sec", "2025_05_28_led", 1)
#if args.ccd or args.pd: plt.ylabel("Power per area (watts/m^3)")
if args.ccd or args.pd: plt.ylabel("Relative units", fontsize = 16)
#if args.ccd or args.pd: plt.ylabel("rel units")
if args.avgcts: plt.ylabel("Average Counts (ADU)")
#ax.plot(pl['wavelength'], pl['normalization'], 'o', label = "PL from Lyasnikova et al.")
#ax.plot(ab['wavelength'], ab['normalization'], 'o', label = "Absorption from Literature")
#ax.set_yscale("log")
plt.xlabel("Wavelength (nm)", fontsize = 16)
plt.title(title_name, fontsize = 20)
plt.xlim(250,550)
plt.ylim(-0.25, 1.5)
plt.legend()
plt.show()

#if we have already run the processing code on this folder just load in the saved .csv and skip to plotting it

sys.exit()
