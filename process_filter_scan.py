import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import os
import sys
import pandas as pd
import csv
import argparse

#needed constants
h = 6.626e-34 #plank's constant
c = 3e8 #speed of light
scale = (0.01)**2  #size of the photodiode

#A function to extract the power from the CCD images looking at all four amplifiers
def get_power(file_path):
        pixels = 4*col*row*(1.5e-5)**2 #calculate the total area of the CCD
        this_qe = new_qe.loc[float(wavelength)] # Get the quantum efficiency for this wavelength
        energy = h*c/(float(wavelength)*1e-9) #calculate the energy of the photons for this wavelength

        #cycle through all four amplifiers and record the number of counters excluding the cosmics and the overscan
        channel_power = np.zeros(4)
        error = np.zeros(4)
        for n in range(4):
                hdul = fits.getdata(file_path, n)
                filter_data = sigma_clip(hdul[:3072])
                channel_power[n] = np.sum(hdul[:3072][~filter_data.mask])
                avg_counts = np.mean(hdul[:3072][~filter_data.mask])/this_qe.iloc[0]
                error[n] = np.std(hdul[:3072][~filter_data.mask])/this_qe.iloc[0]
                print("error: ", error)
                print("avg cts: ", avg_counts)
        #sum all four amplifiers and scale appropriately power = (counts*energy)/(gain*time*area)
        ccd_power =sum(channel_power)*energy*1e12/140/time/pixels
        error_cts = np.mean(error)
        error_power = np.mean(error)*energy*1e12/140/time/np.sqrt(4*col*row)/(1.5e-5)**2
        print(pixels, ccd_power, error_power)
        #ccd_power = channel_power[0]/this_qe.iloc[0]
        return ccd_power, avg_counts, error_cts, error_power

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

#get the folder we are processing in this run of the code
folder = args.filename
override = args.override
print(override)
file_num = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]) #num of files in that folde
out_suffix = "save_data.csv"
save_file = os.path.join(folder, out_suffix)
print(save_file)

#if we have already run the processing code on this folder just load in the saved .csv and skip to plotting it
if os.path.isfile(save_file) and override==False:
        df = pd.read_csv(save_file)
        sorted_power = df.iloc[0, :].to_numpy()
        sorted_photo_diode = df.iloc[1, :].to_numpy()
        waves = df.columns.values
        sorted_waves = np.array(list(map(float, waves)))


#otherwise process the folder
else: 
        #create arrays to store the power and wavelengths
        waves = np.zeros(file_num) #wavelengths
        photo_diode = np.zeros(file_num) #power recorded by the photo diode
        power_qe = np.zeros(file_num) #count from the median fits value
        avg_counts = np.zeros(file_num)
        error_cts = np.zeros(file_num)
        error_pwr = np.zeros(file_num)

        #for each file in the folder calculate the power 
        for i, file in enumerate(os.listdir(folder)):
                file_path = os.path.join(folder, file)
                print(file_path)
                if os.path.isfile(file_path) and file.endswith('.fits'):
                        #get the info from the primary header
                        hdul, header = fits.getdata(file_path,header = True)
                        col = int(header['CCDNCOL'])/2
                        row = int(header['NROW'])
                        if row > 500:
                                row = 500
                        wavelength  = header['WAVE']
                        time = header['EXPTIME']
                        photo_diode[i] = header['POWER']*1e12/scale
                        waves[i] = wavelength

                        #calculate the CCD power
                        power_qe[i], avg_counts[i], error_cts[i], error_pwr[i]  = get_power(file_path)

        #sort all of the data by the wavelengths for graphing
        sort = np.argsort(waves)
        sorted_waves = waves[sort]
        sorted_power = power_qe[sort]
        sorted_photo_diode = photo_diode[sort]
        sorted_counts = avg_counts[sort]
        sorted_error_pwr = error_pwr[sort]
        sorted_error_cts = error_cts[sort]
        
        with open(save_file, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow(sorted_waves)
                writer.writerow(sorted_power)
                writer.writerow(sorted_photo_diode)
                writer.writerow(sorted_counts)
                writer.writerow(sorted_error_cts)
                writer.writerow(sorted_error_pwr)
                
#plot the data and print out the power ratio between the photodiode and the CCD
#print(max(photo_diode)/max(power_qe))   
#plt.plot(sorted_waves, sorted_power, label = "CCD")
plt.semilogy(sorted_waves[sorted_power >0], sorted_power[sorted_power >0], label = "CCD")
plt.semilogy(sorted_waves[sorted_photo_diode != 0], sorted_photo_diode[sorted_photo_diode != 0], label = "PD")
plt.legend()
#plt.ylim(0, 100000)
#plt.xlim(250, 400)
plt.title(folder)
plt.xlabel("wavelength (nm)")
plt.ylabel("Power (picoWatts)")
plt.show()







sys.exit()
try: print(sys.argv[2])
except: print("only one dataset provided")
else:

        folder = sys.argv[2]
        file_num = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        power = np.zeros(file_num) 
        waves = np.zeros(file_num)#wavelengths
        photo_diode = np.zeros(file_num) #power recorded by the photo diode
        power_qe = np.zeros(file_num) #count from the median fits value

        for i, file in enumerate(os.listdir(folder)):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path) and file.endswith('.fits'):
                        hdul, header = fits.getdata(file_path,header = True)
                        col = int(header['NCOl'])
                        row = int(header['NROW'])
                        wavelength  = header['WAVE']
                        time = header['EXPTIME']

                        waves[i] = wavelength
                        power_qe[i] = get_power(file_path)
        sort = np.argsort(waves)
        sorted_waves = waves[sort]
        sorted_filter = power_qe[sort]
        sorted_photo = photo_diode[sort]
        sorted_photo_diode = photo_diode[sort]
        plt.semilogy(sorted_waves, sorted_filter, label = "No cube CCD")

        plt.semilogy(sorted_waves[sorted_photo_diode != 0], sorted_photo_diode[sorted_photo_diode != 0], label = "No cube PD")
#plt.axhline(2e-3)
#plt.plot(sorted_waves[sorted_photo_diode != 0], sorted_photo_diode[sorted_photo_diode != 0]/sorted_power_qe[sorted_photo_diode !=0])
#plt.plot(sorted_waves, sorted_power_qe/max(sorted_power_qe) - sorted_filter[:54]/max(sorted_filter[:54]), label = "difference")
plt.show()
