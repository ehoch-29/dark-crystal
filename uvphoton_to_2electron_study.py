#needed constants
h = 6.626e-34 #plank's constant
c = 3e8 #speed of light
scale = (0.01)**2  #size of the photodiode
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import argparse
from astropy.io import fits
from scipy.integrate import quad

import Obsidian_lib as OB

if __name__ == "__main__":        
        #Grab the inputs from the command line
        parser = argparse.ArgumentParser('Parse text in the file')
        parser.add_argument('filename', help = 'folder you want to process')
        parser.add_argument( '-o', '--override', action = 'store_true', help = 'include -o flag to rewrite the save data?')
        
        args = parser.parse_args()
        """
        #input the qe data and make a dataset that has a value for every 5 nm using interpolation
        qe = pd.read_csv("qe.csv")
        full_wavelengths = np.arange(200, qe.iloc[:, 0].max() + 1, 1)
        interpolated_qe  = np.interp(full_wavelengths, qe.iloc[:, 0], qe.iloc[:, 1] )
        new_qe = pd.DataFrame({"wavelengths": full_wavelengths, "qe": interpolated_qe})
        new_qe.set_index('wavelengths', inplace=True)
        """
        #get the folder we are processing from the command line and count the number of files in it
        folder = args.filename
        file_num = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        data = []

        num_samples = []
        #make a canvas to plot the whole count histogram onto 
        hdus = [0, 1, 2,3]
        #loop through each fits file in the folder
        for i, file in enumerate(os.listdir(folder)):
                file_path = os.path.join(folder, file)
                print("file being processed: ", file)
                fig, axes = plt.subplots(2, 2, figsize =(8,8))
                axes = axes.flatten()

                #check to make sure that we are only processing fits files
                if os.path.isfile(file_path) and file.endswith('.fits'):
                        hdul_dummy, header = fits.getdata(file_path, header = True) #read the header from the file
                        samples = float(header['NSAMP']) #get the number of samples from the header
                        print("number of samples: ", samples)
                        num_samples.append(samples)
                        f0s = []
                        #fig.suptitle(samples)
                        #print(exposure)
                        time = samples*3.86 + 5.21 #time for the readout (calculated somewhat manually)
                        
                        #set parameters for masking
                        edge_mask = 5 #number of rows to exclude along the edge

                #loop through each of the 4 amplifiers in each image
                for n in hdus:
                        #define which subplot we are working in
                        ax = axes[n]

                        print("HDU: ", n)
                        #fig2, axes2 = plt.subplots(1, 1, figsize = (50, 10))
                        #take the data from the fits file and turn it into a list of pixel values
                        hdul = fits.getdata(file_path, n)

                        #create a mask for the cosmic rays 
                        cosmic_indices = np.where(hdul > 100*140) #find the pixels with values greater than 100 e-

                        row, col = hdul.shape
                        halo_mask = np.full((row, col), True, dtype=bool) #create a mask of the same dimensions of hdul

                        #for each of the hot pixels mask the surrounding area
                        for i in range(len(cosmic_indices[0])):
                                cosmic_index = (int(cosmic_indices[0][i]), int(cosmic_indices[1][i]))
                                OB.cosmic_masks(hdul, cosmic_index, halo_mask)

                        #make a masked array using the halo mask
                        masked_hdul = np.where(halo_mask, hdul, np.nan)
                        #plot_fits(masked_hdul, file_path + str(n), axes2, fig2)
                        #count the number of unmasked pixels
                        unmasked_pixel = np.count_nonzero(halo_mask)
                        print(f"Number of unmasked pixels: {unmasked_pixel}")
                        
                        #flatten the 2D array into a list to make into a histogram
                        count_list = masked_hdul.flatten().tolist()
                        bins = np.arange(np.nanmin(count_list),np.nanmax(count_list), 20)

                        
                        counts, bin_edges = np.histogram(count_list, bins=bins, density=False)
                        count_density = counts/20
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                        #fit the data with a multi_gaussian to get skipper parameters
                        heights, centers, widths, peaks  = OB.fit_multi_gaussian(count_density, bin_centers)
                        OB.check_gauss_fit(heights, centers, widths[0], bin_centers, ax, count_list, peaks, counts, file_path)
                        print("checking gaussian")

                        #integrate the zero peak gaussian to get f0
                        area1, err = quad(lambda x: OB.gaussian(x, heights[0], centers[0], widths[0]),
                                         (centers[0]-3*widths[0]), (centers[0]+3*widths[0]))
                        area = heights[0]*widths[0]*np.sqrt(2*np.pi)
                        print("area under zero peak: ", area, area1)
                        f0 = area/unmasked_pixel
                        f0s.append(float(f0))
                        print("f0: ", f0)
                        
                        #calculate the average gain per pixel
                        differences = []
                        for i in range(len(centers)-1):
                                differences.append(centers[i+1]-centers[i])
                        try: gain = sum(differences)/len(differences)
                        except: gain = 2
                        print("average gain is: ", gain)
                        #get_counts(file_path)
                        #one_e_bound_min, one_e_bound_max = one_e_parameters(heights[1], centers[1], widths[0], time)
                        
                        mask_list = masked_hdul.flatten().tolist()
                        bins = np.arange(-100, 1000, 20)
                        #ax.hist(mask_list, bins = bins)
                plt.show()                        
                """
                        start = (masked_hdul < one_e_bound_max) & (masked_hdul > one_e_bound_min)
                        indices = np.where(start == True)
                        neighbors = []
                        one_e_count = 0
                        for i in range(len(indices[0])):
                                index = (int(indices[0][i]), int(indices[1][i]))
                                event = check_neighbors(masked_hdul, index, neighbors, one_e_bound_min)
                                if event == True:
                                        one_e_count += 1
                        print('# of no-filtered single electron events: ', len(indices[0]))
                        print("# of single electron event: ", one_e_count)
                        sample = MultiSampleData(samples, gain, widths[0], one_e_bound_min, exposure, n, unmasked_pixel, len(indices[0]), one_e_count)
                        data.append(sample)
                """
                monivars = [{"ANSAMP":samples, "f0":f0s, "EXP": 60}]
                #OB.update_evolFile(file, monivars, "monitoring_DB.tsv")

"""
        #fit data with line
        slope, intercept = np.polyfit(num_samples, f0s, 1)
        x = np.arange(0, 420, 20)
        y_fitted = slope*x+intercept
        plt.plot(x, y_fitted, '-')
        plt.plot(num_samples, f0s, 'o')
        plt.xlabel("# of Samples")
        plt.ylabel("fraction of pixels with 0 e-")
        plt.title("Fraction of 0 e- pixels versus exposure time")
        plt.show()
"""
        

