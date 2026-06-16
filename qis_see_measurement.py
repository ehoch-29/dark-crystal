import argparse
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

import Obsidian_lib as OB

if __name__ == "__main__":
    #Grab the inputs from the command line                                                                                                                               
    parser = argparse.ArgumentParser('Parse text in the file')
    parser.add_argument('filename', help = 'folder you want to process')
    parser.add_argument( '-o', '--override', action = 'store_true', help = 'include -o flag to rewrite the save data?')

    args = parser.parse_args()

    #get the folder we are processing from the command line and count the number of files in it                                                                          
    folder = args.filename
    file_num = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    data = []
    num_samples = []
    fig, axes = plt.subplots(4, 4, figsize =(8,8))
    axes = axes.flatten()
    hdus = [0, 1, 2, 3]
    for i, file in enumerate(os.listdir(folder)):
        file_path = os.path.join(folder, file)
        print("file being processed: ", file)

        #check to make sure that we are only processing fits files                                                                                                   
        if os.path.isfile(file_path) and file.endswith('.fits'):
            hdul_dummy, header = fits.getdata(file_path, header = True) #read the header from the file                                                           
            samples = float(header['NSAMP']) #get the number of samples from the header
            overscan_start = int(header['CCDNCOL'])+10
            num_rows = int(header['NROW'])
            print(overscan_start)
            print("number of samples: ", samples)
            num_samples.append(samples)
            f0s = []
            fig.suptitle("Zero electron peak drift in the overscan")

            time = samples*3.86 + 5.21 #time for the readout (calculated somewhat manually)                                                                      
            
            #set parameters for masking                                                                                                                          
            edge_mask = 5 #number of rows to exclude along the edge                                                                                              
            for n in hdus:
                #define which subplot we are working in
                zero_peak = []
                print(n)
                ax = axes[4*i+n]
                hdul = fits.getdata(file_path, n)
                overscan_hdul = hdul[:,:overscan_start]
                #flatten the 2D array into a list to make into a histogram
                for r in range(num_rows):
                    hdul_slice = hdul[r:r+1, :overscan_start]
                    slice_list = hdul_slice.flatten().tolist()
                    bins = np.arange(np.nanmin(slice_list), np.nanmax(slice_list), 20)
                    slice_counts, bin_edges = np.histogram(slice_list, bins=bins, density=False)
                    max_bin = bin_edges[np.argmax(slice_counts)]
                    print(r, "max counts: ", max_bin)
                    zero_peak.append(max_bin)
                ax.plot(np.arange(num_rows), zero_peak)
                count_list = overscan_hdul.flatten().tolist()
                bins = np.arange(np.nanmin(count_list), np.nanmin(count_list)+2000, 10)
                print("min: ", np.nanmin(count_list), " max: ", np.nanmax(count_list))
                counts, bin_edges = np.histogram(count_list, bins=bins, density=False)
                print("mean: ", np.mean(counts))
                count_density = counts/20
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                #heights, centers, widths, peaks = OB.fit_multi_gaussian(counts, bin_centers)
                #print("heights", heights, "peaks: ", centers)
                #ax.plot(centers, heights, 'x')
                #ax.hist(count_list, bins=bins, density=False, histtype = 'step')

    plt.show()

