import spectral.io.envi as envi
from spectral import imshow
from numba import jit, prange
import numpy as np
import cv2
import sys
import argparse
import time

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', nargs = '+', type = str)
    parser.add_argument('-b', '--bands', nargs = '+', type = int, default = [])
    parser.add_argument('-c', '--clustering', type = int, default = 0)
    parser.add_argument('-t', '--threshold', type = float, default = 0.3)
    parser.add_argument('-s', '--scale', type = int, default = 0)
    parser.add_argument('-r', '--rayleigh', type = int, default = 1)
    return parser

def get_rayleigh_scattering(hsi):
    hsi_bands = hsi.shape[2]
    rayleigh_sig = np.zeros(shape = hsi_bands)
    for i in range(hsi_bands):
        tmp = hsi[:, :, i].flatten()
        tmp = tmp[tmp > 0]
        rayleigh_sig[i] = tmp.min()
    return np.int16(rayleigh_sig)

@jit(nopython = True, cache = True)
def dev_std(reference, pixel):
    return np.sqrt( np.sum((reference - pixel) ** 2) / len(pixel) )

@jit(nopython = True, parallel = True, cache = True)
def count_compare_dev_std(reference_pix, pix_image):
    compare = np.zeros(shape = len(reference_pix))
    for r in prange(len(reference_pix)):
        compare[r] = dev_std(reference_pix[r], pix_image)
    return compare

def classification_std_norm_linalg(hsi, threshold, reference = []):
    num_pix = 100 / (hsi.shape[0] * hsi.shape[1])
    segmentation_mask = np.zeros(shape = (hsi.shape[0], hsi.shape[1]), dtype = np.int16)
    segmentation_mask = np.full_like(segmentation_mask, -1)
    compare_mask = np.zeros(shape = (hsi.shape[0], hsi.shape[1]))    
    
    if len(reference) == 0:
        reference = np.empty((0, hsi.shape[2]))
        true_pix = False
        for i in range(hsi.shape[0]):
            if true_pix == True: break
            for j in range(hsi.shape[1]):
                if np.unique(hsi[i, j]).shape[0] == 1:
                    continue
                else:
                    reference = np.append(reference, [hsi[i, j]], axis = 0)
                    true_pix = True
                    break
                
    mean_reference = np.mean(reference, axis = 1)
    
    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            if np.unique(hsi[i, j]).shape[0] == 1:
                continue
            
            compare = count_compare_dev_std(np.int64(reference), np.int64(hsi[i, j]))
            #print("#########################################################")
            
            if any((compare / mean_reference) < threshold) == True:
                bool_mask = np.invert(np.array((compare / mean_reference) < threshold))
                arr = np.ma.array(compare, mask = bool_mask)
                min_arg_dev =  arr.argmin()
                
                segmentation_mask[i, j] = min_arg_dev
                compare_mask[i, j] = compare[min_arg_dev]
                
            else:
                reference = np.append(reference, [hsi[i, j]], axis = 0)
                segmentation_mask[i, j] = reference.shape[0] - 1
                compare_mask[i, j] = 0
                #print(hsi[i, j], i, j)
                
                mean_reference = np.mean(reference, axis = 1)
            
            #print("#########################################################")
            
            
        #print('\r', end = '')
        #print(str(i) + " num ref: " + str(reference.shape[0]), end = '')
        process = num_pix * (i * hsi.shape[1] + j)
        print('\r', end = '')
        print("#######  " + str("%.2f" % process) + "%  #######", end = '')
        
    return segmentation_mask, compare_mask, np.uint16(reference)

def get_scaling(hsi, reference_pix, segmentation_mask):
    num_pix = 100 / (hsi.shape[0] * hsi.shape[1])
    len_reference = np.zeros(shape = reference_pix.shape[0], dtype = np.uint16)
    for i in range(reference_pix.shape[0]):
        len_reference[i] = np.linalg.norm(reference_pix[i])
        #len_reference[i] = np.mean(reference_pix[i])
    
    height, width, _ = hsi.shape
    scaling_parameters = np.zeros(shape = (height, width), dtype = np.float64)
    
    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            
            n_ref = int(segmentation_mask[i, j])
            
            if n_ref != -1:
                scaling = len_reference[n_ref] / np.linalg.norm(hsi[i, j])
                #scaling = len_reference[n_ref] / np.mean(hsi[i, j])
                scaling_parameters[i, j] = scaling 
                
            else:
                scaling_parameters[i, j] = 0
        
        process = num_pix * (i * hsi.shape[1] + j)
        print('\r', end = '')
        print("#######  " + str("%.2f" % process) + "%  #######", end = '')
        
    return scaling_parameters

if __name__ == "__main__":
    parser = arg_parser()
    namespace = parser.parse_args(sys.argv[1:])
    
    if len(namespace.path) == 2:
        print("*******\nCompression process started")
        bin_path = namespace.path[0]
        hdr_path = namespace.path[1]

        open_hsi = envi.open(hdr_path, bin_path)
        hsi = np.rot90(np.array(open_hsi.open_memmap()))
        print("*******\nHSI opened", hsi.shape)


        if namespace.rayleigh == 1:
            rayleigh_sig = get_rayleigh_scattering(hsi)
            hsi = hsi - rayleigh_sig
            print("*******\nRayleigh scattering count completed")

        if len(namespace.bands) == 2:
            if namespace.bands[0] < namespace.bands[1]:
                hsi = hsi[:, :, namespace.bands[0]:namespace.bands[1]]
                print("*******\nNew HSI shape", hsi.shape)
            else:
                print("*******\nError: slice bands")
                exit()

        hsi[hsi < 0] = 0
        hsi = np.uint16(hsi)

        print("*******\nHSI compression, stage 1 / 2")
        start = time.time()
        segmentation_mask, compare_mask, reference = classification_std_norm_linalg(hsi, namespace.threshold)
        end = time.time()
        print("\nStage 1 completed, time stage: ", int(end - start), " seconds")

        print("*******\nHSI compression, stage 2 / 2")
        print("Enter name compressed file")
        c_name = input()

        if namespace.scale == 0:
            if namespace.rayleigh == 1:
                np.savez(c_name, (0, 1), segmentation_mask, reference, rayleigh_sig)
            else:
                np.savez(c_name, (0, 0), segmentation_mask, reference)
            print("\nStage 2 completed")


        elif namespace.scale == 1:
            start = time.time()
            scaling_parameters = get_scaling(hsi, reference, segmentation_mask)
            if namespace.rayleigh == 1:
                np.savez(c_name, (1, 1), segmentation_mask, reference, scaling_parameters, rayleigh_sig)
            else:
                np.savez(c_name, (1, 0), segmentation_mask, reference, scaling_parameters)
            end = time.time()
            print("\nStage 2 completed, time stage: ", int(end - start), " seconds")

        print("*******\nFile seved")


    elif len(namespace.path) == 1:
        print("*******\nDecompression process started")
        compression_hsi = np.load(namespace.path[0])
        method = compression_hsi['arr_0']
        segmentation_mask = compression_hsi['arr_1']
        reference = compression_hsi['arr_2']
        hsi = np.zeros(shape = (segmentation_mask.shape[0], segmentation_mask.shape[1], reference.shape[1]), dtype = np.uint16)

        if method[0] == 0 and method[1] == 1:
            for i in range(hsi.shape[0]):
                for j in range(hsi.shape[1]):
                    hsi[i, j] = reference[int(segmentation_mask[i, j])]
        
        if method[0] == 0 and method[1] == 0:    
            rayleigh_sig = compression_hsi['arr_3']
            for i in range(hsi.shape[0]):
                for j in range(hsi.shape[1]):
                    hsi[i, j] = reference[int(segmentation_mask[i, j])] + rayleigh_sig[:reference.shape[1]]

        if method[0] == 1 and method[1] == 1:
            scaling_parameters = compression_hsi['arr_3']
            rayleigh_sig = compression_hsi['arr_4']
            for i in range(hsi.shape[0]):
                for j in range(hsi.shape[1]):
                    hsi[i, j] = np.uint16(reference[int(segmentation_mask[i, j])] * scaling_parameters[i, j]) + rayleigh_sig[:reference.shape[1]]

        if method[0] == 1 and method[1] == 0:
            scaling_parameters = compression_hsi['arr_3']
            for i in range(hsi.shape[0]):
                for j in range(hsi.shape[1]):
                    hsi[i, j] = reference[int(segmentation_mask[i, j])] * scaling_parameters[i, j]

        print("*******\nDecompression process finished")
        print("\nEnter name decompressed file")
        c_name = input()
        np.savez(c_name, hsi)
        print("*******\nFile seved")

    else:
        print("*******\nError: Path")
        exit()
