from group1_features import *
from group2_features import *
from group3_glcm import *
from group3_glrl import *
from dicom_module import *

import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glrl_loop import _glrl_vector_loop

import scipy.misc
import matplotlib.pyplot as plt
import pywt
import pickle
from profiling_tools import time

import csv

def loadImageAndSurface():

    try:

        with open('masked_volume.pickle', mode = 'rb') as f:

            masked_volume = pickle.load(f)

        struct_dcm = dicom.read_file('./data/struct.dcm')

        #for i, structure in enumerate(struct_dcm.StructureSetROISequence):

        #    name = structure.ROIName

        #    print i, name

        #selected = int(raw_input("Select the structure for mesh creation: "))

        structureLoader = StructureLoader(struct_dcm, 0)

        polydata = structureLoader.surface

    except:

        struct_dcm = dicom.read_file('./data/struct.dcm')

        #for i, structure in enumerate(struct_dcm.StructureSetROISequence):

        #    name = structure.ROIName

        #    print i, name

        #selected = int(raw_input("Select the structure for mesh creation: "))

        structureLoader = StructureLoader(struct_dcm, 0)

        polydata = structureLoader.surface

        imageLoader = ImageLoader('./data/fusedMRI')

        masked_volume = makeMaskedVolume(imageLoader, structureLoader)

        with open('masked_volume.pickle', mode = 'wb') as f:

            pickle.dump(masked_volume, f)

    return masked_volume, polydata

def wavelet_transform(image):

    return pywt.dwtn(image, 'coif1')

def saveDictionary(dict, filepath):

    with open(filepath, 'ab') as f:

        csv_writer = csv.writer(f, delimiter = ',')

        for key in dict.keys():

            _data = []
            _data.append(key)
            _data.append(dict[key])
            csv_writer.writerow(_data)

        f.close()

def main():

    image, polydata = loadImageAndSurface()

    bin_width = 20

    # 0 -> 0, 1 ~ bin -> 1, bin+1 ~ 2bin -> 2

    scaled_image = np.copy(image)
    scaled_image += bin_width - 1
    scaled_image /= bin_width

    dict = {}

    dict.update(group1_features(image))

    dict.update(group2_features(polydata))

    dict.update(group3_glcm_features(scaled_image, 1))

    dict.update(group3_glrl_features(scaled_image))

    saveDictionary(dict, 'radiomics.csv')

    decomposed_images = wavelet_transform(image)

    for key in decomposed_images.keys():

        decomposed_dict = {}

        decomposed_image = (decomposed_images[key])
        decomposed_image -= np.min(decomposed_image)   

        decomposed_image = decomposed_image.astype(np.uint16)

        decomposed_dict.update(group1_features(decomposed_image))

        scaled_decomposed_image = np.copy(decomposed_image)
        scaled_decomposed_image += bin_width - 1
        scaled_decomposed_image /= bin_width

        decomposed_dict.update(group3_glcm_features(scaled_decomposed_image, 1))
        decomposed_dict.update(group3_glrl_features(scaled_decomposed_image))

        for _key in decomposed_dict.keys():

            val = decomposed_dict[_key]

            decomposed_dict[key + '_' + _key] = val

            del decomposed_dict[_key]

        saveDictionary(decomposed_dict, 'radiomics.csv')  

if __name__ == '__main__':

    main()