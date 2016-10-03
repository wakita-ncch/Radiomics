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

def main():

    image = np.array([[[0,0,4,5,7],
                      [1,2,4,6,0],
                      [3,4,6,0,1],
                      [7,4,0,1,4]]]).astype(np.uint16)

    print "initial image: ", image

    GLCM_Matrix(image, 1, [0,1,0],1)

    #group1_features(masked_volume)

    #group2_features(polydata)

    #GLCM_Matrix(masked_volume, 1, [0,1,1], 25)

    #group3_glrl_features(masked_volume, 25)

    #decomposed_image = wavelet_transform(masked_volume)

    #for key in decomposed_image.keys():

    #    image = decomposed_image[key]

if __name__ == '__main__':

    main()