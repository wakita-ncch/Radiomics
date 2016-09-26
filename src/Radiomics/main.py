from group1_features import *
from group2_features import *
from group3_glcm import *

import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glrl_loop import _glrl_vector_loop

#import scipy.misc
#import matplotlib.pyplot as plt

from profiling_tools import time

@time
def glrl_vector_loop(image, direction, bin_width):

    # convert pixel intensities into gray levels wi

    bin_width = int(bin_width)

    image /= bin_width

    return _glrl_vector_loop(image, direction)

def main():

    image = np.array([[[5, 2, 5, 4, 4], 
                       [3, 3, 3, 1, 3],
                       [2, 1, 1, 1, 3],
                       [4, 2, 2, 2, 3],
                       [3, 5, 3, 3, 2]]])

    #image = np.arange(3 * 3 * 3).reshape((3, 3, 3))

    print glrl_vector_loop(image, 1, 1)

    #volume = np.arange(5 * 7 * 8).reshape((5, 7, 8))

    #vector = [0, 1, 1]

    #glcm = GLCM_Matrix(volume, 1.0, vector, 10)

    ##print "matrix: ", glcm.glcm_matrix
    #print "sum: ", np.sum(glcm.glcm_matrix)
    #print "levels: ", np.arange(1, glcm.glcm_matrix.shape[0]+1)
    #print "ux: ", np.sum(np.sum(glcm.glcm_matrix, axis=1))
    #print "uy: ", np.sum(np.sum(glcm.glcm_matrix, axis=0))

    #print "autocorrelation: ", glcm.autocorrelation()
    #print "cluster prominence: ", glcm.cluster_prominence()
    #print "cluster shade: ", glcm.cluster_shade()
    #print "cluster tendency: ", glcm.cluster_tendency()
    #print "contrast: ", glcm.contrast()
    ##print "difference entropy: ", glcm.difference_entropy()
    #print "dissimilarity: ", glcm.dissimilarity()
    #print "energy: ", glcm.energy()
    ##print "entropy: ", glcm.entropy()
    #print "homogeneity1: ", glcm.homogeneity1()
    #print "homogeneity2: ", glcm.homogeneity2()
    ##print "IMC1: ", glcm.IMC1()
    ##print "IMC2: ", glcm.IMC2()
    ##print "IDMN: ", glcm.IDMN()
    ##print "IDN: ", glcm.IDN()
    ##print "inverse variance: ", glcm.inverse_variance()

if __name__ == '__main__':

    main()