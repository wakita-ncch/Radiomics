import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glrl_loop import _glrl_vector_loop

from profiling_tools import time

#def glrl_vector_loop(image, direction, bin_width):

#    # convert pixel intensities into gray levels wi

#    bin_width = int(bin_width)

#    image /= bin_width

#    return _glrl_vector_loop(image, direction)

def group3_glrl_features(image, bin_width):

    SRE = []
    LRE = []
    GLN = []
    RLN = []
    RP = []
    LGLRE = []
    HGLRE = []
    SRLGLE = []
    SRHGLE = []
    LRLGLE = []
    LRHGLE = []

    group3_features = {}

    for i in range(1, 13):

        glrl_matrix = GLRL_Matrix(image, i, bin_width)

        SRE.append(glrl_matrix.short_run_emphasis())
        LRE.append(glrl_matrix.long_run_emphasis())
        GLN.append(glrl_matrix.gray_level_non_uniformity())
        RLN.append(glrl_matrix.run_length_non_uniformity())
        RP.append(glrl_matrix.run_percentage())
        LGLRE.append(glrl_matrix.low_gray_level_run_emphasis())
        HGLRE.append(glrl_matrix.high_gray_level_rum_emphasis())
        SRLGLE.append(glrl_matrix.short_run_low_gray_level_emphasis())
        SRHGLE.append(glrl_matrix.short_run_high_gray_level_emphasis())
        LRLGLE.append(glrl_matrix.long_run_low_gray_level_emphasis())
        LRHGLE.append(glrl_matrix.long_run_high_gray_level_emphasis())

    group3_features['SRE'] = np.mean(SRE)
    group3_features['LRE'] = np.mean(LRE)
    group3_features['GLN'] = np.mean(GLN)
    group3_features['RLN'] = np.mean(RLN)
    group3_features['RP'] = np.mean(RP)
    group3_features['LGLRE'] = np.mean(LGLRE)
    group3_features['HGLRE'] = np.mean(HGLRE)
    group3_features['SRLGLE'] = np.mean(SRLGLE)
    group3_features['SRHGLE'] = np.mean(SRHGLE)
    group3_features['LRLGLE'] = np.mean(LRLGLE)
    group3_features['LRHGLE'] = np.mean(LRHGLE)

    print group3_features

    return group3_features

class GLRL_Matrix:

    def __init__(self, image, direction, bin_width):

        bin_width = int(bin_width)

        image += (bin_width - 1) 
        image /= bin_width # 0 -> 0, 1 ~ bin -> 1, bin+1 ~ 2bin -> 2

        self.image = image

        self.glrl_matrix = _glrl_vector_loop(image, direction).astype(np.float)

        self.Ng = np.arange(1, self.glrl_matrix.shape[0] + 1).astype(np.float)
        self.Nr = np.arange(1, self.glrl_matrix.shape[1] + 1).astype(np.float)
        self.Np = float(len(self.image.ravel()))

        self.jj, self.ii = np.meshgrid(self.Nr, self.Ng)
        self.jj = self.jj.astype(np.float)
        self.ii = self.ii.astype(np.float)
        
        self.sum_matrix = float(np.sum(self.glrl_matrix))

        #print "Image: ", self.image
        #print "GLRL matrix: ", self.glrl_matrix

        #print "Ng: ", self.Ng
        #print "Nr: ", self.Nr
        #print "Np: ", self.Np

        #print "sum_matrix: ", self.sum_matrix

        #print "ii: ", self.ii
        #print "jj: ", self.jj

        #print "SRE: ", self.short_run_emphasis()
        #print "LRE: ", self.long_run_emphasis()
        #print "GLN: ", self.gray_level_non_uniformity()
        #print "RLN: ", self.run_length_non_uniformity()
        #print "RP: ", self.run_percentage()
        #print "LGLRE: ", self.low_gray_level_run_emphasis()
        #print "HGLRE: ", self.high_gray_level_rum_emphasis()
        #print "SRLGLE: ", self.short_run_low_gray_level_emphasis()
        #print "SRHGLE: ", self.short_run_high_gray_level_emphasis()
        #print "LRLGLE: ", self.long_run_low_gray_level_emphasis()
        #print "LRHGLE: ", self.long_run_high_gray_level_emphasis()

    def short_run_emphasis(self):

        return np.sum(self.glrl_matrix / self.jj**2.0) / self.sum_matrix

    def long_run_emphasis(self):

        return np.sum(self.jj ** 2 * self.glrl_matrix) / self.sum_matrix

    def gray_level_non_uniformity(self):

        return np.sum(np.sum(self.glrl_matrix, axis=1)**2) / self.sum_matrix

    def run_length_non_uniformity(self):

        return np.sum(np.sum(self.glrl_matrix, axis=0)**2) / self.sum_matrix

    def run_percentage(self):

        return np.sum(self.glrl_matrix) / self.Np

    def low_gray_level_run_emphasis(self):

        return np.sum(self.glrl_matrix / self.ii**2) / self.sum_matrix

    def high_gray_level_rum_emphasis(self):

        return np.sum(self.ii**2 * self.glrl_matrix) / self.sum_matrix

    def short_run_low_gray_level_emphasis(self):

        return np.sum(self.glrl_matrix / (self.ii**2 * self.jj**2)) / self.sum_matrix

    def short_run_high_gray_level_emphasis(self):

        return np.sum(self.ii**2 * self.glrl_matrix / self.jj**2) / self.sum_matrix

    def long_run_low_gray_level_emphasis(self):

        return np.sum(self.jj**2 * self.glrl_matrix / self.ii**2) / self.sum_matrix

    def long_run_high_gray_level_emphasis(self):

        return np.sum(self.ii**2 * self.jj**2 * self.glrl_matrix) / self.sum_matrix