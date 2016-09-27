import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glrl_loop import _glrl_vector_loop

from profiling_tools import time

def glrl_vector_loop(image, direction, bin_width):

    # convert pixel intensities into gray levels wi

    bin_width = int(bin_width)

    image /= bin_width

    return _glrl_vector_loop(image, direction)

class GLRL_Matrix:

    def __init__(self, image, direction, bin_width):

        bin_width = int(bin_width)

        image /= bin_width

        self.image = image

        self.glrl_matrix = _glrl_vector_loop(image, direction).astype(np.float)

        self.Ng = np.arange(1, self.glrl_matrix.shape[0] + 1).astype(np.float)
        self.Nr = np.arange(1, self.glrl_matrix.shape[1] + 1).astype(np.float)
        self.Np = float(len(self.image.ravel()))

        self.jj, self.ii = np.meshgrid(self.Nr, self.Ng)
        self.jj = self.jj.astype(np.float)
        self.ii = self.ii.astype(np.float)
        
        self.sum_matrix = float(np.sum(self.glrl_matrix))

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

        return np.sum(self.glrl_matrix) / self.sum_matrix

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