import pyximport
import numpy as np
import math

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glcm_loop import _3d_glcm_vector_loop 

import profiling_tools

"""
Group 3 : Textural features (Class version)
"""

def glcm_vector_loop(volume, distance, dx, dy, dz, bin_width):

    bin_width = int(bin_width)

    volume = (volume / bin_width) * bin_width

    return _3d_glcm_vector_loop(volume, distance, dx, dy, dz, bin_width)

class GLCM_Matrices:

    def __init__(self, volume, bin_width):

        self.volume = volume
        self.bin_width = bin_width

        self.vectors = [[0, 1, 1],[-1, 1, 1],[-1, 0, 1], 
                        [-1, -1, 1],[0, -1, 1],[1, -1, 1], 
                        [1, 0, 1],[1, 1, 1],[0, 0, 1],
                        [0, 1, 0],[-1, 1, 0],[-1, 0, 0], 
                        [-1, -1, 0]]

        self.glcm_matrices = []

    def calcGLCMMatrices(self):

        for vector in self.vectors:

            self.glcm_matrices.append(GLCM_Matrix(self.volume, distance, vector, self.bin_width))

class GLCM_Matrix:

    def __init__(self, volume, distance, vector, bin_width, normal = True):

        glcm_matrix = glcm_vector_loop(volume, distance, vector[0], vector[1], vector[2], bin_width)

        if normal:

            self.glcm_matrix = glcm_matrix/np.sum(glcm_matrix).astype(np.float)

        else:

            self.glcm_matrix = glcm_matrix

        self.levels = np.arange(1, self.glcm_matrix.shape[0]+1)

        self.ii, self.jj = np.meshgrid(self.levels, self.levels)

        self.ux, self.uy = np.meshgrid(np.sum(self.glcm_matrix, axis=1),
                             np.sum(self.glcm_matrix, axis=0))

        self.HXY = self.entropy()
        self.HX = - np.sum(self.ux * np.log2(self.ux))
        self.HY = - np.sum(self.uy * np.log2(self.uy))
        self.HXY1 = - np.sum(self.glcm_matrix * np.log2(self.ux * self.uy))
        self.HXY2 = - np.sum(self.ux * self.uy * np.log2(self.ux * self.uy))

        ## calculate P(x+y)

        #sum_ii_jj = self.ii + self.jj
        #sum_matrix = np.zeros((1, 2 * len(self.levels) - 1))

        #for idx, elem in self.glcm_matrix.ravel():

        #    s_idx = sum_ii_jj.ravel()[idx] - 2 * self.levels[0]

        #    sum_matrix[s_idx] += elem

        #self.sum_matrix = sum_matrix

    def autocorrelation(self):

        return np.sum(self.ii * self.jj * self.glcm_matrix)

    def cluster_prominence(self):

        return np.sum((self.ii + self.jj - self.ux - self.uy) ** 4 * self.glcm_matrix)

    def cluster_shade(self):

        return np.sum((self.ii + self.jj - self.ux - self.uy) ** 3 * self.glcm_matrix)

    def cluster_tendency(self):

        return np.sum((self.ii + self.jj - self.ux - self.uy) ** 2 * self.glcm_matrix)

    def contrast(self):

        return np.sum((self.ii - self.jj) ** 2 * self.glcm_matrix)

    def correlation(self):

        # return np.sum((ii * jj * self.glcm_matrix - ux * uy)/(np.std(ux) * np.std(uy)))

        pass

    def difference_entropy(self):

        # calculate P(x-y)

        subtraction_matrix = np.abs(self.ii - self.jj)

        difference_entropy = np.zeros((1, len(self.levels)))

        for idx, elem in self.glcm_matrix.ravel():

            difference_entropy[0, subraction_matrix.ravel()[index]] += elem

        return np.sum(difference_entropy * np.log2(difference_entropy))

    def dissimilarity(self):

        return np.sum(np.abs(self.ii - self.jj) * self.glcm_matrix)

    def energy(self):

        return np.sum(self.glcm_matrix ** 2)

    def entropy(self):

        return - np.sum(self.glcm_matrix * np.log2(self.glcm_matrix))

    def homogeneity1(self):

        return np.sum(self.glcm_matrix / (1.0 + np.abs(self.ii - self.jj)).astype(np.float))

    def homogeneity2(self):

        return np.sum(self.glcm_matrix / (1.0 + (self.ii - self.jj) ** 2).astype(np.float))

    def IMC1(self):

        # IMC1 : Informational measure of correlation 1

        return (self.HXY - self.HXY1) / max(self.HX, self.HY)

    def IMC2(self):

        # IMC2 : Informational measure of correlation 2

        return math.sqrt(1.0 - math.exp(-2 * (self.HXY2 - self.HXY)))

    def IDMN(self):

        # IDMN : Inverse difference moment normalized

        return np.sum(self.glcm_matrix / (1.0 + (self.ii - self.jj) ** 2 / len(self.levels) ** 2))

    def IDN(self):

        # IDN : Inverse difference normalized

        return np.sum(self.glcm_matrix / (1.0 + np.abs(self.ii - self.jj) / len(self.levels)))

    def inverse_variance(self):

        return np.sum(self.glcm_matrix / (self.ii - self.jj) ** 2)

    def maximum_probability(self):

        return np.max(self.glcm_matrix)

    def sum_average(self):

        return np.sum(np.arange(2, 2 * len(self.levels) + 1) * self.sum_matrix)

    def sum_entropy(self):

        return - np.sum(self.sum_matrix * np.log2(self.sum_matrix))

    def sum_variance(self):

        SE = self.sum_entropy()

        return np.sum((np.arange(2, 2 * len(self.levels) + 1) - SE) ** 2 * self.sum_matrix)

    def variance(self):

        u = np.mean(self.glcm_matrix)

        return np.sum((self.levels - u) ** 2 * self.glcm_matrix)