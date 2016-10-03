import pyximport
import numpy as np
import math

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _glcm_loop import _3d_glcm_vector_loop 

import profiling_tools

"""
Group 3 : Textural features (Class version)

"""

def calc_entropy(array):

    # 0 x log(0) = 0

    _entropy = []

    for val in array.ravel():

        if val != 0:

            _entropy.append(float(val) * math.log(val, 2))

    return - np.sum(_entropy)

#def glcm_vector_loop(volume, distance, dx, dy, dz, bin_width):

#    #bin_width = int(bin_width)

#    #volume = (volume / bin_width) * bin_width

#    return _3d_glcm_vector_loop(volume, distance, dx, dy, dz, bin_width)

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

    def __init__(self, image, distance, vector, bin_width):

        bin_width = int(bin_width)

        image += (bin_width - 1)
        image /= bin_width

        self.image = image

        glcm_matrix = _3d_glcm_vector_loop(image, distance, vector[0], vector[1], vector[2])
        self.glcm_matrix = glcm_matrix.astype(np.float)

        self.glcm_matrix_norm = self.glcm_matrix / np.sum(self.glcm_matrix)

        levels = np.arange(1, self.glcm_matrix.shape[0]+1)
        self.levels = levels.astype(np.float)

        self.ii, self.jj = np.meshgrid(self.levels, self.levels)

        # ux/uy should be the marginal row/column probalilities

        self.ux, self.uy = np.meshgrid(np.sum(self.glcm_matrix_norm, axis=1),
                             np.sum(self.glcm_matrix_norm, axis=0))

        self.HXY = self.entropy()
        self.HX = calc_entropy(self.ux) 
        self.HY = calc_entropy(self.uy)
        self.HXY1 = self.HXY1()
        self.HXY2 = self.HXY2()

        self.p_plus = self.p_plus()

        print "HXY: ", self.HXY
        print "HX: ", self.HX
        print "HY: ", self.HY
        print "HXY1: ", self.HXY1
        print "HXY2: ", self.HXY2

        print "autocorrelation: ", self.autocorrelation()
        print "cluster prominence: ", self.cluster_prominence()
        print "cluster shade: ", self.cluster_shade()
        print "cluster tendency: ", self.cluster_tendency()
        print "contrast: ", self.contrast()
        print "correlation: ", self.correlation()
        print "difference entropy: ", self.difference_entropy()
        print "dissimilarity: ", self.dissimilarity()
        print "energy: ", self.energy()
        print "entropy: ", self.entropy()
        print "homogeneity 1: ", self.homogeneity1()
        print "homogeneity 2: ", self.homogeneity2()
        print "IMC 1:, ", self.IMC1()
        print "IMC 2:, ", self.IMC2()
        print "IDMN: ", self.IDMN()
        print "IDN: ", self.IDN()
        print "inverse variance: ", self.inverse_variance()
        print "maximum probability: ", self.maximum_probability()
        print "sum average: ", self.sum_average()
        print "sum entropy: ", self.sum_entropy()
        print "sum variance: ", self.sum_variance()
        print "variance: ", self.variance()

    def HXY1(self):

        _HXY1 = []

        _x_y = self.ux * self.uy

        for idx, elem in enumerate(self.glcm_matrix.ravel()):

            if _x_y.ravel()[idx] != 0:

                _HXY1.append(float(elem) * math.log(_x_y.ravel()[idx], 2))

        return - np.sum(_HXY1)

    def HXY2(self):

        return calc_entropy(self.ux * self.uy)

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

        return np.sum((self.ii * self.jj * self.glcm_matrix - self.ux * self.uy)/(np.std(self.ux) * np.std(self.uy)))

    def difference_entropy(self):

        _minus = np.abs(self.ii - self.jj)

        p_minus = np.zeros((1, len(self.levels)))

        for idx, elem in enumerate(self.glcm_matrix.ravel()):

            p_minus[0, int(_minus.ravel()[idx])] += elem

        return - calc_entropy(p_minus)

    def dissimilarity(self):

        return np.sum(np.abs(self.ii - self.jj) * self.glcm_matrix)

    def energy(self):

        return np.sum(self.glcm_matrix ** 2)

    def entropy(self):

        return calc_entropy(self.glcm_matrix)

    def homogeneity1(self):

        return np.sum(self.glcm_matrix / (1.0 + np.abs(self.ii - self.jj)))

    def homogeneity2(self):

        return np.sum(self.glcm_matrix / (1.0 + (self.ii - self.jj) ** 2))

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

        _inverse_variance = []

        for idx, _ii in enumerate(self.ii.ravel()):

            _jj = self.jj.ravel()[idx]

            if _ii != _jj:

                _inverse_variance.append(self.glcm_matrix.ravel()[idx] / (_ii - _jj) **2)

        return np.sum(_inverse_variance)

    def maximum_probability(self):

        return np.max(self.glcm_matrix)

    def sum_average(self):

        _sum_average = []

        for i, val in enumerate(self.p_plus):

            i  += 2

            _sum_average.append(i * val)

        return np.sum(_sum_average)

    def sum_entropy(self):

        return calc_entropy(self.p_plus)

    def sum_variance(self):

        _sum_variance = []
        _SE = self.sum_entropy()

        for i, val in enumerate(self.p_plus):

            i += 2

            _sum_variance.append((i - _SE) ** 2 * val)

        return np.sum(_sum_variance)

    def variance(self):

        _u = np.mean(self.glcm_matrix)

        return np.sum((self.ii - _u) ** 2 * self.glcm_matrix)

    def p_plus(self):

        _plus = self.ii + self.jj

        p_plus = np.zeros((1, 2 * len(self.levels) - 1))

        for idx, elem in enumerate(self.glcm_matrix.ravel()):

            p_plus[0, int(_plus.ravel()[idx]) - 2] += elem

        return p_plus.ravel()