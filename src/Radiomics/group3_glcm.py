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

def group3_glcm_features(image, distance):

    autocorrelation = []
    cluster_prominence = []
    cluster_shade = []
    cluster_tendency = []
    contrast = []
    correlation = []
    difference_entropy = []
    dissimilarity = []
    energy = []
    entropy = []
    homogeneity_1 = []
    homogeneity_2 = []
    IMC_1 = []
    IMC_2 = []
    IDMN = []
    IDN = []
    inverse_variance = []
    maximum_probability = []
    sum_average = []
    sum_entropy = []
    sum_variance = []
    variance = []

    vectors = [[0, 1, 1],[-1, 1, 1],[-1, 0, 1], 
               [-1, -1, 1],[0, -1, 1],[1, -1, 1], 
               [1, 0, 1],[1, 1, 1],[0, 0, 1],
               [0, 1, 0],[-1, 1, 0],[-1, 0, 0],[-1, -1, 0]]

    glcm_features = {}

    for vector in vectors[1:]:

        glcm_matrix = GLCM_Matrix(image, distance, vector)

        autocorrelation.append(glcm_matrix.autocorrelation())
        cluster_prominence.append(glcm_matrix.cluster_prominence())
        cluster_shade.append(glcm_matrix.cluster_shade())
        cluster_tendency.append(glcm_matrix.cluster_tendency())
        contrast.append(glcm_matrix.contrast())
        correlation.append(glcm_matrix.correlation())
        difference_entropy.append(glcm_matrix.difference_entropy())
        dissimilarity.append(glcm_matrix.dissimilarity())
        energy.append(glcm_matrix.energy())
        entropy.append(glcm_matrix.entropy())
        homogeneity_1.append(glcm_matrix.homogeneity1())
        homogeneity_2.append(glcm_matrix.homogeneity2())
        IMC_1.append(glcm_matrix.IMC1())
        IMC_2.append(glcm_matrix.IMC2())
        IDMN.append(glcm_matrix.IDMN())
        IDN.append(glcm_matrix.IDN())
        inverse_variance.append(glcm_matrix.inverse_variance())
        maximum_probability.append(glcm_matrix.maximum_probability())
        sum_average.append(glcm_matrix.sum_average())
        sum_entropy.append(glcm_matrix.sum_entropy())
        sum_variance.append(glcm_matrix.sum_variance())
        variance.append(glcm_matrix.variance())

    glcm_features['autocorrelation'] = np.mean(autocorrelation)
    glcm_features['cluster_prominence'] = np.mean(cluster_prominence)
    glcm_features['cluster_shade'] = np.mean(cluster_shade)
    glcm_features['cluster_tendency'] = np.mean(cluster_tendency)
    glcm_features['contrast'] = np.mean(contrast)
    glcm_features['correlation'] = np.mean(correlation)
    glcm_features['difference_entropy'] = np.mean(difference_entropy)
    glcm_features['dissimilarity'] = np.mean(dissimilarity)
    glcm_features['energy'] = np.mean(energy)
    glcm_features['entropy'] = np.mean(entropy)
    glcm_features['homogeneity_1'] = np.mean(homogeneity_1)
    glcm_features['homogeneity_2'] = np.mean(homogeneity_2)
    glcm_features['IMC_1'] = np.mean(IMC_1)
    glcm_features['IMC_2'] = np.mean(IMC_2)
    glcm_features['IDMN'] = np.mean(IDMN)
    glcm_features['IDN'] = np.mean(IDN)
    glcm_features['inverse_variance'] = np.mean(inverse_variance)
    glcm_features['maximum_probability'] = np.mean(maximum_probability)
    glcm_features['sum_average'] = np.mean(sum_average)
    glcm_features['sum_entropy'] = np.mean(sum_entropy)
    glcm_features['sum_variance'] = np.mean(sum_variance)
    glcm_features['variance'] = np.mean(variance)

    print glcm_features

    return glcm_features

class GLCM_Matrix:

    def __init__(self, image, distance, vector):

        self.image = image

        glcm_matrix = _3d_glcm_vector_loop(image, distance, vector[0], vector[1], vector[2])

        # normalize glcm_matrix

        self.glcm_matrix = glcm_matrix.astype(np.float) / np.sum(glcm_matrix)

        levels = np.arange(1, self.glcm_matrix.shape[0]+1)
        self.levels = levels.astype(np.float)

        self.ii, self.jj = np.meshgrid(self.levels, self.levels)

        # ux/uy should be the marginal row/column probalilities

        self.ux, self.uy = np.meshgrid(np.sum(self.glcm_matrix, axis=1),
                             np.sum(self.glcm_matrix, axis=0))

        self.HXY = self.entropy()
        self.HX = calc_entropy(self.ux) 
        self.HY = calc_entropy(self.uy)
        self.HXY1 = self.HXY1()
        self.HXY2 = self.HXY2()

        self.p_plus = self.p_plus()

        #print "glcm: ", self.glcm_matrix

        #print "HXY: ", self.HXY
        #print "HX: ", self.HX
        #print "HY: ", self.HY
        #print "HXY1: ", self.HXY1
        #print "HXY2: ", self.HXY2

        #print "autocorrelation: ", self.autocorrelation()
        #print "cluster prominence: ", self.cluster_prominence()
        #print "cluster shade: ", self.cluster_shade()
        #print "cluster tendency: ", self.cluster_tendency()
        #print "contrast: ", self.contrast()
        #print "correlation: ", self.correlation()
        #print "difference entropy: ", self.difference_entropy()
        #print "dissimilarity: ", self.dissimilarity()
        #print "energy: ", self.energy()
        #print "entropy: ", self.entropy()
        #print "homogeneity 1: ", self.homogeneity1()
        #print "homogeneity 2: ", self.homogeneity2()
        #print "IMC 1:, ", self.IMC1()
        #print "IMC 2:, ", self.IMC2()
        #print "IDMN: ", self.IDMN()
        #print "IDN: ", self.IDN()
        #print "inverse variance: ", self.inverse_variance()
        #print "maximum probability: ", self.maximum_probability()
        #print "sum average: ", self.sum_average()
        #print "sum entropy: ", self.sum_entropy()
        #print "sum variance: ", self.sum_variance()
        #print "variance: ", self.variance()

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