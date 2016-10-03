import vtk
import math

import pyximport
import numpy as np

pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

from _maximum_3d_distance import _maximum_3d_distance

import profiling_tools

"""
Group 2 : Shape and size based features
polydata : VOI as a vtkPolydata
V : Volume
A : Surface area
"""

def group2_features(polydata):

    V = volume(polydata)
    A = surface_area(polydata)

    points_array = []
    
    for i in range(polydata.GetNumberOfPoints()):

        point = polydata.GetPoint(i)

        points_array.append(point)

    points_array = np.array(points_array)

    group2_features = {}
    group2_features["compactness 1"] = compactness_1(V, A)
    group2_features["compactness 2"] = compactness_2(V, A)
    group2_features["maximum 3d diameter"] = c_maximum_diameter(points_array)
    group2_features["spherical disproportion"] = spherical_disproportion(V, A)
    group2_features["sphericity"] = sphericity(V, A)
    group2_features["surface area"] = A
    group2_features["surface to volume ratio"] = surface_to_volume_ratio(V, A)
    group2_features["volume"] = V

    print group2_features

    return group2_features

def compactness_1(V, A):

    return V/(math.sqrt(math.pi) * A ** (2.0/3.0))

def compactness_2(V, A):

    return (36.0 * math.pi * V ** 2) / (A ** 3)

def vtk_maximum_3d_diameter(polydata):

    maximum_diameter = 0.0

    for i in range(polydata.GetNumberOfPoints()):

        point1 = polydata.GetPoint(i)

        for j in range(i+1, polydata.GetNumberOfPoints()):

            point2 = polydata.GetPoint(j)

            squared_distance = vtk.vtkMath.Distance2BetweenPoints(point1, point2)

            if squared_distance > maximum_diameter: maximum_diameter = squared_distance

    return math.sqrt(maximum_diameter)

def p_maximum_diameter(points_array):

    # points_array: np.ndarray

    maximum_diameter = 0.0

    for i, point1 in enumerate(points_array):

        for point2 in points_array[i:]:

            squared_distance = (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2

            if squared_distance > maximum_diameter: maximum_diameter = squared_distance

    return math.sqrt(maximum_diameter)

def c_maximum_diameter(points_array):

    # points_array : np.ndarray

    return _maximum_3d_distance(points_array)

def spherical_disproportion(V, A):

    # Calculate the radius of a sphere with the same volume as the tumor

    R = ((3 * V) / (4 * math.pi)) ** (1.0/3.0)

    return A / (4 * math.pi * R ** 2)

def sphericity(V, A):

    return (math.pi ** (1.0/3.0)) * ((6 * V) ** (2.0/3.0)) / A

def surface_area(polydata):

    mass = vtk.vtkMassProperties()
    mass.SetInput(polydata)
    mass.Update()

    return mass.GetSurfaceArea()

def surface_to_volume_ratio(V, A):

    return A / V

def volume(polydata):

    mass = vtk.vtkMassProperties()
    mass.SetInput(polydata)
    mass.Update()

    return mass.GetVolume()