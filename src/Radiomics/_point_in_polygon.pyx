import numpy as np
cimport cython
from cython.parallel cimport prange
#from libcpp cimport bool

def _point_in_polygon(double x, double y, poly):

    cdef int n, i, inside
    #cdef bool inside
    cdef double p1x, p1y, p1z, p2x, p2y, p2z, xinters

    n = len(poly)
    inside = -1
    p1x, p1y, p1z = poly[0]
    for i in range(n+1):
        p2x, p2y, p2z = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside *= -1 
        p1x,p1y = p2x,p2y
        
    return inside