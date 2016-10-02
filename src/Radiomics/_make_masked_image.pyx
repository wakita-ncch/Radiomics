import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def _point_in_polygon(double x, double y, poly):

    cdef int n, i, inside
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

@cython.boundscheck(False)
def _make_masked_image(unsigned short [:,:] image, contours, int shape0, int shape1, double psX, double psY, double ippX, double ippY):

    cdef int x, y
    cdef double X, Y

    for y in range(shape0):

        for x in range(shape1):

            X = x * psX + ippX
            Y = y * psY + ippY

            for contour in contours:

                if _point_in_polygon(X, Y, contour) < 0:

                    image[y, x] = 0

    return image