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

@cython.boundscheck(False)
def _form_stacked_image(stacked_image): #unsigned short [:,:,:] stacked_image

    cdef int x, idx, idx_first, idx_last
    cdef int y, idy, idy_first, idy_last

    sum_x = np.sum(np.sum(stacked_image, axis = 0), axis = 1)
    sum_y = np.sum(np.sum(stacked_image, axis = 1), axis = 1)

    for idx, x in enumerate(sum_x):

        if x != 0:

            idx_first = idx

            break

    for idx, x in enumerate(sum_x[::-1]):

        if x != 0:

            idx_last = idx

            break

    for idy, y in enumerate(sum_y):

        if y != 0:

            idy_first = idy

            break

    for idy, y in enumerate(sum_y[::-1]):

        if y != 0:

            idy_last = idy

            break

    formed_array = stacked_image[idy_first:-idy_last, idx_first:-idx_last, :] 

    return formed_array