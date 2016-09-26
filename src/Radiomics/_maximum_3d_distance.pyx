import numpy as np
#cimport numpy as np
cimport cython
from cython.parallel cimport prange
#from cython.view cimport array as cvarray

cdef extern from "math.h":
    double sqrt(double)

@cython.boundscheck(False)
def _maximum_3d_distance(double [:,:] points_array):

    cdef:
        int i, j, rows, n=0
        double x1, y1, z1, x2, y2, z2, max_distance=0.0, distance

#def _maximum_3d_distance(np.ndarray[dtype=np.float64_t, ndim=2, negative_indices=False, mode='c'] points_array):

#    cdef:
#        np.int32_t i, j, rows, num, n=0
#        np.float64_t x1, y1, z1, x2, y2, z2, max_distance=0.0, distance

    rows = points_array.shape[0]
    distances = np.zeros([rows, rows], dtype=np.float64)

    cdef double [:,:] distances_view = distances

    for i in prange(0, rows, nogil=True):

        x1 = points_array[i, 0]
        y1 = points_array[i, 1]
        z1 = points_array[i, 2]

        for j in range(i+1, rows):

            x2 = points_array[j, 0]
            y2 = points_array[j, 1]
            z2 = points_array[j, 2]

            distance = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2

            distances_view[i,j] = distance

    return sqrt(np.max(distances))