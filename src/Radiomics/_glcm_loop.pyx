import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport prange

cdef extern from "math.h" nogil:
    double sin(double)
    double cos(double)

@cython.boundscheck(False)
def _2d_glcm_loop(int [:,:] image, double distance, double angle, int bin_width):

    cdef int r, c, row, col, rows, cols
    cdef int i, j, ii, jj, min, max

    rows = image.shape[0]
    cols = image.shape[1]

    min = np.min(image)
    max = np.max(image)

    levels = (max - min) / bin_width + 1

    out = np.zeros([levels, levels], dtype=np.int)
    cdef int [:,:] out_view = out

    for r in prange(rows, nogil=True):

        for c in range(cols):

            i = image[r, c]

            row = <int>(r - distance * sin(angle))
            col = <int>(c + distance * cos(angle))

            if row >= 0 and row < rows and col >= 0 and col < cols:

                j = image[row, col]

                ii = (i - min) / bin_width
                jj = (j - min) / bin_width

                out_view[ii, jj] += 1

    return out

@cython.boundscheck(False)
def _3d_glcm_polar_loop(int [:,:,:] image, double distance, double phi, double theta, int bin_width):

    # phi : angle on x-y plane
    # theta : angle around z-axis

    cdef int x, y, z, xs, ys, zs, len_x, len_y, len_z
    cdef int i, j, ii, jj, min, max
    cdef double epsilon = 0.00001

    len_z = image.shape[0]
    len_x = image.shape[1]
    len_y = image.shape[2]

    min = np.min(image)
    max = np.max(image)

    levels = (max - min) / bin_width + 1

    out = np.zeros([levels, levels], dtype=np.int)
    cdef int [:,:] out_view = out

    for z in range(len_z):

        for x in range(len_x):

            for y in range(len_y):

                i = image[z, x, y]

                xs = x - <int>(distance * sin(theta) * sin(phi) + epsilon)
                ys = y + <int>(distance * sin(theta) * cos(phi) + epsilon)
                zs = z - <int>(distance * cos(theta) + epsilon)

                if xs >= 0 and xs < len_x and ys >= 0 and ys < len_y and zs >= 0 and zs < len_z:

                    j = image[zs, xs, ys]

                    ii = (i - min) / bin_width
                    jj = (j - min) / bin_width

                    out_view[ii, jj] += 1

    return out

@cython.boundscheck(False)
def _3d_glcm_vector_loop(int [:,:,:] image, double distance, int dx, int dy, int dz, int bin_width):

    cdef int x, y, z, xs, ys, zs, len_x, len_y, len_z
    cdef int i, j, ii, jj, min, max
    cdef double epsilon = 0.00001

    len_z = image.shape[0]
    len_x = image.shape[1]
    len_y = image.shape[2]

    min = np.min(image)
    max = np.max(image)

    levels = (max - min) / bin_width + 1

    out = np.zeros([levels, levels], dtype=np.int)
    cdef int [:,:] out_view = out

    for z in range(len_z):

        for x in range(len_x):

            for y in range(len_y):

                i = image[z, x, y]

                xs = x + <int>(distance * dx)
                ys = y + <int>(distance * dy)
                zs = z - <int>(distance * dz)

                if xs >= 0 and xs < len_x and ys >= 0 and ys < len_y and zs >= 0 and zs < len_z:

                    j = image[zs, xs, ys]

                    ii = (i - min) / bin_width
                    jj = (j - min) / bin_width

                    out_view[ii, jj] += 1

    return out