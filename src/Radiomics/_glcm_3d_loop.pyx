import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double tan(double)

@cython.boundscheck(False)
def _glcm_3d_loop(int [:,:,:] image,
                  double [:] distances,
                  double [:] alphas,
                  double [:] betas,
                  int levels,
                  int [:,:,:,:,:] out):

    cdef:
        int a_idx, b_idx, d_idx
        int xs, ys, zs, x, y, z, len_x, len_y, len_z


#def _glcm_3d_loop(np.ndarray[dtype=np.uint8_t, ndim=3, negative_indices=False, mode='c'] image,
#                  np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] distances,
#                  np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] alphas,
#                  np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] betas,
#                  int levels,
#                  np.ndarray[dtype=np.uint32_t, ndim=5, negative_indices=False, mode='c'] out):

#    cdef:
#        np.int32_t a_idx, b_idx, d_idx
#        np.int32_t xs, ys, zs, x, y, z, len_x, len_y, len_z
#        np.int32_t i, j

    len_x = image.shape[0]
    len_y = image.shape[1]
    len_z = image.shape[2]

    for a_idx, alpha in enumerate(alphas):

        for b_idx, beta in enumerate(betas):

            for d_idx, distance in enumerate(distances):

                for x in range(len_x):

                    for y in range(len_y):

                        for z in range(len_z):

                            i = image[x, y, z]

                            # compute the location of the offset pixel
                            # alpha : angle on the x-y plane
                            # beta : angle around the z-axis

                            xs = x + <int>(distance * cos(beta) * cos(alpha) + 0.5)
                            ys = y + <int>(distance * cos(beta) * sin(alpha) + 0.5)
                            zs = z + <int>(distance * sin(beta) + 0.5)

                            # make sure the offset is within bounds

                            if xs >= 0 and xs < len_x and ys >= 0 and ys < len_y and zs >= 0 and zs < len_z:

                                j = image[xs, ys, zs]

                                if i >= 0 and i < levels and j >= 0 and j < levels:

                                    out[i, j, d_idx, a_idx, b_idx] += 1