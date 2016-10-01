import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h" nogil:
    double sqrt(double)

def _run_length(int [:] array, int [:,:] out):

    cdef int i, len_array, value
    cdef int last_value = -1, count = 0

    len_array = len(array)

    for i, value in enumerate(array):

        if i == 0:

            last_value = value
            count = 1

        elif 0 < i < len_array - 1:

            if value == last_value:

                count += 1

            elif value != last_value:

                out[last_value, count - 1] += 1

                last_value = value
                count = 1

        elif i == len_array - 1:

            if value == last_value:

                count += 1
                out[last_value, count - 1] += 1

            elif value != last_value:

                out[last_value, count - 1] += 1
                out[value, 0] += 1

def _boundary_check(int xs, int ys, int zs, int len_x, int len_y, int len_z):

    if xs >= 0 and xs < len_x and ys >= 0 and ys < len_y and zs >= 0 and zs < len_z:

        return True

    else:

        return False

def _calc_run_length(int [:,:,:] image, 
                     int x0, int y0, int z0, 
                     int dx, int dy, int dz, 
                     int [:,:] out):

    cdef int xs, ys, zs, len_x, len_y, len_z, len_array
    cdef int n = 0

    len_z = image.shape[0]
    len_x = image.shape[1]
    len_y = image.shape[2]

    len_array = <int>(sqrt((len_x * dx) ** 2 + (len_y * dy) ** 2 + (len_z * dz) ** 2))

    array = np.zeros(len_array, dtype = np.int)

    xs = x0
    ys = y0
    zs = z0
    
    while _boundary_check(xs, ys, zs, len_x, len_y, len_z):

        array[n] = image[zs, xs, ys]

        xs += dx
        ys += dy
        zs += dz
        n += 1

    _run_length(array, out)

def _form_np_matrix(np.ndarray out):

    # np.ndarray[dtype=np.int64_t, ndim=2, negative_indices=False, mode='c'] got error

    cdef int i, val, idx

    out = out[1:,:]
    sum_col = np.sum(out, axis=0)

    for i, val in enumerate(sum_col[::-1]):

        if val != 0:

            break

    return out[:,0:-i]
    
@cython.boundscheck(False)
def _glrl_vector_loop(int [:,:,:] image, int direction):

    # image : converted to gray levels

    cdef int x, y, z, xs, ys, zs, len_x, len_y, len_z
    cdef int i, j, ii, jj, _min, _max, n
    cdef int dx, dy, dz, max_len_of_out
    cdef int last_num = -1, count = 0

    len_z = image.shape[0]
    len_x = image.shape[1]
    len_y = image.shape[2]

    _min = np.min(image)
    _max = np.max(image)

    levels = _max + 1
    max_len_of_out = <int>(sqrt(len_x ** 2 + len_y ** 2 + len_z ** 2))

    out = np.zeros([levels, max_len_of_out], dtype = np.int) # you should determine the size of out
    cdef int [:,:] out_view = out

    if direction == 1:

        dx = 0
        dy = 1
        dz = 0

        y = 0

        for z in range(len_z):

            for x in range(len_x):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 2:

        dx = -1
        dy = 1
        dz = 0

        for z in range(len_z):
   
            y = 0

            for x in range(len_x):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

            x = len_x - 1

            for y in range(1, len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 3:

        dx = 1
        dy = 0
        dz = 0

        x = 0

        for z in range(len_z):

            for y in range(len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 4:

        dx = 1
        dy = 1
        dz = 0

        for z in range(len_z):

            y = 0

            for x in range(len_x):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

            x = 0

            for y in range(1, len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 5:

        dx = 0
        dy = 1
        dz = -1

        y = 0

        for z in range(len_z):

            for x in range(len_x):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)      

        z = len_z - 1

        for y in range(1, len_y):

            for x in range(len_x):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 6:

        dx = -1
        dy = 1
        dz = -1

        y = 0

        for z in range(len_z):

            for x in range(len_x):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        z = len_z - 1

        for y in range(1, len_y):

            for x in range(len_x):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        x = len_x - 1

        for z in range(len_z - 1):

            for y in range(1, len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 7:

        dx = -1
        dy = 0
        dz = -1

        x = len_x - 1

        for z in range(len_z):

            for y in range(len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        z = len_z - 1

        for x in range(len_x - 1):

            for y in range(len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 8:

        dx = -1
        dy = -1
        dz = -1

        x = len_x - 1

        for z in range(len_z):

            for y in range(len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        z = len_z - 1

        for x in range(len_x - 1):

            for y in range(len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        y = len_y - 1

        for x in range(len_x - 1):

            for z in range(len_z - 1):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 9:

        dx = 0
        dy = -1
        dz = -1

        y = len_y - 1

        for x in range(len_x):

            for z in range(len_z):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        z = len_z - 1

        for x in range(len_x):

            for y in range(len_y - 1):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 10:

        dx = 1
        dy = -1
        dz = -1

        x = 0

        for y in range(len_y):

            for z in range(len_z):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        y = len_y - 1

        for x in range(1, len_x):

            for z in range(len_z):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        z = len_z - 1

        for x in range(1, len_x):

            for y in range(len_y - 1):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 11:

        dx = 1
        dy = 0
        dz = -1

        x = 0

        for y in range(len_y):

            for z in range(len_z):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        z = len_z - 1

        for x in range(1, len_x):

            for y in range(len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 12:

        dx = 1
        dy = 1
        dz = -1

        y = 0

        for x in range(len_x):

            for z in range(len_z):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        x = 0

        for y in range(1, len_y):

            for z in range(len_z):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

        z = len_z - 1

        for x in range(1, len_x):

            for y in range(1, len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)

    elif direction == 13:

        dx = 0
        dy = 0
        dz = -1

        z = len_z - 1

        for x in range(len_x):

            for y in range(len_y):

                _calc_run_length(image, x, y, z, dx, dy, dz, out_view)
    
    return _form_np_matrix(out)