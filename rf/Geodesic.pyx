from libc.math cimport sqrt

def sweep(float[:,:] A, float[:,:] Cost):
    cdef int i, j
    cdef float t0, t1, t2, C, max_diff = 0.0
    for i in xrange(1, A.shape[0]):
        for j in xrange(1, A.shape[1]):
            t1, t2 = A[i, j-1], A[i-1, j]
            C = Cost[i, j]
            if abs(t1-t2) > C:
                t0 = min(t1, t2) + C  # handle degenerate case
            else:    
                t0 = 0.5*(t1 + t2 + sqrt(2*C**2 - (t1-t2)**2))
            max_diff = max(max_diff, A[i, j] - t0)
            A[i, j] = min(A[i, j], t0)
    return max_diff

import itertools as it

def GDT(float[:,:] A, float[:,:] C):
    A = A.copy()
    sweeps = [A, A[:,::-1], A[::-1], A[::-1,::-1]]
    costs = [C, C[:,::-1], C[::-1], C[::-1,::-1]]
    for i, (a, c) in enumerate(it.cycle(zip(sweeps, costs))):
        print (i),
        if sweep(a, c) < 1.0 or i >= 40:
            break
    return A
