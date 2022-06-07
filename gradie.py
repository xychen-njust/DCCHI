import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy.sparse import kron
# from function.spkron import spkron
def gradie(w, h):
    # e = np.ones(w)
    # yDerMat = np.spdiags([-e e], 0:1, w, w)
    y1 = np.ones((w, 1)) * [-1, 1]
    y1[w - 1, 0] = 0
    # print(y1)
    x1 = np.ones((h, 1)) * [-1, 1]
    x1[h - 1, 0] = 0
    yDerMat = sparse.spdiags(y1.transpose(), [0, 1], w, w)
    xDerMat = sparse.spdiags(x1.transpose(), [0, 1], h, h)
    # print(xDerMat)
    # return K
    sx = sparse.spdiags((np.ones((w, 1))).transpose(), 0, w, w)
    # print(sx)
    sy = sparse.spdiags((np.ones((h, 1))).transpose(), 0, h, h)
    # print(np.kron(xDerMat.T, sx))
    coo1 = coo_matrix(kron(xDerMat.T, sx))
    coo2 = coo_matrix(kron(sy, yDerMat))

    K = vstack((coo1, coo2))
    K.tocsr()
    # print(K)
    return K
# x = gradie(2,2)
# print(x)