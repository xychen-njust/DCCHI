import numpy as np
# from function.proje import *
def proj(x, norm):
    (m, n) = x.shape
    z = 0.25
    if norm == 111:
        x = np.sign(x) * np.minimum(np.abs(x), 1)
    if norm == 211:
        temp = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
        x = x / np.tile(np.maximum(temp, 1), (1, x.shape[1]))


    if norm == 121:
        x = x.reshape((m // 2, 2, n))
        temp = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
        x = x / np.tile(np.maximum(temp, 1), (x.shape[1],1))
        x = x.reshape((m, n))

    if norm == 221:
        x = x.reshape((m // 2, 2 * n))
        temp = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
        x = x / np.tile(np.maximum(temp, 1), (1,x.shape[1]))
        x = x.reshape((m, n))

    if norm == -111:

        x = x.T
        (m, n) = x.shape
        y=np.abs(x)
        v = -1* np.sort(-1* y, axis =  0)
        w = np.cumsum(v, axis=0)
        i = np.tile(np.arange(1,m+1,1).reshape(m,1), (1,n))
        P = ((v - (w - z)/i) > 0) + 0
        P = np.vstack(((-1* P+1), np.ones((1, n))))
        p = np.argmax(P, axis=0)
        p=p+1
        p = np.maximum(1, p - 1)
        pl = p + (np.arange(0,n,1) * m)
        pl=pl-1
        (q1,q2)=w.shape
        w2=w.reshape((q1*q2,1),order='F')
        w=w2[pl].T
        theta = np.maximum(0, (w - z) / p)
        x = np.sign(x) * np.maximum(np.abs(x) - np.tile(theta, (m, 1)), 0)
        x=x.T

    # if norm == 123:
    #     x = x.reshape((m // 2, 2, n))
    #     for i in range(n):
    #         x[:,:, i] = proje(x[:,:, i], z)
    #     x = x.reshape((m, n))
    #
    # if norm == 124:
    #     x = x.reshape((m // 2, 2 * n))
    #     x = proje(x, z)
    #     x = x.reshape((m, n))

    return x
# a=np.array([[1,2,3],[3,4,5]])
# print(proj(a,211))