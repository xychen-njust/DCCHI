# import config
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio
from numpy.linalg import *
from proj import *
from gradie import *
from scipy import sparse
from scipy.sparse import hstack

batch_size,c, m, n = [1,242, 500, 256]

respon = sio.loadmat('./sp_response.mat')
res0 = respon['spec_resp']
res = (res0[:,1])

pan_matrix = sparse.spdiags((res[0] * (np.ones((m*n, 1)))).transpose(), 0, m*n, m*n)
for k in range(1, 31):
    sx = sparse.spdiags((res[k] * (np.ones((m*n, 1)))).transpose(), 0, m*n, m*n)
    pan_matrix = hstack((pan_matrix, sx))
for k in range(31, c):
    sx = sparse.spdiags((res[3]* (np.ones((m*n, 1)))).transpose(), 0, m*n, m*n)
    pan_matrix = hstack((pan_matrix, sx))

T = np.round(np.random.rand(m, n))
T = T.reshape(1, m*n)
cassi_matrix = sparse.spdiags(T, 0, m * (n+c-1), m * n)
for k in range(1, c):
    sx = sparse.spdiags(T, -k * m, m * (n+c-1), m * n)

    cassi_matrix = hstack((cassi_matrix, sx))

cassi_matrix.tocsr()
pan_matrix.tocsr()
# 观测矩阵
path = './1.mat'
# 数据
data = sio.loadmat(path)
result,twist = data['result'],data['twist']
orig = data["label"]
# orig = orig.transpose(2,0,1)
orig = orig / (np.max(orig) - np.min(orig))
# orig = np.flip(orig, axis=1)
# 图像归一化

# plt.imshow(orig[20,:,:])
# plt.show()
c, m, n = orig.shape
f = orig.reshape((c,m*n), order = 'F')
f = f.reshape((c*m*n))
if cassi_matrix.shape[1] == f.shape[0]:
    HSI_cassi = cassi_matrix @ f

    HSI_pan = pan_matrix @ f
# HSI_cassi = HSI_cassi.reshape((m,n+c-1), order = 'F')
# plt.imshow(HSI_cassi)
# plt.show()
# HSI_pan = HSI_pan.reshape((m,n), order = 'F')
# plt.imshow(HSI_pan)
# plt.show()
# 输入数据
f0 = cassi_matrix.T @ HSI_cassi + pan_matrix.T @ HSI_pan
f = np.zeros((c*m*n))
h = np.zeros((c*m*n))
h2 = np.zeros(((c,m*n)))
# input = f0.reshape((c,m*n))
# input = input.reshape((c,m,n), order = 'F')
# plt.imshow(input[1,:,:])
# plt.show()
K = gradie(m, n)
D = np.zeros((2 * m * n, c))
normctv = 211
kesi = 0.0003
# 梯度下降
eta = 0.005
tau = 0.00005
# 正则化参数
# Phi = (1-kesi*eta)*np.identity(m*n*c) - kesi*cassi_matrix.T@cassi_matrix- kesi*pan_matrix.T@pan_matrix
for i in range(100):
    f00 = f
    for j in range(1):
        t=f
        temp_1 = cassi_matrix @ f
        temp_2 = pan_matrix @ f
        temp = (1-kesi*eta) * f - kesi*cassi_matrix.T@temp_1-kesi*pan_matrix.T@temp_2
        f = temp + kesi * f0 +kesi *eta*h
        dout = f.reshape((c, m * n))
        dout = dout.reshape((c, m, n), order='F')
        # plt.imshow(dout[23, :, :])
        # plt.show()
        tol = norm(f-t) / norm(t)
        # print(tol)
        if  tol<0.1:
            break
    Nuu4 = f.reshape((c,m*n)).T
    oufa = tau / eta
    gama = 0.2
    if not (f.all() == 0):
        for iter2 in range(1):
            D0 = D
            D = D - gama * K @ (K.T @ D - Nuu4 / oufa)
            D = proj(D, normctv)
            if not (norm(D - D0, 'fro') == 0):
                tol1 = norm(D0 - D) / norm(D0)
            else:
                tol1 = 1
            # tol1 = norm(D0 - D) /norm(D0)
            if tol1 < 0.001:
                break
            # print(tol1)
    V33 = (Nuu4 - oufa * K.T @ D)
    h2 = V33.T
    h = h2.reshape((c*m*n))
    tol = norm(f-f00) / norm(f00)
    print(tol)
    if tol < 0.001:
        break
print(i)
dout = f.reshape((c,m*n))
dout = dout.reshape((c,m,n), order = 'F')
# plt.imshow(dout[23,:,:])
# plt.show()
#
# input = orig
# plt.imshow(input[23,:,:])
# plt.show()
res = dout - orig
dout = orig+res/2
sio.savemat('./2.mat', {'ae': dout, 'label': orig,'result':result,'twist':twist},do_compression=True)