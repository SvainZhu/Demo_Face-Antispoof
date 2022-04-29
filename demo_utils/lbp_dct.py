
import cv2, os, sys
import math
import numpy as np
from glob import glob
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern


def dict(mat, p):
    mat1 = np.transpose(mat)  # 转置矩阵
    lbp_dct_mat = dct(mat1)

    return lbp_dct_mat[:, 0:p]


def lbp_hist(lbp):
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    return hist


def lbp_matrix(db_dir):
    matrix = []
    for i in glob('%s/*' % db_dir):

        gray = cv2.imread(i, 0)

        # settings for LBP
        refs = {'i': local_binary_pattern(gray, 8, 1, 'nri_uniform')}

        i_lbp = lbp_hist(refs['i'])

        matrix.append(i_lbp)

    return np.array(matrix)


def lbp_dct(data_dir):

    m = lbp_matrix(data_dir)
    lbp_dct1 = dict(m, 1)  # transform in the first layer

    f = len(m)
    s = math.floor(f / 2)
    t = math.floor(s / 2)
    # transform in the second layer
    s1 = m[0:s, :]
    s2 = m[s:f, :]
    lbp_dct21 = dict(s1, 1)
    lbp_dct22 = dict(s2, 1)
    # transform in the third layer
    t1 = s1[0:t, :]
    t2 = s1[t:s, :]
    t3 = s2[0:t, :]
    t4 = s2[t:s, :]
    lbp_dct31 = dict(t1, 1)
    lbp_dct32 = dict(t2, 1)
    lbp_dct33 = dict(t3, 1)
    lbp_dct34 = dict(t4, 1)

    feature = np.concatenate([lbp_dct1, lbp_dct21, lbp_dct22, lbp_dct31, lbp_dct32, lbp_dct33, lbp_dct34], axis=1)
    # 把矩阵按列连接

    lbp_dct = list(np.reshape(feature, (59*7)))  # 数组转化为列表

    return lbp_dct
