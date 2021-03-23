import numpy as np


def normalise(arr):
    """
    Inputs:
    ----
     - arr (numpy.array): (H x W x N):
    Returns:
    ----
     -
    """
    mean = np.mean(arr, axis=1, keepdims=True)
    std = np.std(arr, axis=1, keepdims=True)
    return (arr - mean) / std


def flatten_3d(arr):
    """
    Inputs:
    ----
     - arr (numpy.array): (H x W x N):
    Returns:
    ----
    """
    return arr.reshape(-1, arr.shape[-1]).T


def sub2ind(array_shape, rows, cols):
    """
    """
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    """
    """
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def cropbox(img, bbox):
    """
    """
    h, w, c = img.shape
    if bbox[0] >= 1 and bbox[2] <= w and bbox[1] >= 1 and bbox[3] <= h:
        imgout = img[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
        h_pad = 24 - imgout.shape[0]
        w_pad = 24 - imgout.shape[1]
        return np.pad(imgout, ((0, h_pad), (0, w_pad), (0, 0)))
    else:
        xx, yy = np.meshgrid(range(bbox[0], bbox[2]+1), range(bbox[1], bbox[3]+1))
        g = 1
        while g:
            g = 0
            xx_flatten = xx.ravel()
            yy_flatten = yy.ravel()
            if (xx_flatten < 1).any():
                xx_flatten[xx_flatten < 1] = abs(xx_flatten[xx_flatten < 1]) + 2
                xx = xx_flatten.reshape(xx.shape)
            if (yy_flatten < 1).any():
                yy_flatten[yy_flatten < 1] = abs(yy_flatten[yy_flatten < 1]) + 2
                yy = yy_flatten.reshape(yy.shape)
            if (xx_flatten > w).any():
                xx_flatten[xx_flatten > w] = 2 * w - xx_flatten[xx_flatten > w] - 1
                xx = xx_flatten.reshape(xx.shape)
                g = 1
            if (yy_flatten > h).any():
                yy_flatten[yy_flatten > h] = 2 * h - yy_flatten[yy_flatten > h] - 1
                yy = yy_flatten.reshape(yy.shape)
                g = 1
        size_orig = np.array(img.shape)
        img = img.reshape(-1, 1)
        ind = sub2ind((h, w), yy.reshape(-1, 1), xx.reshape(-1, 1))
        size = np.array(xx.shape)
        if (size_orig > 2).any():
            size = np.concatenate((size, size_orig[2:]))

        return img[ind, :].reshape(size)
