
import argparse
import math
import copy
import numpy as np
import skimage.io
from scipy.signal import convolve


def BTV(z):
    h, w = z.shape
    Q = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    res = 0
    zpadded = np.pad(z, 1, mode='edge')
    for x, y in Q:
        res += np.sum(np.abs(zpadded[ 1 + y:h + 1 + y, 1 + x:w + 1 + x] - z)) / (x ** 2 + y ** 2) ** 0.5
    
    return res / len(Q)

def BTV2(z):
    h, w = z.shape
    Q = [(1, 0), (0, 1)]
    res = 0
    zpadded = np.pad(z, 1, mode='edge')
    for x, y in Q:
        res += np.sum(np.abs(zpadded[ 1 - y:h + 1 - y, 1 - x:w + 1 - x] + zpadded[ 1 + y:h + 1 + y, 1 + x:w + 1 + x] - 2 * zpadded[1:h + 1, 1: w + 1])) / (x ** 2 + y ** 2)
    
    return res / len(Q)


def func(z, A, u, alpha, beta):
    h, w = z.shape
    size = h * w
    h, w = A.shape
    h -= 1
    w -= 1
    return np.mean((convolve(np.pad(z, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode='edge'), A, mode='valid') - u) ** 2) + alpha * BTV(z) / size  + beta * BTV2(z) / size
    # return reg(z)


def D_BTV(z):
    h, w = z.shape
    Q = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    res = np.zeros_like(z)
    zpadded = np.pad(z, 1, mode='edge')
    for x, y in Q:
        sign = np.sign(zpadded[ 1 + y:h + 1 + y, 1 + x:w + 1 + x] - z)
        signpad = np.pad(sign, 1, mode='edge')
        res += (signpad[ 1 - y:h + 1 - y, 1 - x:w + 1 - x] - sign) / (x ** 2 + y ** 2) ** 0.5
        
    return res / len(Q)

def D_BTV2(z):
    h, w = z.shape
    Q = [(1, 0), (0, 1)]
    res = np.zeros_like(z)
    zpadded = np.pad(z, 1, mode='edge')
    for x, y in Q:
        sign = np.sign(zpadded[ 1 - y:h + 1 - y, 1 - x:w + 1 - x] + zpadded[ 1 + y:h + 1 + y, 1 + x:w + 1 + x] - 2 * z)
        signpad = np.pad(sign, 1, mode='edge')
        res += (signpad[ 1 - y:h + 1 - y, 1 - x:w + 1 - x] + signpad[ 1 + y:h + 1 + y, 1 + x:w + 1 + x] - 2 * sign) / (x ** 2 + y ** 2)
        
    return res / len(Q)


def D_func(z, A, u, alpha, beta):
    h, w = A.shape
    h -= 1
    w -= 1
    return 2 * convolve(np.pad((convolve(np.pad(z, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode='edge'), A, mode='valid') - u),
                               ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode='edge'), A[::-1, ::-1], mode='valid') + alpha * D_BTV(z)  + beta * D_BTV2(z)
    # return D_BTV(z)

def deconvolve(img, ker, alpha, beta):
    max_iter = 200
    lr = 2.0        
    mu = 0.0
    
    res = copy.deepcopy(img)
    v = np.zeros_like(res)
    i = 0
    desc = func(res, ker, img, alpha, beta)
    while i < max_iter:
        g = D_func(res, ker, img, alpha, beta)
        v = mu * v - g
        res += lr * v
        i += 1
        
        desc2 = func(res, ker, img, alpha, beta)
        if (desc - desc2) / desc < 1e-8 :
            lr /= 2
        desc = desc2
        # print(desc)
    return res
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Deconvolution',
        description='',
    )
    parser.add_argument('input_image')
    parser.add_argument('kernel')
    parser.add_argument('output_image')
    parser.add_argument('noise_level')
    args = parser.parse_args()
    
    img = skimage.io.imread(args.input_image)
    ker = skimage.io.imread(args.kernel)
    img = img.astype(float) / 255
    ker = ker.astype(float) / 255
    if len(img.shape) == 3:
        img = img[:, :, 0]
    if len(ker.shape) == 3:
        ker = ker[:, :, 0]
    ker = ker / ker.sum()
    noise = float(args.noise_level)
    
    if noise < 1:
        noise = 1
    alpha = -6.291493665719671e-08 * noise ** 5 + \
    1.4211790996814216e-06 * noise ** 4 + \
    2.270478936984355e-06 * noise ** 3 + \
    6.414923088249327e-05 * noise ** 2 + \
    0.0007796523936374943 * noise - 0.0004774303676199961
    
    beta = -6.411108384792595e-08 * noise ** 5 + \
    2.9077124142913616e-06 * noise ** 4 - \
    4.63593850107008e-05 * noise ** 3 + \
    0.0003222438027372238 * noise ** 2 - \
    0.0006548542612358402 * noise + 0.0004961262421788738
    
    res = deconvolve(img, ker, alpha, beta)
    res = np.clip(res, 0, 1)  
    res = (res * 255).astype(np.uint8) 
    skimage.io.imsave(args.output_image, res)
