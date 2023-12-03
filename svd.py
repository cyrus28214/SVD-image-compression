import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import os

def compress(path, rate):
    img = PIL.Image.open(path)
    m, n = img.size
    '''
    设原图像size为m*n
    则占用空间为m*n*3
    设保留k个奇异值
    压缩后占用空间为(m+n+1)*k*3*4
    压缩率为rate=(m+n+1)*k*4/(m*n)
    k = rate*m*n/((m+n+1)*4)
    '''
    k = int(rate*m*n/((m+n+1)*4))
    img = np.array(img).astype('float32')
    img = np.transpose(img, (2, 0, 1)) # (m, n, 3) -> (3, m, n)
    u, s, v = np.linalg.svd(img) # SVD分解
    u = u[..., :k] # 保留前k列
    s = s[:, :k] # 保留前k个奇异值
    v = v[:, :k] # 保留前k行
    return u, s, v

def decompress(u, s, v):
    img = (u * s[:, np.newaxis]) @ v # (3, m, k) * (3, 1, k) @ (3, k, n) -> (3, m, n)
    img = np.transpose(img, (1, 2, 0)) # (3, m, n) -> (m, n, 3)
    img = np.round(img.clip(0, 255)).astype('uint8')
    return img

def preview(path, rates = [
        1, 0.95, 0.9, 0.8, 0.7, 
        0.6, 0.5, 0.4, 0.3, 0.2
    ], col=5):
    row = (len(rates) + col - 1) // col
    fig, axes = plt.subplots(row, col)
    for i, rate in enumerate(rates):
        img = decompress(*compress(path, rate))
        ax = axes[i // col, i % col]
        ax.set_title(f'rate={rate}')
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def save(path, u, s, v):
    np.savez(path, u=u, s=s, v=v)

def load(path):
    d = np.load(path)
    return d['u'], d['s'], d['v']

preview('Lenna.png')
