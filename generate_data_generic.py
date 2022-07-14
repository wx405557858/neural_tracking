import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def generate(xx, yy, img_blur=None, rng=0., W=48, H=48, N=6, M=6):
    if img_blur is None:
        img_blur = (np.random.random((15, 15, 3)) * 0.7) + 0.3
        img_blur = cv2.resize(img_blur, (H, W))
    
    img = img_blur + np.random.randn(W, H, 3)*0.05 - 0.025
    for i in range(N):
        for j in range(M):
            r = int(yy[i, j])
            c = int(xx[i, j])
            if r >= W or r < 0 or c >= H or c < 0:
                continue
            
            img[r-1:r+2, c-1:c+2, :] = img_blur[r-1:r+2, c-1:c+2, :] * rng
            img[r, c, :] = img_blur[r, c, :] * (random.random() * 0.)
            
#             for channel in range(3):
#                 img[r-2:r+3, c-2:c+3, channel] = img_blur[r-2:r+3, c-2:c+3, channel] * random.random()*0.2
            
#             for pi in range(-1,2):
#                 for pj in range(-1,2):
#                     if r + pi<0 or r + pi >= W or c + pj <0 or c + pj >= H:
#                         continue
#                     if random.random()<0.7:
#                         img[r+pi,c+pj,:] = random.random()*0.2
#             img[r-1:r+1, c-1:c+1, :] = 0.
    img = cv2.GaussianBlur(img, (3,3), 0)
    img[img<0]=0.
    img[img>1]=1.
    return img


def shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy):
    g = np.exp(-( ((xx-center_x)**2 + (yy-center_y)**2)) / ( 2.0 * sigma**2 ) ) 
    xx_ = xx + shear_x * g
    yy_ = yy + shear_y * g
    return xx_, yy_
    
    
def twist(center_x, center_y, sigma, theta, xx, yy):
    g = np.exp(-( ((xx-center_x)**2 + (yy-center_y)**2)) / ( 2.0 * sigma**2 ) ) 
    dx = (xx - center_x)
    dy = (yy - center_y)
    
    rotx = dx * np.cos(theta) - dy * np.sin(theta)
    roty = dx * np.sin(theta) + dy * np.cos(theta)
    
    xx_ = xx + (rotx - dx) * g 
    yy_ = yy + (roty - dy) * g
    return xx_, yy_


def random_shear(xx, yy):
    shear_ratio = 3
    center_x = random.random() * W
    center_y = random.random() * H
    sigma = random.random() * W / 2
    shear_x = random.random() * interval * shear_ratio - interval * shear_ratio / 2
    shear_y = random.random() * interval * shear_ratio - interval * shear_ratio / 2
    
    xx_, yy_ = shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy)
    return xx_, yy_
    
    
def random_twist(xx, yy):
    twist_degree = 100
    center_x = random.random() * W
    center_y = random.random() * H
    sigma = random.random() * W / 2
    theta = (random.random()*twist_degree - twist_degree/2.) / 180.0 * np.pi
    
    xx_, yy_ = twist(center_x, center_y, sigma, theta, xx, yy)
    return xx_, yy_




def preprocessing(img):
#     Brightness
    ret = img
    
    sz = int(3 + random.random() * 15)
    x = int(random.random() * (W - sz))
    y = int(random.random() * (H  - sz))
    
    
    ret = ret * (0.9+random.random()*0.2)
    
    return ret

def generate_img(batch_size=32, setting=None):
    
    while True:

        X, Y = [], []
        
#         if np.random.random() < 0:
#             W, H = 48, 48
#             N, M = 6, 6
#         else:
#             W, H = 48, 64
#             N, M = 6, 8
            
        N, M = np.random.randint(4,15), np.random.randint(4,15)
        W = np.random.randint(N * 6, 96)
        H = np.random.randint(M * 6, 96)
        W = (W//16+1)*16
        H = (H//16+1)*16
        
        if not(setting is None):
            W, H, N, M = setting
        
        
        x = np.arange(0, W, 1)
        y = np.arange(0, H, 1)
        xx, yy = np.meshgrid(y, x)
        
        
        interval_x = W/(N+1)
        interval_y = H/(M+1)

        x = np.arange(interval_x, W, interval_x)[:N]
        y = np.arange(interval_y, H, interval_y)[:M]
        xind, yind = np.meshgrid(x, y)
        xind = (xind.reshape([1,-1])[0]).astype(np.int)
        yind = (yind.reshape([1,-1])[0]).astype(np.int)
        xind += (np.random.random(xind.shape)*4-2).astype(np.int)
        yind += (np.random.random(xind.shape)*4-2).astype(np.int)
        
        for i in range(batch_size):
            rng = np.random.random()

            img_blur = (np.random.random((15, 15, 3)) * 0.7) + 0.3
            img_blur = cv2.resize(img_blur, (W, H))



        #     Random markers
#             xind = (np.random.random(N*M) * W).astype(np.int)
#             yind = (np.random.random(N*M) * H).astype(np.int)

        #     Grid markers
#             x_grid = np.arange(0, N, 1) * interval + padding
#             y_grid = np.arange(0, M, 1) * interval + padding
#             xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

#             xind, yind = np.reshape(xx_grid,[-1]), np.reshape(yy_grid,[-1])
#             xind += (np.random.random(xind.shape)*4-2).astype(np.int)
#             yind += (np.random.random(xind.shape)*4-2).astype(np.int)


            xx_marker, yy_marker = xx[xind,yind].reshape([N,M]), yy[xind, yind].reshape([N,M])
            img0 = generate(xx_marker, yy_marker, img_blur=None, rng=rng, W=W, H=H, N=N, M=M)

        #     Random distortion
            xx_, yy_ = xx, yy
            xx_, yy_ = random_shear(xx_, yy_)
            xx_, yy_ = random_twist(xx_, yy_)


        #     Distorted markers
            xx_marker_, yy_marker_ = xx_[xind,yind].reshape([N,M]), yy_[xind, yind].reshape([N,M])
            img = generate(xx_marker_, yy_marker_, img_blur=None, rng=rng, W=W, H=H, N=N, M=M)

            t = np.zeros([xx_.shape[0], xx_.shape[1], 2])
            t[:,:,0] = xx_ - xx
            t[:,:,1] = yy_ - yy

            X.append(np.dstack([img0-0.5, img-0.5, np.reshape(xx,[W,H,1]), np.reshape(yy,[W,H,1])]))
            Y.append(t)

        X = np.array(X)
        Y = np.array(Y)

#         X = X[:,:W,:H] - 0.5
#         print(X.shape, xx.shape, yy.shape)
#         X = np.dstack([X])
        Y = Y[:,:W,:H]
        
        Y_list = [Y, Y[:,1::2,1::2], Y[:,2::4,2::4], Y[:, 4::8, 4::8], Y[:, 8::16, 8::16]]


        yield X, Y_list

    

# for i in range(500000):
# for i in range(100000):
#     if i % 10000 == 0: print(i)
#     xx_, yy_ = xx, yy
#     xx_, yy_ = random_shear(xx_, yy_)
#     xx_, yy_ = random_twist(xx_, yy_)
    
# #     xx_, yy_ = twist(21, 21, 3, -40/180.*np.pi, xx_, yy_)
#     img = generate(xx_, yy_)
#     X.append(img)
    
#     t = np.zeros([xx_.shape[0], xx_.shape[1], 2])
#     t[:,:,0] = xx_.astype(np.int)
#     t[:,:,1] = yy_.astype(np.int)
#     Y.append(t)
    
