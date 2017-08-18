import openpiv.tools
import openpiv.process
import openpiv.scaling
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import cv2


def computeVelocityField(image1, image2):
    thframe1 = cv2.adaptiveThreshold(image1.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,0)
    thframe2 = cv2.adaptiveThreshold(image2.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,0)
    u, v, sig2noise = openpiv.process.extended_search_area_piv( thframe1.astype(np.int32), \
                                                       thframe2.astype(np.int32), \
                                                       window_size=64, \
                                                       overlap=48, \
                                                       dt=1, \
                                                       search_area_size=128, \
                                                       sig2noise_method='peak2peak' )
    x, y = openpiv.process.get_coordinates( image_size=image1.shape, window_size=64, overlap=48 )
    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.25 )
    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 1 )
    #openpiv.tools.save(x, y, u, v, mask, )
    return (x,y,u,v)

def loadImages(fnames, nframes, step, nchannels):
    frames = np.zeros((384,384,nchannels,nframes))
    cvframes = np.zeros((384,384,nchannels,nframes))
    for i in range(nframes):
        for j in range(nchannels):
            im = openpiv.tools.imread( fnames[j]%(i*step+100) ).astype(np.float32)
            frames[:,:,j,i]  = im[500:884,1100:1484]
            cvframes[:,:,j,i]  = 255.0*frames[:,:,j,i]/np.max(frames[:,:,j,i])
            cvframes[:,:,j,i] = cv2.GaussianBlur(cvframes[:,:,j,i],(5,5),0)
    plt.imshow(frames[:,:,0,0])
    plt.figure()
    plt.imshow(frames[:,:,0,-1])
    return (frames, cvframes)
    

plt.ion()

fpath = '/Users/timrudge/CavendishMicroscopy/10.01.16/Pos0000'
fnames = [fpath+'/Frame0/Frame0000Step%04d.tif', fpath+'/Frame0002Step%04d.tif']
nframes = 50
step = 1
nchannels = 2
(frames, cvframes) = loadImages(fnames, nframes, step, nchannels)

# Compute velocity fields
vel = np.zeros((21,21,2,nframes-1))
svel = np.zeros((21,21,2,nframes-1))
div = np.zeros((20,20,nframes-1))

dt = 5

plt.figure(figsize=(12,8))
for i in range(nframes-5):
    im1 = cvframes[:,:,0,i]
    im2 = cvframes[:,:,0,i+dt]
    (x,y,u,v) = computeVelocityField(im1,im2)
    vel[:,:,0,i] = u
    vel[:,:,1,i] = v
    svel[:,:,0,i] = cv2.GaussianBlur(u,(15,15),0)
    svel[:,:,1,i] = cv2.GaussianBlur(v,(15,15),0)
    du = np.diff(svel[:,:,0,i], axis=0)
    dv = np.diff(svel[:,:,1,i], axis=1)
    div[:,:,i] = du[:,:-1] + dv[:-1,:]
    
    uu = u
    vv = v
    vmag = np.sqrt(uu*uu + vv*vv)
    uu = 2.0*uu/vmag
    vv = 2.0*vv/vmag
    
#    plt.subplot(121)
    plt.imshow(im1)
    plt.hold(True)
    plt.quiver( x, im1.shape[0]-y, uu, vv)
    plt.hold(False)

    plt.pause(0.1)

#    plt.subplot(122)
    plt.imshow(im2)
    plt.hold(True)
    plt.quiver( x, im1.shape[0]-y, uu, vv )
    plt.hold(False)

    plt.pause(0.1)
