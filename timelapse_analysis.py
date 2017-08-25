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
                                                       search_area_size=256, \
                                                       sig2noise_method='peak2peak' )
    x, y = openpiv.process.get_coordinates( image_size=image1.shape, window_size=64, overlap=48 )
    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.5 )
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
            frames[:,:,j,i]  = im[500:884,850:1234]
            cvframes[:,:,j,i]  = 255.0*frames[:,:,j,i]/np.max(frames[:,:,j,i])
            cvframes[:,:,j,i] = cv2.GaussianBlur(cvframes[:,:,j,i],(5,5),0)
    plt.imshow(frames[:,:,0,0])
    plt.figure()
    plt.imshow(frames[:,:,0,-1])
    return (frames, cvframes)

def remapImage(img, u, v, x, y):
    w = img.shape[0]
    h = img.shape[1]

    stepw = 12 #w/x.shape[0]
    steph = 12 #h/x.shape[1]

    xx = np.linspace(0,stepw,w).astype(np.float32)
    yy = np.linspace(0,steph,h).astype(np.float32)
    
    xv,yv = np.meshgrid(xx,yy)
    ufull = cv2.remap(u, xv.astype(np.float32), yv.astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
    vfull = cv2.remap(v, xv.astype(np.float32), yv.astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
    
    xv,yv = np.meshgrid(np.arange(0,w),np.arange(0,h))
    w_img = cv2.remap(img, xv.astype(np.float32)-ufull, yv.astype(np.float32)+vfull[:,-1:], cv2.INTER_LINEAR)
    return(w_img)

plt.ion()

fpath = '/Users/timrudge/CavendishMicroscopy/10.01.16/Pos0000'
fnames = [fpath+'/Frame0/Frame0000Step%04d.tif', fpath+'/Frame0002Step%04d.tif']
nframes = 50
step = 1 
nchannels = 2
(frames, cvframes) = loadImages(fnames, nframes, step, nchannels)

# Compute velocity fields
vel = np.zeros((21,21,nchannels,nframes-1))
svel = np.zeros((21,21,nchannels,nframes-1))
div = np.zeros((21,21,nframes-1))
df = np.zeros((384,384,nchannels,nframes-1))

dt = 2

plt.figure(figsize=(12,8))
for i in range(nframes-dt):
    im1 = cvframes[:,:,0,i]
    im1b = cvframes[:,:,1,i]
    sim1 = cv2.GaussianBlur(im1,(5,5),0)
    sim1b = cv2.GaussianBlur(im1b,(5,5),0)
    im2 = cvframes[:,:,0,i+dt]
    im2b = cvframes[:,:,1,i+dt]
    sim2 = cv2.GaussianBlur(im2,(5,5),0)
    sim2b = cv2.GaussianBlur(im2b,(5,5),0)

    (x,y,u,v) = computeVelocityField(im1,im2)
    vel[:,:,0,i] = u
    vel[:,:,1,i] = v
    svel[:,:,0,i] = cv2.GaussianBlur(u,(15,15),0)
    svel[:,:,1,i] = cv2.GaussianBlur(v,(15,15),0)
    du = np.gradient(svel[:,:,0,i], axis=0)
    dv = np.gradient(svel[:,:,1,i], axis=1)
    div[:,:,i] = (du + dv)/(5.0*dt)
    
    uu = svel[:,:,0,i]
    vv = svel[:,:,1,i] 
    vmag = np.sqrt(uu*uu + vv*vv)
    uu = 2.0*uu/vmag
    vv = 2.0*vv/vmag
    
    sim1 = cv2.GaussianBlur(frames[:,:,0,i],(5,5),0)
    sim1b = cv2.GaussianBlur(frames[:,:,1,i],(5,5),0)
    sim2 = cv2.GaussianBlur(frames[:,:,0,i+dt],(5,5),0)
    sim2b = cv2.GaussianBlur(frames[:,:,1,i+dt],(5,5),0)

    plt.subplot(121)
    plt.imshow(sim1)
    plt.hold(True)
    plt.quiver( x, sim1.shape[0]-y, u, v)
    plt.hold(False)

    plt.subplot(122)
    w_sim1 = remapImage(sim1, u,v,x,y)
    w_sim1b = remapImage(sim1b, u,v,x,y)
    print sim2.shape
    print w_sim1.shape
    df[:,:,0,i] = sim2 - w_sim1
    df[:,:,1,i] = sim2b - w_sim1b
    plt.imshow(df[:,:,0,i])
    #plt.colorbar()

    plt.pause(0.1)

    plt.subplot(121)
    plt.imshow(sim2)
    plt.hold(True)
    plt.quiver( x, sim1.shape[0]-y, u, v )
    plt.hold(False)

    plt.savefig('output_images/frame%04d.png'%i)

    vel.tofile('vel%04d.np'%i)
    svel.tofile('svel%04d.np'%i)
    div.tofile('div%04d.np'%i)
    df.tofile('df%04d.np'%i)

    plt.pause(0.1)

    '''
    plt.subplot(121)
    plt.imshow(w_sim1)
    plt.hold(True)
    plt.quiver( x, im1.shape[0]-y, uu, vv )
    plt.hold(False)
    '''

    plt.pause(0.1)
