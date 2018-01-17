import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def entropy(hgram, ax):
    '''
    Entropy H(X) of one variable given joint histogram

    hgram = joint histogram (2d array)

    ax = axis over which to sum histogram to compute marginal distribution

    returns: entropy value
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=ax) # marginal for x over y
    nzs = px > 0 # Only non-zero pxy values contribute to the sum
    ex = -np.sum(px[nzs] * np.log2(px[nzs]))
    return ex

def mutual_information(hgram):
    '''
    Mutual information I(X,Y) for joint histogram

    hgram = joint histogram (2d array) 
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    
def joint_entropy(hgram):
    '''
    Joint entropy H(X,Y) for joint histogram

    hgram = joint histogram (2d array) 

    returns: joint entropy value
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    #print pxy[nzs]
    return -np.sum(pxy[nzs] * np.log2(pxy[nzs]))

def conditional_entropy(hgram, ax):
    '''
    Conditional entropy H(Y|X) for joint histogram

    hgram = joint histogram (2d array) 

    ax = axis over which to sum to compute marginal distribution of X

    returns: joint entropy value
    '''
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=ax) # marginal for x over y
    je = joint_entropy(hgram)
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = px > 0 # Only non-zero pxy values contribute to the sum
    ex = -np.sum(px[nzs] * np.log2(px[nzs]))
    return je - ex 



def mutual_info_offset(im1,im2, w,h, vx_max, vy_max, px,py, nbins)
    '''
    Compute mutual information between regions in image pairs over a range of
    offsets. This gives an estimate of the structural similarity between the
    image pair in the specified region. 

    im1,im2 = image pair, numpy float arrays with same shape

    w,h = dimensions of image region are 2w+1, 2h+1

    vx_max,vy_max = tuple giving the maximum offset in each image dimension, in each
    direction

    px,py = position of centre

    nbins = number of bins to use for image histograms

    returns: float array of mutual information at offsets in [-vx_max:vx_max,
    -vy_max,vy_max]
    '''

    vw = vmax[0]*2 + 1
    vh = vmax[1]*2 + 1
    hy = np.zeros((vw,vh))
    mi = np.zeros((vw,vh))
    im1_roi = im1[px-w:px+w, py-h:py+h]
    for vx in range(-vx_max,vx_max+1):
        for vy in range(-vx_max,vy_max+1):
            im2_roi = im2[px+vx-w:px+w+vx, py+vy-h:py+h+vy]
            hgram, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im2_roi.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,2**16),(0,2**16)])
            mutinf = mutual_information(hgram)
            mi[vx+vx_max,vy+vy_max] = mutinf
    return mi

def find_peak(arr):
    '''
    Find location (x,y) of peak in values of array arr

    returns: (x,y) position of peak
    '''
    pk = arr>0.99*np.max(arr,axis=(0,1))
    pkx,pky = np.meshgrid(np.arange(-15,16), np.arange(-15,16))
    x = np.sum(zx*pk)/np.sum(pk)
    y = np.sum(zy*pk)/np.sum(pk)
    return x,y

# Compute conditional entropy at different uniform velocities
def analyze_iterative(im1,im2, vmax, p0, ns, ofname=None):
    hy = np.zeros((ns,31,31))
    mi = np.zeros((ns,31,31))
    z = np.zeros((ns,2))
    #dI = np.zeros((ns,1))
    
    px2,py2 = px,py

    for s in range(ns):
        gs = 2**(5-s)
        cim1 = im1[px-gs:px+gs, py-gs:py+gs]
        for vx in range(-15,16):
            for vy in range(-15,16):
                #print gi,gj,vx,vy
                # Compute conditional entropy
                cim2 = im2[px2+vx-gs:px2+gs+vx, py2+vy-gs:py2+gs+vy]
                hgram, xedges, yedges = np.histogram2d(cim1.ravel(), cim2.ravel(), bins=256, range=[(0,2**16),(0,2**16)])
                #ce = conditional_entropy(hgram, ax=1)
                #hv[s,vx+15,vy+15] = -ce
                mutinf = mutual_information(hgram)
                mi[s,vx+15,vy+15] = mutinf
        #Find peak and compute mean velocity estimate
        ehv = np.exp2(mi[s,:,:])
        pk = ehv>0.99*np.max(ehv,axis=(0,1))
        zx,zy = np.meshgrid(np.arange(-15,16), np.arange(-15,16))
        zxpk = np.sum(zx*pk)/np.sum(pk)
        zypk = np.sum(zy*pk)/np.sum(pk)
        z[s,:] = [zxpk,zypk]
        dcim2 = im2[px2-gs+zypk:px2+gs+zypk, py2-gs+zxpk:py2+gs+zxpk]
        #dI[s] = np.mean(dcim2-cim1, axis=(0,1))

        hgram, xedges, yedges = np.histogram2d(cim1.ravel(), dcim2.ravel(), bins=256, range=[(0,2**16),(0,2**16)])
        hy = entropy(hgram, ax=0)
        print hy
        
        #mutinf = mutual_information(hgram)
        mutinf = mutual_information(hgram)/entropy(hgram, ax=0)
        print mutinf
        
        print z[s,:]
        print gs
        print px2,py2
#        plt.imshow(mi[s,:,:])

        '''
        plt.subplot(241)
        cim2 = im2[px2-gs:px2+gs, py2-gs:py2+gs]
        diffim = cim2-cim1
        clim = [np.min(diffim), np.max(diffim)]
        plt.imshow(cim2-cim1, clim=clim)
        plt.colorbar()
        
        plt.subplot(242)
        plt.imshow(dcim2-cim1, clim=clim)
        plt.colorbar()        
        
        plt.subplot(243)
        plt.imshow(mi[s,:,:])
        plt.colorbar()
        
        plt.subplot(244)
        plt.hist((dcim2).ravel(), bins=20)

        plt.subplot(245)
        plt.imshow(cim1)
        plt.colorbar()   
        
        plt.subplot(246)
        plt.imshow(cim2)
        plt.colorbar() 
        
        plt.subplot(247)
        plt.imshow(dcim2)
        plt.colorbar() 
        
        plt.subplot(248)
        plt.hist(cim1.ravel(), bins=20)
        '''
        
        px2 = px2 + zypk
        py2 = py2 + zxpk
 
    print 'px2 = ', px2
    print 'py2 = ', py2
    #plt.imsave(ofname, cim1)
    return z,mutinf,hy




def main():
    # Number of scales to descend
    ns = 2
    # Number of time points
    nt = 60
    # Grid dimensions and spacing for regions of interest
    gx,gy = 4,4
    gw,gh = 16,16

    # Top left corner of grid of ROIs
    px0,py0 = 1200,600

    # Load images
    fnamebase = 'weiner-17-12-07-17-56/step-%05d'
    fnamebase = '/Users/timrudge/CavendishMicroscopy/10.01.16/Pos0000/Frame%04dStep%04d'
    #fnamebase = '/Volumes/MICROSCOPYD/Microscopy/09.01.16/Pos0001/Frame%04dStep%04d'

    fname = fnamebase + '.png'
    fname = fnamebase + '.tif'
    im1 = [plt.imread(fname%(2,100+i*2)).astype(np.float32) for i in range(nt)]
    im2 = [plt.imread(fname%(2,102+i*2)).astype(np.float32) for i in range(nt)]

    #im1b = [plt.imread(fname%(0,100+i*2)).astype(np.float32) for i in range(nt)]
    #im2b = [plt.imread(fname%(0,102+i*2)).astype(np.float32) for i in range(nt)]

    w,h = im1[0].shape
    w,h = im1[0].shape


    print w,h

    print np.max(im1), np.min(im1)
    print np.max(im2), np.min(im2)


    # Filter images to remove noise
    from scipy.ndimage.filters import gaussian_filter
    im1 = [gaussian_filter(im1[i],3) for i in range(nt)]
    #im1b = [gaussian_filter(im1b[i],3) for i in range(nt)]
    im2 = [gaussian_filter(im2[i],3) for i in range(nt)]
    #im2b = [gaussian_filter(im2b[i],3) for i in range(nt)]


    # Compute velocity and position of ROIs based on maximum mutual information translation
    pos = np.zeros((gx,gy,nt,2))
    zz = np.zeros((gx,gy,ns,2,nt))
    mutinf = np.zeros((gx,gy,nt))
    entropy = np.zeros((gx,gy,nt))

    # Set initial grid positions
    for ix in range(gx):
        for iy in range(gy):
            pos[ix,iy,0,:] = [px0+ix*gw,py0+iy*gh]
    for i in range(nt-1):
        print '------------ Step %d ---------'%i
        for ix in range(gx):
            for iy in range(gy):
                print ix,iy
                #fig = plt.figure(figsize=(12,3))
                #im1 = plt.imread(fname%(2,100+i*2)).astype(np.float32)
                #im2 = plt.imread(fname%(2,102+i*2)).astype(np.float32)
                ofname = 'grid4_4/dcim2-pos%d_%d_step%04d.tif'%(pos[ix,iy,0,0],pos[ix,iy,0,1],i)
                z,mi,hy = analyze_iterative(im1[i], im2[i], int(pos[ix,iy,i,1]), int(pos[ix,iy,i,0]), ns, ofname)
                print z.shape
                zz[ix,iy,:,:,i] = z
                mutinf[ix,iy,i] = mi
                entropy[ix,iy,i] = hy
                pos[ix,iy,i+1,:] = pos[ix,iy,i,:] + np.sum(zz[ix,iy,:,:,i], axis=0)
        #pos[:,:,i+1,:] = pos[:,:,i,:] + np.sum(zz[:,:,:,:,i], axis=(0,1,2))
        print 'pos[i+1] =', pos[ix,iy,i+1,:]
        #plt.draw()
        #plt.savefig('mi-%04d.pdf'%i)
        #plt.close(fig)

    pos.tofile('pos.np', sep=',')
    zz.tofile('z.np', sep=',')
    mutinf.tofile('mutinf.np', sep=',')
    entropy.tofile('entropy.np', sep=',')

    '''
    plt.ion()
    plt.close('all')
    plt.figure(figsize=(16,16))
    i=9
    im1 = plt.imread(fname%(2,100+i*2)).astype(np.float32)
    im2 = plt.imread(fname%(2,100)).astype(np.float32)
    plt.imshow(im1[500:900,1050:1500]-im2[500:900,1050:1500])

    plt.plot(-1050+pos[:,:,:i,0].reshape(gx*gy,i).T, -500+pos[:,:,:i,1].reshape(gx*gy,i).T, 'k-')
    plt.xlim(0,500)
    plt.ylim(0,500)
    #plt.show()
    print pos[0,5,0,:]

    plt.figure()
    for i in range(1,12):
        for j in range(1,12):
            plt.plot(mutinf[i,j,:])

    '''



main()
