import numpy as np
from numpy.fft import fft2,ifft2,fftshift
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.signal import correlate2d
import math
import infotheory

plotting = True
save_images = False



def mutual_info_offset(im1,im2, w,h, vx_max, vy_max, px1,py1, px2,py2, nbins):
    '''
    Compute mutual information between regions in image pairs over a range of
    offsets. This gives an estimate of the structural similarity between the
    image pair in the specified region. 

    im1,im2 = image pair, numpy float arrays with same shape

    w,h = dimensions of image region are 2w+1, 2h+1

    vx_max,vy_max = tuple giving the maximum offset in each image dimension, in each
    direction

    px,py = position of centre in im1
    px2,py2 = position of centre in im2

    nbins = number of bins to use for image histograms

    returns: float array of mutual information at offsets in [-vx_max:vx_max,
    -vy_max,vy_max]
    '''


    vw = vx_max*2 + 1
    vh = vy_max*2 + 1

    mi = np.zeros((vw,vh))
    di = np.zeros((vw,vh))
    hy = np.zeros((vw,vh))
    hz = np.zeros((vw,vh))
    im1_roi = im1[px1-w:px1+w+1, py1-h:py1+h+1]
    for vx in range(-vx_max,vx_max+1):
        for vy in range(-vx_max,vy_max+1):
            im2_roi_offset = im2[px2-w+vx:px2+w+vx+1, py2-h+vy:py2+h+vy+1]
            im2_roi = im2[px2-w:px2+w+1, py2-h:py2+h+1]
            hgram_offset, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im2_roi_offset.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,2**16),(0,2**16)])
            hy_val_offset = infotheory.entropy(hgram_offset, ax=0)
            hgram, xedges, yedges = np.histogram2d( im1_roi.ravel(), \
                                                    im2_roi.ravel(), \
                                                    bins=nbins, \
                                                    range=[(0,2**16),(0,2**16)])
            hy_val = infotheory.entropy(hgram, ax=0)
            hy[vx+vx_max,vy+vy_max] = hy_val_offset
            mutinf = infotheory.mutual_information(hgram_offset)
            mi[vx+vx_max,vy+vy_max] = mutinf
            hz[vx+vx_max,vy+vy_max] = mutinf - infotheory.joint_entropy(hgram) + infotheory.joint_entropy(hgram_offset)
            di[vx+vx_max,vy+vy_max] = np.sum((im2_roi-im2_roi.mean())*(im1_roi-im1_roi.mean()), axis=(0,1))
            #np.var(im2_roi-im1_roi, axis=(0,1))# + np.var(im2_roi, axis=(0,1)) + np.var(im1_roi, axis=(0,1))
    return mi,hy,di,hz

def find_peak(arr):
    '''
    Find location (x,y) of peak in values of array arr

    returns: (x,y) position of peak
    '''
    w,h = arr.shape
    ww = (w-1)/2
    hh = (h-1)/2

    pk = arr>0.99*np.max(arr,axis=(0,1))
    pky,pkx = np.meshgrid(np.arange(-ww,ww+1), np.arange(-hh,hh+1))
    x = np.sum(pkx*pk)/np.sum(pk)
    y = np.sum(pky*pk)/np.sum(pk)
    return x,y

# Compute conditional entropy at different uniform velocities
def track_cond_entropy(im1,im2, vx_max,vy_max, px1,py1, gs, nbins=256, ofname=None):
    fullw,fullh = im1.shape
    mi = np.zeros((ns,vx_max*2+1,vy_max*2+1))
    z = np.zeros((ns,2))
    #dI = np.zeros((ns,1))
    
    # Start at same position in 1st and 2nd image
    px2,py2 = px1,py1


    # Compute mutual information between image regions over offset grid
    mi_grid,hy_grid,di_grid,hz_grid = mutual_info_offset(im1,im2, \
                                                            gs,gs, \
                                                            vx_max,vy_max, \
                                                            px1,py1, \
                                                            px2,py2, \
                                                            nbins)
    # Original ROIs
    im1_roi = im1[px1-gs:px1+gs,py1-gs:py1+gs]
    im2_roi = im2[px1-gs:px1+gs,py1-gs:py1+gs]

        if np.max(hy_grid, axis=(0,1))<1.0:
            # Entropy too low, probably background
            zxpk,zypk = 0,0
        else:
            #Find peak in I(X,Y) to compute mean velocity estimate
            zxpk,zypk = find_peak(mi_grid/hy_grid)

        # Estimated velocity
        z[s,:] = [zxpk,zypk]

        # Increment velocity
        px2 = px2 + zxpk
        py2 = py2 + zypk
            
        # Shifted 2nd image ROI
        im2_roi_shifted = im2[px2-gs:px2+gs,py2-gs:py2+gs]

        # Compute mutual info and variance on registered ROI
        mi_grid2,hy_grid2,di_grid2,hz_grid2 = mutual_info_offset(im1,im2, gs,gs, vx_max,vy_max, px1,py1, px2,py2, nbins)

        # Compute some measures on the result
        '''
        diffim = im2_roi_shifted-im1_roi
        plt.hist(diffim.ravel(), bins=20, normed=True)
        mu = np.mean(im2_roi_shifted-im1_roi, axis=(0,1))
        variance = 1.0e3
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x,mlab.normpdf(x, mu, sigma))
        '''
        im2_roi = im2[px2-32:px2+32,py2-32:py2+32]
        im1_roi_crop = np.zeros(im1.shape)
        im1_roi_crop[px1-32:px1+32,py1-32:py1+32] = im1[px1-32:px1+32,py1-32:py1+32]
        im1_roi_crop = im1_roi_crop[px1-32:px1+32,py1-32:py1+32] 

        diffim = im2_roi - im1_roi_crop
        mdiffim = diffim - np.mean(diffim, axis=(0,1))/0.5
        probcell = (1.0/math.sqrt(2.0*math.pi*2048.0)) * np.exp( -mdiffim*mdiffim*0.5/2048.0 )

        #C = correlate2d(im1_roi_crop,im2_roi)
        my,mx = np.meshgrid(np.arange(-32,32), np.arange(-32,32))
        r = np.sqrt(my*my+mx*mx)
        phi = np.abs(np.arctan2(my,mx))
        fim2_roi = fft2(im2_roi-im2_roi.mean(axis=(0,1)))
        fim1_roi_crop = fft2(im1_roi_crop-im1_roi_crop.mean(axis=(0,1)))
        fC = fim1_roi_crop*fim2_roi.conj()
        fC = (fim1_roi_crop-fim2_roi)*(fim1_roi_crop-fim2_roi).conj()
        fC_filt = fC
        fC_filt[np.where(r>500)] = 0
        C = fftshift(ifft2(fC_filt))
        Cxy = mi_grid2[vx_max,vy_max]
        #print 'Cxy = ',Cxy

        # Plot stuff
        if plotting:
            plt.subplot(131)
            #plt.imshow(di_grid)
            plt.cla()
            #p = np.polyfit(((mi_grid2-hy_grid2)).ravel(), np.log2(di_grid2.ravel()), deg=1)
            #Cxy = p[1]
            my,mx = np.meshgrid(np.arange(-vx_max,vx_max+1), np.arange(-vy_max,vy_max+1))
            r = np.sqrt(my*my+mx*mx)
            phi = np.arctan(my,mx)
            idx = np.where(r>=0)
            ploty = hz_grid[idx] #((mi_grid2[idx]-hy_grid2[idx])).ravel()
            plotx = r[idx]
            #plt.plot(plotx, ploty, 'r.')
#            plotx = (r[idx].ravel()) #np.log2(di_grid2.ravel())
            ploty = mi_grid[idx] #(np.absolute(C[32-vx_max:32+vx_max+1,32-vy_max:32+vy_max+1]/(64*64))).ravel()
            #plt.plot(plotx, ploty, 'g.')

            plt.imshow(hz_grid)

            #print 'Slope :', p
            #print 'Cmin = ', p[1]
            #print 'Cmax = ', p[1] - p[0]
            #plt.xlim([-1,0])
            #plt.imshow(mdiffim)
            #plt.ylim([0,8])
            #plt.imshow(im1_roi_crop)
            #if s==0:
            #    plt.colorbar()


        if plotting:
            plt.subplot(133)
            plt.imshow(im1_roi)
            #plt.imshow(im1_roi, interpolation='nearest')
            #plt.imshow(mi_grid-hy_grid)
            #if s==0:
            #    plt.colorbar()
            #plt.subplot(133)
            #plt.imshow(im2_roi_shifted-im1_roi, interpolation='nearest')
            #plt.imshow(di_grid2)
            plt.subplot(132)
            plt.cla()
            #plt.imshow(np.absolute(fftshift(fC)))
            #plotx = r[32-vx_max:32+vx_max+1,32-vy_max:32+vy_max+1]
            plotx = mi_grid-hy_grid
            ploty = (np.absolute(C[32-vx_max:32+vx_max+1,32-vy_max:32+vy_max+1]/(64*64))) # + (mi_grid-hy_grid)*2.7
            #mploty = [np.mean(ploty[np.where(phi==p)]) for p in phi.ravel()]
            #H,xedges,yedges = np.histogram2d(phi.ravel(), ploty.ravel(), bins=(16,32))
            #idx = np.where((r>2)*(r<10))
            #plt.plot(phi[idx], ploty[idx], '.')
            #plt.plot(plotx.ravel(), ploty.ravel(), '.')
            plt.imshow(mi_grid)
            ax = plt.gca()
            ax.set_aspect('auto')
            #plt.plot(plotx.ravel(), ploty.ravel(), '.')
            #plt.ylim([0,20])
            #if s==0:
            #    plt.colorbar()
        #print z[s,:]
        #print gs
        #print px2,py2

        if plotting:
            plt.show()
            plt.pause(0.1)

        if save_images:
            plt.imsave(ofname, im2_roi_shifted)
 
    return px2,py2,Cxy,im2_roi_shifted




def main():
    if plotting:
        plt.ion()
        plt.figure(figsize=(12,4))


    # Load images

    fnamebase = sys.argv[1]
    fname = fnamebase + sys.argv[2]

    startframe = sys.argv[3]
    nframes = sys.argv[4]
    im1 = [plt.imread(fname%(startframe+i*2)).astype(np.float32) for i in range(nt)]
    im2 = [plt.imread(fname%(startframe+2+i*2)).astype(np.float32) for i in range(nt)]

    w,h = im1[0].shape
    w,h = im1[0].shape

    # Grid dimensions and spacing for regions of interest
    gx,gy = 2,2
    gw,gh = w/(gx-1),h/gy

    print "Image dimensions: ",w,h

    print "Image intensity range:"
    print np.max(im1), np.min(im1)
    print np.max(im2), np.min(im2)


    # Filter images to remove noise
    from scipy.ndimage.filters import gaussian_filter
    im1 = [gaussian_filter(im1[i],1) for i in range(nt)]
    im2 = [gaussian_filter(im2[i],1) for i in range(nt)]


    # Compute velocity and position of ROIs based on maximum mutual information translation
    pos = np.zeros((gx,gy,nt,2))
    Cxy = np.zeros((gx,gy,nt))
    roi = np.zeros((gx,gy,nt,64,64))

    # Set initial grid positions
    for ix in range(gx):
        for iy in range(gy):
            pos[ix,iy,0,:] = [px0+ix*gw,py0+iy*gh]
    for i in range(nt-1):
        print '------------ Step %d ---------'%i
        for ix in range(gx):
            for iy in range(gy):
                ofname = 'gridtesting/im2-pos%d_%d_step%04d.tif'%(pos[ix,iy,0,0],pos[ix,iy,0,1],i)
                px = int(pos[ix,iy,i,0])
                py = int(pos[ix,iy,i,1])
                px2,py2,C,im2_roi = analyze_iterative(im1[i], im2[i], \
                                            7, 7, \
                                            px,
                                            py, \
                                            ns, nbins=256, ofname=ofname)
                pos[ix,iy,i+1,:] = [px2,py2]
                Cxy[ix,iy,i+1] = C
                roi[ix,iy,i+1,:,:] = im2_roi
                #plt.ylim([0,30])
                #if i==0:
                #    plt.colorbar()
                print 'vel = ', px2-px, py2-py

        #print 'pos[i+1] =', pos[ix,iy,i+1,:]

    pos.tofile('pos.np', sep=',')
    roi.tofile('roi.np')
    plt.figure()
    plt.plot((Cxy[0,0,1:]))
    plt.pause(20)
    #if plotting:
        #plt.pause(20)


# Run analysis
if __name__ == "__main__": 
    main()
