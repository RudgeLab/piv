import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time
import infotheory

gx,gy = 8,8
nt = 4
rw,rh = 64,64

pos = np.fromfile('pos.np', sep=',').reshape((gx*gy,nt,2))
#z = np.fromfile('z.np', sep=',').reshape((gx*gy,60,2,2))
#z = np.sum(z, axis=2)
#mi = np.fromfile('mutinf.np', sep=',').reshape((gx*gy,60))
#hy = np.fromfile('entropy.np', sep=',').reshape((gx*gy,60))
roi = np.fromfile('roi.np').reshape((gx*gy,nt,rw,rh))
print roi.shape

fnamebase = '/Users/timrudge/CavendishMicroscopy/10.01.16/Pos0000/Frame0/Frame%04d_regStep%04d'
fname = fnamebase + '.tif'


plt.figure()
mi = np.zeros((gx*gy,nt-1))
hy = np.zeros((gx*gy,nt-1))
for j in range(gx*gy):
    for i in range(nt-1):
        hgram, xedges, yedges = np.histogram2d( roi[j,i,:,:].ravel(), \
                                                    roi[j,i+1,:,:].ravel(), \
                                                    bins=256, \
                                                    range=[(0,2**16),(0,2**16)])
        hy_val = infotheory.entropy(hgram, ax=0)
        mutinf = infotheory.mutual_information(hgram)
        mi[j,i] = mutinf
        hy[j,i] = hy_val
plt.plot((hy[11,:]-mi[11,:]).transpose(), '.-')
#plt.pause(10)

plt.figure()
for i in range(1):
    plt.imshow(mi.reshape((gx,gy,nt-1))[:,:,i+1])
    plt.colorbar()
    #for j in range(gx*gy):
    #    plt.subplot(6,6,j+1)
    #    plt.imshow(roi[j,i,:,:])
    plt.pause(0.5)
#plt.subplot(122)
#plt.imshow(roi[11,-1,:,:])

#plt.pause(100)

'''
for idx in range(gx*gy):
    dx = pos[idx,:,0] - pos[32,:,0]
    dy = pos[idx,:,1] - pos[32,:,1]
    plt.plot(dx*dx+dy*dy,'.')
plt.pause(100)
'''

plt.figure(figsize=(12,12))
plt.ion()
posx = int(pos[0,0,0])
posy = int(pos[0,0,1])
while(1):
    for j in range(gx*gy):
        plt.clf()
        im1 = plt.imread(fname%(0,100+i*2)).astype(np.float32)
        #plt.hist(roi[j,i,:,:].ravel(), bins=256, range=[0,2**16])

        cx,cy = 400,700
        plt.imshow(im1[cx:cx+800,cy:cy+800])

        for i in range(1,nt):
            posx = pos[:,0:i,0].transpose()
            posy = pos[:,0:i,1].transpose()

            plt.plot(posy-cy,posx-cx,'.-')
            plt.xlim(0,800)
            plt.ylim(0,800)


        plt.show()
        plt.pause(0.1)


