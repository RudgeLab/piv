import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time

gx,gy = 4,4

pos = np.fromfile('pos.np', sep=',').reshape((gx*gy,60,2))
z = np.fromfile('z.np', sep=',').reshape((gx*gy,60,2,2))
z = np.sum(z, axis=2)
mi = np.fromfile('mutinf.np', sep=',').reshape((gx*gy,60))
hy = np.fromfile('entropy.np', sep=',').reshape((gx*gy,60))

fnamebase = '/Users/timrudge/CavendishMicroscopy/10.01.16/Pos0000/Frame%04dStep%04d'
fname = fnamebase + '.tif'

plt.figure(figsize=(12,12))

plt.plot(hy.T, '.')
plt.show()

plt.figure(figsize=(12,12))
plt.ion()
posx = int(pos[0,0,0])
posy = int(pos[0,0,1])
for i in range(40):
    im1 = plt.imread(fname%(2,100+i*2)).astype(np.float32)
    plt.clf()

    vy = int(z[0,i,0])
    vx = int(z[0,i,1])
    posx += vx
    posy += vy

    print posx, posy
    plt.imshow(im1[posy-16:posy+16,posx-16:posx+16])
    #plt.plot(15,15,'ro')
    #plt.xlim(0,100)
    #plt.ylim(0,100)
    plt.show()
    plt.pause(0.1)
