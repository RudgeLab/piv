import sys
import os
import math
import numpy as np
sys.path.append('.')
import cPickle
import CellModeller
import matplotlib.pyplot as plt



file = sys.argv[1] #select pickle file in command line

data = cPickle.load(open(file,'r'))

def rad_pos(cellstate):
    return np.sqrt(cellstate.pos[0]*cellstate.pos[0]+cellstate.pos[1]*cellstate.pos[1])

def area_cell(cellstate):
    return 2.0*cellstate.length*cellstate.radius+np.pi*cellstate.radius*cellstate.radius

def r_delta(rmax,lmax):
    n=int(np.floor(rmax/(6*lmax)))
    rDelta=int(np.floor(rmax/n))
    return rDelta



cs = data['cellStates']

ncells=len(cs)

r=np.zeros((ncells,4))

i=0

for it in cs:
    radialPosition = rad_pos(cs[it])
    area=area_cell(cs[it])
    r[i,0]=radialPosition
    r[i,1]=cs[it].length
    r[i,2]=cs[it].radius
    r[i,3]=area
    i=i+1
    
rsort_poss=np.sort(r,0)
lmax=np.amax(r[:,1])
rmax=int(np.floor(np.amax(r[:,0])))

rDelta=r_delta(rmax,lmax)

for r0 in range(0,rmax-rDelta):
    r1=r0+rDelta
    a_ring=np.pi*(r1*r1-r0*r0)
    a_cell=0.0
    
    j=0
    for i in range(len(rsort_poss[:,0])):
        if float(r0)<=rsort_poss[i,0] and float(r1)>=rsort_poss[i,0]:
            a_cell=a_cell+rsort_poss[i,3]
            j+=1
        elif r1<rsort_poss[i,0]:
            break
    print(r0,r1,a_ring,a_cell,a_cell/a_ring,j)        
  

   

plt.plot(rsort_poss[:,0],rsort_poss[:,1])
plt.show()    
    