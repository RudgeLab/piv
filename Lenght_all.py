import sys
import os
import math
import numpy as np
sys.path.append('.')
import cPickle


#RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/Velocity_all.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/ 5 700 800 5.0 65795


#Path to files
path = sys.argv[1]
#step: from one file to another
step = int(sys.argv[2])-1
#start: in wich file it stars
start = int(sys.argv[3])
#stop: in wich file it ends
stop = int(sys.argv[4])
#Ra = radius
Ra =  float(sys.argv[5])
#Number max cell
Maxcell = int(sys.argv[6])

#Trabajar con diccionarios en vez de arreglos 

def velocity(pickle1, pickle2, maxcell):
    
    #Load both pickes
    data1 = cPickle.load(open(pickle1,'r'))
    data2 = cPickle.load(open(pickle2,'r'))
    
    #Extract position information:
    
    #1. From pickles, data
    cs1 = data1['cellStates']
    cs2 = data2['cellStates']
    
    
    
    it1 = iter(cs1)
    it2 = iter(cs2)
    n1 = len(cs1)
    n2 = len(cs2)
    
    
    print 'Number of cells time t  = '+str(n1)
    print 'Number of cells time t+1 = '+str(n2)
    
    #x1,y1,z1 (poss time 1) x2,y2,z2 (poss time 2)
    poss = np.zeros((maxcell,6))
    
    # x,y,z, m
    velocity = np.zeros((maxcell,4))
    # x, y, z, (x^2+y^2+z^2)^(1/2), vpromradiodef
    
    #2. From the data, get the position of the same cell in each pickle (x1,y1,z1,x2,y2,z2)
    for it1 in cs1:
        p1 = cs1[it1].pos
        poss[it1,0] = p1[0]
        poss[it1,1] = p1[1]
        poss[it1,2] = p1[2]
        
    for it2 in cs2:
        p2 = cs2[it2].pos
        poss[it2,3] = p2[0]
        poss[it2,4] = p2[1]
        poss[it2,5] = p2[2]
        
        
    # Calculate velocity by comparing the cell's possition in time 1 vs in time 2
    for i in range (0,maxcell):
        if (poss[i,3] != 0 or poss[i,4] != 0 or poss[i,5] != 0) and (poss[i,0] != 0 or poss[i,1] != 0 or poss[i,2] != 0):
            
            #Velocity x,y,z
            velocity[i,0] = poss[i,3]-poss[i,0]
            velocity[i,1] = poss[i,4]-poss[i,1]
            velocity[i,2] = poss[i,5]-poss[i,2]
            
            #Velocity modulus
            velocity[i,3] = np.sqrt(velocity[i,0]*velocity[i,0]+velocity[i,1]*velocity[i,1]+velocity[i,2]*velocity[i,2])

            
    return poss,velocity

def average_velocity(poss, velocity, Radio, maxcell):

    # x, y, z, vx,vy,vz,vm, vpx,vpy,vpz, vpm, cos(alpha) (COMPARE)
    compare = np.zeros((maxcell,12))
    cont = 0
    
    #For each of the cells
    for i in range (0,maxcell):
        
        # Iff position in both time is different than 0, the cell exists
        if (poss[i,3] != 0 or poss[i,4] != 0 or poss[i,5] != 0) and (poss[i,0] != 0 or poss[i,1] != 0 or poss[i,2] != 0):
            
            cont = cont +1
            
            #vx,vy,vz,vm
            v_sum = np.zeros(4)
            ca = 0
            
            #Chech the cell's distance to all other cells 
            for j in range (0,maxcell):
                
                #Iff position in both time is different than 0, the cell exists
                if (poss[j,3] != 0 or poss[j,4] != 0 or poss[j,5] != 0) and (poss[j,0] != 0 or poss[j,1] != 0 or poss[j,2] != 0): 
                    
                    #Distance between cells
                    d = np.sqrt((poss[j,0]-poss[i,0])*(poss[j,0]-poss[i,0])+(poss[j,1]-poss[i,1])*(poss[j,1]-poss[i,1])+(poss[j,2]-poss[i,2])*(poss[j,2]-poss[i,2]))
                    
                    #Destance less than radius 
                    if d < Radio:
                        
                        ca = ca +1
                        
                        v_sum[0] = (v_sum[0] + velocity[j,0])
                        v_sum[1] = (v_sum[1] + velocity[j,1])
                        v_sum[2] = (v_sum[2] + velocity[j,2])
                        v_sum[3] = (v_sum[3] + velocity[j,3])
                       
            #Mean 
            v_prom = v_sum/ca
            
            
            compare[i,0] = poss[i,3]
            compare[i,1] = poss[i,4]
            compare[i,2] = poss[i,5]
            
            compare[i,3] = velocity[i,0]
            compare[i,4] = velocity[i,1]
            compare[i,5] = velocity[i,2]
            compare[i,6] = velocity[i,3]
            
            compare[i,7] = v_prom[0]
            compare[i,8] = v_prom[1]
            compare[i,9] = v_prom[2]
            compare[i,10] = v_prom[3]
            
            u = velocity[i,:3]
            v = v_prom[:3]
            
            uu = np.linalg.norm(u)
            vv = np.linalg.norm(v)

            compare[i,11] = np.dot(u,v)/(uu*vv) 
            
                        
    print 'Numero de celulas consideradas:',cont
#    print velocity
    
    return compare
                





outfile = '%sVelocity_%s_%s_%s.txt' %(path,Ra, start, stop)
print 'Sus datos estan guardados en:', outfile

with open(outfile, 'w') as out_file:
    
    out_file.write('Tiempo   Celula     X           Y        Z            VX           VY        VZ         VM        VPX           VPY        VPZ        VPM         Cos(alpha) \n')
        
    contador = start
        
    # Loop Unrolling
    while contador < stop:
        print contador
        a = str(contador).zfill(5)
        contador = contador +1
        b = str(contador).zfill(5)
        file1 = '%sstep-%s.pickle' %(path,a)
        file2 = '%sstep-%s.pickle' %(path,b)
        

        poss, vel = velocity(file1,file2, Maxcell)
        e = average_velocity(poss,vel, Ra, Maxcell)


        # Guardar en variable local y escribir al final
        for i in range (0,len(e)):
            if e[i,3] != 0:
                #print  t, i,'      ', e[i]
                out_file.write(str(contador-1)+'  '+str(i)+'  '+str(e[i,0])+'  '+str(e[i,1])+'  '+str(e[i,2])+'  '+str(e[i,3])+'  '+str(e[i,4])+'  '+str(e[i,5])+'  '+str(e[i,6])+'  '+str(e[i,7])+'  '+str(e[i,8])+'  '+str(e[i,9])+'  '+str(e[i,10])+'  '+str(e[i,11])+'\n')
        
        contador = contador + step
        
    out_file.close()
    



