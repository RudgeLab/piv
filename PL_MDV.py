import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/PL_MDV.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/

#Path file 
path = sys.argv[1]



def pl_mdv(file, Ra, time, path):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    mv = []
    
    for line in lines:
        if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
            
            line = line.split()
            
            v = [float(line[5]), float(line[6]), float(line[7])]
            v = np.asarray(v)
            v1 = [float(line[9]), float(line[10]), float(line[11])]
            v1 = np.asarray(v1)
        
            u = v - v1 
            
            uu = np.linalg.norm(u)
            
            mv.append(uu)
            
    mv = np.asarray(mv)
    n = len(mv)
    f = []

    for i in range(0, 20):
        f.append(float(i)/1000)
        
    con = np.zeros(20)
    
    for i in range(0,20):
        for j in range(0,n):
            
            if mv[j] > f[i]:
                con[i] = con[i]+1

    
    
    plt.plot(f, con, 'bo')
    plt.xscale('log')
    plt.yscale('log')
    # plt.title('MDV Histogram; Time: %s  Radius: %s' %(time, Ra))
 #    plt.xlabel("MDV")
 #    plt.ylabel("Frecuency")
 #    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sPL_MDV/PL_MDV_R_%s_t_%s.png' %(path,Ra2,t))
    plt.close()



for i in range (0,20):
    

    time = str(i*5+700)
    time = time+'   '
    print time

    c = 5

    while c <= 6 :
    
        Ra = round(c, 1)
        file_velocity = '%sVelocity_%s.txt' %(path,Ra)
        pl_mdv(file_velocity,Ra, time, path)
        c = c+1
    
    
    
