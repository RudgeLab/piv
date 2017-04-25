import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/Histogram_MDV_xy.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-03-27-21-20/


#Path file 
path = sys.argv[1]



def dmv(file, Ra, path):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    x = []
    y = []
    z = []
    
    for line in lines:
        if line[0] != 'T':
            line = line.split()
        
            vx = float(line[5])
            vy = float(line[6])
            vz = float(line[7])
        
            vx1 = float(line[9])
            vy1 = float(line[10])
            vz1 = float(line[11])
        
            dx = vx-vx1
            x.append(dx)
            dy = vy-vy1
            y.append(dy)
            dz = vz-vz1
            z.append(dz)
    
    
    plt.hist(x, bins=100, log = True, alpha=0.5, color='g', label='x')
    plt.hist(y, bins=100, log = True, alpha=0.5, color='b', label='y')
    plt.legend(loc='upper right')
    #plt.hist(z, bins=100, color='b')
    plt.title('MDV Histogram; R = %s' %(Ra))
    plt.xlabel("MDV")
    plt.ylabel("Frecuency")
    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    plt.savefig('%sHistogram_MDV/Histogram_xy_R_%s.png' %(path,Ra2))
    plt.close()



c = 0.5

while c <= 2.5 :
    
    Ra = round(c, 1)
    file_velocity = '%sVelocity_%s.csv' %(path,Ra)
    dmv(file_velocity,Ra, path)
    c = c+0.5
    
    
    
