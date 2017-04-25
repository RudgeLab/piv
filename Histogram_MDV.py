import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/Histogram_MDV.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-03-27-21-20/

#Path file 
path = sys.argv[1]



def dmv(file, Ra, path):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    mv = []
    
    for line in lines:
        if line[0] != 'T':
            line = line.split()
            
            v = [float(line[5]), float(line[6]), float(line[7])]
            v = np.asarray(v)
            v1 = [float(line[9]), float(line[10]), float(line[11])]
            v1 = np.asarray(v1)
        
            u = v - v1 
            
            uu = np.linalg.norm(u)
            
            mv.append(uu)
            
    
    
    plt.hist(mv, bins=100, log = True, color='g')
    plt.title('MDV Histogram; R = %s' %(Ra))
    plt.xlabel("MDV")
    plt.ylabel("Frecuency")
    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    plt.savefig('%sHistogram_MDV/Histogram_R_%s.png' %(path,Ra2))
    plt.close()



c = 0.5

while c <= 2.5 :
    
    Ra = round(c, 1)
    file_velocity = '%sVelocity_%s.csv' %(path,Ra)
    dmv(file_velocity,Ra, path)
    c = c+0.5
    
    
    
