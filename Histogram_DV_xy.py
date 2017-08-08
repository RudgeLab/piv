import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/Histogram_DV_xy.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/


#Path file 
path = sys.argv[1]



def dv_xy(file, Ra, time, path):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    x = []
    y = []
    z = []


    for line in lines:
        if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
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
    plt.title('DVx and DVy Histogram; R = %s, Time = %s' %(Ra, time))
    plt.xlabel("DV")
    plt.ylabel("Frecuency")
    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sHistogram_MDV/Histogram_xy_R_%s_t_%s.png' %(path,Ra2,t))
    plt.close()





for i in range (0,20):
    
    time = str(i*5+700)
    time = time+'   '
    print time
    
    c = 5
    
    while c <= 6 :
        
        Ra = round(c, 1)
        file_velocity = '%sVelocity_%s.txt' %(path,Ra)
        dv_xy(file_velocity,Ra, time, path)
        c = c+1
    
    
