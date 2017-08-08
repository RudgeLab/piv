import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt
import powerlaw


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/Histogram_MDV.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/

#Path file 
path = sys.argv[1]



def hist_mdv(file, Ra, time, path):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    mv = []
    
    #Read file, output from Velocity_all get velocity of cell and neighbourhood, calculate MDV 
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
            
    
    mdv = np.asarray(mv)
    mdvl = np.log(mdv)

    # Plot log
    plt.hist(mdvl, bins=50, log = True, color='g')
    plt.title('MDV Histogram; Time: %s  Radius: %s' %(time, Ra))
    plt.xlabel("log(MDV)")
    plt.ylabel("Frecuency")
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sHistogram_MDV/Histogram_R_%s_t_%s_log.png' %(path,Ra2,t))
    plt.close()
    
    # Plot 
    plt.hist(mdv, bins=50, log = True, color='b')
    plt.title('MDV Histogram; Time: %s  Radius: %s' %(time, Ra))
    plt.xlabel("MDV")
    plt.ylabel("Frecuency")
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sHistogram_MDV/Histogram_R_%s_t_%s.png' %(path,Ra2,t))
    plt.close()



for i in range (0,20):
    
        time = str(i*5+700)
        time = time+'   '
        print time
    
        c = 5
    
        while c <= 6 :
        
            Ra = round(c, 1)
            file_velocity = '%sVelocity_%s.txt' %(path,Ra)
            hist_mdv(file_velocity,Ra, time, path)
            c = c+1
    
    
    
