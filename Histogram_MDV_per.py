import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import powerlaw 


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/Histogram_MDV_per.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/

#Path file 
path = sys.argv[1]



def hist_mdv(file, Ra, time, path, Per):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    mdv = []
    dc = []

    for line in lines:
        if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
            
            line = line.split()
            
            pos = [float(line[2]), float(line[3]), float(line[4])]
            pos = np.asarray(pos)
            distc = np.linalg.norm(pos)
            dc.append(distc)
            
            v = [float(line[5]), float(line[6]), float(line[7])]
            v = np.asarray(v)
            v1 = [float(line[9]), float(line[10]), float(line[11])]
            v1 = np.asarray(v1)
        
            u = v - v1 
            
            uu = np.linalg.norm(u)
            
            mdv.append(uu)
           
    
    #Split Data, over PER dist center, or under 
    dc = np.asarray(dc)
    a = max(dc)
    l = len(dc)
    dc = dc/a
    
    
    dc1 = []
    mdv1 = []
    dc2 = []
    mdv2 = []
    
    for i in range (0,l):
        if dc[i] < Per: 
            dc1.append(dc[i])
            mdv1.append(mdv[i])
        
        else: 
            dc2.append(dc[i])
            mdv2.append(mdv[i])
    
    
        
    
    #Plot hist dc < PER
    plt.hist(mdv1, bins=100, log = True, color='g')
    plt.title('MDV Histogram; Time: %s  Radius: %s dc < %s' %(time, Ra, Per))
    plt.xlabel("MDV")
    plt.ylabel("Frecuency")
    plt.axis([0, 0.25, 0.9, 10000])
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sHistogram_MDV/Histogram_R_%s_t_%s_Per1.png' %(path,Ra2,t))
    plt.close()
    
    
    #Plot hist dc > PER
    plt.hist(mdv2, bins=100, log = True, color='g')
    plt.title('MDV Histogram; Time: %s  Radius: %s dc > %s' %(time, Ra, Per))
    plt.xlabel("MDV")
    plt.ylabel("Frecuency")
    plt.axis([0, 0.25, 0.9, 10000])
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sHistogram_MDV/Histogram_R_%s_t_%s_Per2.png' %(path,Ra2,t))
    plt.close()



for i in range (0,20):
    
    time = str(i*5+700)
    time = time+'   '
    print time

    c = 5

    while c <= 6 :
    
        Ra = round(c, 1)
        file_velocity = '%sVelocity_%s.txt' %(path,Ra)
        hist_mdv(file_velocity,Ra, time, path, 0.81 )
        c = c+1

    
    
