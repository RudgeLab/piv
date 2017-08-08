import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/mdv_dc_t_ana.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/
#python .py output-velocity




#Path file 
path = sys.argv[1]


def analyze_file(file, Ra, path):
    
    Ra = str(Ra)
    
    #Read file output from plot_mdv_dc_t
    file1  = open(file, 'r')
    lines = file1.readlines()
    
    
    #Get vectors (slope and intercept)
    time = []
    m = []
    n = []
    m1 = []
    m2 = []
    n1 = []
    n2 = []

    for line in lines:
        
        line = line.split()
        
        if line[1] == Ra:
            time.append(line[0])
            m.append(line[2])
            n.append(line[3])
            m1.append(line[4])            
            n1.append(line[5])
            m2.append(line[6])
            n2.append(line[7])
    
    
    #Plot al slopes and intercepts 
    plt.plot(time, m, 'go',label = 'S')
    plt.plot(time, m1, 'bo', label = 'S1')
    plt.plot(time, m2, 'ko', label = 'S2')
    plt.plot(time, n, 'g.', label = 'I')
    plt.plot(time, n1, 'b.', label = 'I1')
    plt.plot(time, n2, 'k.', label = 'I2')
    
    plt.legend()
    plt.title('Regression  Radius: %s' %(Ra))
    # #
    # plt.axis([0, 1, 0, 0.25])
    # Ra2 = str(Ra)
    Ra = Ra[0]+Ra[2]
    # t = time[:3].replace(' ', '')
    plt.savefig('%sMDV_DistCentre/MDVd_R_%s_ana.png' %(path,Ra))
    plt.close()



c = 5.0

while c <= 6 :

    Ra = round(c, 1)
    file_velocity = '%sDMV_distcenter_081.txt' %(path)
    analyze_file(file_velocity,Ra,path)
    c = c+1.0
           

    
    
    
    