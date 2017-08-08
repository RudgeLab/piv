import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt
from sklearn import linear_model


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/PL_MDVN_all_times_v2.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/

#Path file 
path = sys.argv[1]



def plot_pl(file, path):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    
    
    f = []
    
    for i in range(0, 20000):
        f.append(float(i/20))
    
    fc = f[1:]
    
    
    for line in lines:
        if line[0] != 'T':
            line = line.split()
            m = float(line[2])
            print m 
            b = float(line[3])
            print b
            print line[1]
            if line[1]=='5.0': 
                plt.plot(fc, np.power(fc, m) * np.power(10,b), color='green',linewidth=2)
            
            else: 
                plt.plot(fc, np.power(fc, m) * np.power(10,b), color='blue',linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('PL-MDVN ')
    plt.xlabel('log(MDVN)')
    plt.ylabel('log(N(MDVN))') 
    
    plt.savefig('%sPL_MDV/PL_all.png' %(path))
    plt.close()
            # m1 = float(line[4])
            # print m1
            # b1 = float(line[5])
            # print b1
            # plt.plot(fc, np.power(fc, m1) * np.power(10,b1), color='blue',linewidth=2)
            #
            # m2 = float(line[6])
            # print m2
            # b2 = float(line[7])
            # print b2
            # plt.plot(fc, np.power(fc, m2) * np.power(10,b2), color='red',linewidth=2)
            
            
     
            
    #plt.axis([0.0001, 100, 0.0001, 100])        






file_velocity = '%sPLN_slope_N.txt' %(path)
plot_pl(file_velocity, path)



    
