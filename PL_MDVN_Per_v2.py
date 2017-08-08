import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt
from sklearn import linear_model


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/PL_MDVN_Per_v2.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/

#Path file 
path = sys.argv[1]



def PL_mdvn(file, Ra, time, path, Per):
    #Read file col
    file1  = open(file, 'r')
    lines = file1.readlines()
    mvn = []
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
            
            vv1 = np.linalg.norm(v1)
            uu = np.linalg.norm(u)
            mdvn = uu/vv1
            mvn.append(mdvn)
    
    mvn = np.asarray(mvn)
    n = len(mvn)
     
    #Split Data, over PER dist center, or under 
    dc = np.asarray(dc)
    a = max(dc)
    l = len(dc)
    dc = dc/a

    n = len(mvn)
    f = []


    dc1 = []
    mdvn1 = []
    dc2 = []
    mdvn2 = []
    
    for i in range (0,l):
        if dc[i] < Per: 
            dc1.append(dc[i])
            mdvn1.append(mvn[i])
        
        else: 
            dc2.append(dc[i])
            mdvn2.append(mvn[i])
            
            
    mdvn1 = np.asarray(mdvn1)
    n1 = len(mdvn1)
    
    f1 = []
    f2 = []
    
    mdvn2 = np.asarray(mdvn2)
    n2 = len(mdvn2)  
    
    for i in range(0, 20):
        f1.append(float(i)/5)
        f2.append(float(i)/25)
    
    con = np.zeros(20)
    con1 = np.zeros(20)
    con2 = np.zeros(20)
    # pl1l = np.zeros(20)
    # pll = np.zeros(20)
    
    for i in range(0,20):
        for j in range(0,n):
            if mvn[j] > f1[i]:
                con[i] = con[i]+1
                
    for i in range(0,20):
        for j in range(0,n1):
            if mdvn1[j] > f1[i]:
                con1[i] = con1[i]+1
                
    for i in range(0,20):
        for j in range(0,n2):
            if mdvn2[j] > f2[i]:
                con2[i] = con2[i]+1

    
    return f1, f2, con, con1, con2






for i in range (0,20):
        time = str(i*5+700)
        time = time+'   '
        print time

        c = 5

        while c <= 6 :
    
            Ra = round(c, 1)
            file_velocity = '%sVelocity_%s.txt' %(path,Ra)
            f1, f2, con, con1, con2 = PL_mdvn(file_velocity,Ra, time, path, 0.81)
            
            plt.figure(1)
            plt.subplot(221)
            plt.plot(f1, con, marker='.', linestyle='--', label = '%s'%(i))
            plt.xscale('log')
            plt.yscale('log')
            plt.title('PL-all')

            plt.subplot(222)
            plt.plot(f1, con1, marker='.', linestyle='--', label = '%s'%(i))
            plt.xscale('log')
            plt.yscale('log')
            plt.title('PL- -81')
            
            plt.subplot(223)
            plt.plot(f2, con2, marker='.', linestyle='--', label = '%s'%(i))
            plt.xscale('log')
            plt.yscale('log')
            plt.title('PL- +81')
            

            c = c+1
            

plt.show()



    
