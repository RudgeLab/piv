import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt
from sklearn import linear_model


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/PL_MDV_Per.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/

#Path file 
path = sys.argv[1]



def dmv(file, Ra, time, path, Per):
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
    
           
    mdv = np.asarray(mdv)
    n = len(mdv)
    
    
    f = []


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
            
            
    mdv1 = np.asarray(mdv1)
    n1 = len(mdv1)
    
    f1 = []
    f2 = []
    
    mdv2 = np.asarray(mdv2)
    n2 = len(mdv2)  
    
    for i in range(0, 20):
        f1.append(float(i)/1000)
        f2.append(float(i)/2000)
    
    con = np.zeros(20)
    con1 = np.zeros(20)
    con2 = np.zeros(20)
    
    
    for i in range(0,20):
        for j in range(0,n):
            if mdv[j] > f1[i]:
                con[i] = con[i]+1
                
    for i in range(0,20):
        for j in range(0,n1):
            if mdv1[j] > f1[i]:
                con1[i] = con1[i]+1
                
    for i in range(0,20):
        for j in range(0,n2):
            if mdv2[j] > f2[i]:
                con2[i] = con2[i]+1
    
    f1c = f1[6:]
    conc = con[6:]
    con1c = con1[6:]
    
    
    f1cl = np.log10(f1c)
    pll = np.log10(conc)
    pl1l = np.log10(con1c)
    
    f2c = f2[6:]
    con2c = con2[6:]
    
    f2cl = np.log10(f2c)
    pl2l = np.log10(con2c)
    
    f1cl = np.transpose(np.matrix(f1cl))
    f2cl = np.transpose(np.matrix(f2cl))
    
    
    regr = linear_model.LinearRegression()
    regr.fit(f1cl, pll)
    m = regr.coef_[0]
    b = regr.intercept_
    
    regr1 = linear_model.LinearRegression()
    regr1.fit(f1cl, pl1l)
    m1 = regr1.coef_[0]
    b1 = regr1.intercept_
    
    regr2 = linear_model.LinearRegression()
    regr2.fit(f2cl, pl2l)
    m2 = regr2.coef_[0]
    b2 = regr2.intercept_
    
    
    plt.plot(f1, con, 'g.', f1, con1, 'b.', f2, con2, 'r.')
    plt.plot(f1c, conc, 'go',label='T')
    plt.plot(f1c, con1c, 'bo',label='Per<0.81')
    plt.plot(f2c, con2c, 'ro',label='Per>0.81')
    plt.plot(f1c, np.power(f1c, m) * np.power(10,b), color='green',linewidth=2)
    plt.plot(f1c, np.power(f1c, m1)* np.power(10,b1), color='blue',linewidth=2)
    plt.plot(f2c, np.power(f2c, m2)* np.power(10,b2), color='red',linewidth=2)
    

    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('PL-MDV ; Time: %s  Radius: %s' %(time, Ra))
    
    plt.xlabel("log(MDV)")
    plt.ylabel("log(N(MDV))")
    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sPL_MDV/PL_R_%s_t_%s_per.png' %(path,Ra2,t))
    plt.close()
    
    return m, m1, m2

outfile = '%sPL_MDV.txt' %(path)
print 'Sus datos estan guardados en:', outfile

with open(outfile, 'w') as out_file:
    out_file.write('Time    Radius    M    M-82    M+82 \n')

    for i in range (0,20):
    

        time = str(i*5+700)
        time = time+'   '
        print time

        c = 5

        while c <= 6 :
    
            Ra = round(c, 1)
            file_velocity = '%sVelocity_%s.txt' %(path,Ra)
            m, m1, m2 = dmv(file_velocity,Ra, time, path, 0.81)
            
            out_file.write(str(time)+'  '+str(Ra)+'  '+str(m)+'  '+str(m1)+'  '+str(m2)+'\n')
            c = c+1

    out_file.close()
    
