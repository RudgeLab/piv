import sys
import os
import math
import numpy as np
sys.path.append('.')
import matplotlib.pyplot as plt
from sklearn import linear_model


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun//PL_MDVN_Per_NC.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/

#Path file 
path = sys.argv[1]



def PL_mdvN_N(file, Ra, time, path, Per):
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
    

    conN = con/n
    con1N = con1/n1
    con2N = con2/n2
    
    
    f1c = f1[3:]
    conNc = conN[3:]
    con1Nc = con1N[3:]
    
    
    f1cl = np.log10(f1c) 
    pll = np.log10(conNc)
    pl1l = np.log10(con1Nc)
    
    
    f2c = f2[5:]
    con2Nc = con2N[5:]
    
    
    f2cl = np.log10(f2c)
    pl2l = np.log10(con2Nc)
    
    
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
    
    
    #plt.plot(f1, con, 'g.', f1, con1, 'b.', f2, con2, 'r.')
    plt.plot(f1, conN, 'go',label='T')
    plt.plot(f1, con1N, 'bo',label='Per<0.81')
    plt.plot(f2, con2N, 'ro',label='Per>0.81')
    plt.plot(f1c, np.power(f1c, m) * np.power(10,b), color='green',linewidth=2)
    plt.plot(f1c, np.power(f1c, m1)* np.power(10,b1), color='blue',linewidth=2)
    plt.plot(f2c, np.power(f2c, m2)* np.power(10,b2), color='red',linewidth=2)
    
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('PL-MDVN ; Time: %s  Radius: %s' %(time, Ra))
    
    plt.xlabel("log(MDVN)")
    plt.ylabel("log(N(MDVN))")
    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sPL_MDVN/PLN_R_%s_t_%s_per_N.png' %(path,Ra2,t))
    plt.close()
    
    return m, m1, m2, b, b1, b2



outfile = '%sPLN_slope_N.txt' %(path)
print 'Sus datos estan guardados en:', outfile

with open(outfile, 'w') as out_file:
    out_file.write('Time    Radius    M    N    M-81    N-81    M+81    N+81  \n')

    for i in range (0,20):
            time = str(i*5+700)
            time = time+'   '
            print time
    
            c = 5
    
            while c <= 6 :
        
                Ra = round(c, 1)
                file_velocity = '%sVelocity_%s.txt' %(path,Ra)
                m, m1, m2, b, b1, b2= PL_mdvN_N(file_velocity,Ra, time, path, 0.81)
                
                out_file.write(str(time)+'  '+str(Ra)+'  '+str(m)+'   '+str(b)+'  '+str(m1)+'    '+str(b1)+'   '+str(m2)+'   '+str(b2)+'\n')
                c = c+1
    
    out_file.close()
    
