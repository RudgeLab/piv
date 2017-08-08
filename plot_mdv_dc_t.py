import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/plot_mdv_dc_t.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/ 081
#python .py output-velocity




#Path file 
path = sys.argv[1]
pt = sys.argv[2]


# Read file col  
def plot_dmx_velocity(file, Ra, time, path, Per):
    
    file1  = open(file, 'r')
    lines = file1.readlines()
    pt = float(Per)/100
    
    # Get Vectors
    dc = []
    mdv = []
    
    #Read file, output from Velocity_all get position and velocity of cell and neighbourhood
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
            d = np.linalg.norm(u)
            mdv.append(d)
    

    
    # Normalise the distance to centre vector dividing all by the largest distance
    a = max(dc)
    dc = np.asarray(dc)
    dc = dc/a
    
    
    #plot data
    plt.plot(dc,mdv, 'c.', markersize=1)
    
    
    #Split Data, over Per% dist center, or under 
    
    dc1 = []
    mdv1 = []
    dc2 = []
    mdv2 = []
    
    l = len(dc)
    for i in range (0,l):
        if dc[i] < pt: 
            dc1.append(dc[i])
            mdv1.append(mdv[i])
        else: 
            dc2.append(dc[i])
            mdv2.append(mdv[i])
    
          
    l1 = len(dc1)
    l2 = len(dc2)
    
    #Linear regression for all the cells, the nearest (Per<), the further (Per>)
    
    #All: 
    dc = np.transpose(np.matrix(dc))
    regr = linear_model.LinearRegression()
    regr.fit(dc, mdv)
    #Plot regretion
    plt.plot(dc, regr.predict(dc), color='green',linewidth=2)

    #Save slope and intercept
    print 'R:'
    m = regr.coef_[0]
    b = regr.intercept_
    print(' y = {0} * x + {1}'.format(m, b))
    print 'Mean squared error: \n', np.mean((regr.predict(dc) - mdv) ** 2)
    
    #Per<
    if l1 > 2:
        dc1 = np.transpose(np.matrix(dc1))
        regr1 = linear_model.LinearRegression()
        regr1.fit(dc1, mdv1)
        #Plot regretion
        plt.plot(dc1, regr1.predict(dc1), color='blue',linewidth=2)
        
        #Save slope and intercept
        print 'R1:'
        m1 = regr1.coef_[0]
        b1 = regr1.intercept_
        print(' y = {0} * x + {1}'.format(m1, b1))
        print 'Mean squared error: \n', np.mean((regr1.predict(dc1) - mdv1) ** 2)
        
    else: 
        print 'No hay datos suficientes en %s Menor ' %(Per)
        m1 = 'a'
        b1 = 'b'

    #Per>
    if l2 > 2:
        dc2 = np.transpose(np.matrix(dc2))
        regr2 = linear_model.LinearRegression()
        regr2.fit(dc2, mdv2)
        #Plot regretion
        plt.plot(dc2, regr2.predict(dc2), color='black',linewidth=2)
        
        #Save slope and intercept
        print 'R2:'
        m2 = regr2.coef_[0]
        b2 = regr2.intercept_
        print(' y = {0} * x + {1}'.format(m2, b2))
        print 'Mean squared error: \n', np.mean((regr2.predict(dc2) - mdv2) ** 2)
        
    else: 
        print 'No hay datos suficientes en %s Mayor ' %(Per)
        m2 = 'a'
        b2 = 'b'
    
    #Plot 
    plt.xlabel('Normalised Distance to Centre')
    plt.ylabel('MDV')
    plt.title('MDV vs Normalised Distance to Centre;  Time: %s  Radius: %s' %(time, Ra))
    plt.axis([0, 1, 0, 0.35])
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    plt.savefig('%sMDV_DistCentre/MDVd_R_%s_t_%s_%s.png' %(path,Ra2,t,Per))
    plt.close()
    
    
    return m, b, m1, b1, m2, b2



outfile = '%sDMV_distcenter_%s.txt' %(path,pt)
print 'Sus datos estan guardados en:', outfile


with open(outfile, 'w') as out_file:
    out_file.write('Tiempo   Radio       M         N      M-80      N-80        M+80        N+80 \n')

    for i in range (0,20):

        time = str(i*5+700)
        time = time+'   '
        print time
        c = 5.0

        while c <= 6:
            Ra = round(c, 1)
            #The output folder  MDV_DistCentre is needed 
            file_velocity = '%sVelocity_%s.txt' %(path,Ra)
            m, b, m1, b1, m2, b2 = plot_dmx_velocity(file_velocity,Ra,time,path, pt)
            c = c+1
            out_file.write(str(time)+'  '+str(Ra)+'  '+str(m)+'  '+str(b)+'  '+str(m1)+'  '+str(b1)+'  '+str(m2)+'  '+str(b2)+'\n')
    
    out_file.close()
    
    
    
    