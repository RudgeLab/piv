import math
import numpy as np
import matplotlib.pyplot as plt
import sys


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/plot_magdvelocity_2D_xy.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-03-27-21-20/
#python .py output-velocity




#Path file 
path = sys.argv[1]



# Read file col  
def plot_dmx_velocity(file, Ra,time,path):
    
    file1  = open(file, 'r')
    lines = file1.readlines()


    # Get Vectors
    x1 = []
    y1 = []
    dmt1 = []
    x2 = []
    y2 = []
    dmt2 = []
    x3 = []
    y3 = []
    dmt3 = []
    x4 = []
    y4 = []
    dmt4 = []
    x5 = []
    y5 = []
    dmt5 = []

    
    for line in lines:
        if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
                
            line = line.split()
            
            v = [float(line[5]), float(line[6]), float(line[7])]
            v = np.asarray(v)
            
            v1 = [float(line[9]), float(line[10]), float(line[11])]
            v1 = np.asarray(v1)
    
            u = v - v1 
        
            a = np.linalg.norm(u)
    
            xc = float(line[2])
            yc = float(line[3])
            
            
            if  a <=0.015:
                x1.append(xc)
                y1.append(yc)
                dmt1.append(a)
                
            elif  a <=0.03:
                x2.append(xc)
                y2.append(yc)
                dmt2.append(a)
                
            elif  a <=0.05:
                x3.append(xc)
                y3.append(yc)
                dmt3.append(a)
                
            elif a <= 0.07:
                x4.append(xc)
                y4.append(yc)
                dmt4.append(a)
            else: 
                x5.append(xc)
                y5.append(yc)
                dmt5.append(a)

    
    #Plot
    fig = plt.figure()
    plt.plot(x1, y1, 'bo', ms=1)
    plt.plot(x2, y2, 'go', ms=2)
    plt.plot(x3, y3, 'yo', ms=3)
    plt.plot(x4, y4, 'mo', ms=4)
    plt.plot(x5, y5, 'ro', ms=5)
    
    plt.axis([-100, 100, -100, 100])
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vel Mag Diff vs Position; Time: %s Radius: %s' %(time, Ra))
    
    #plt.show()
    t = time[:3].replace(' ', '')
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    plt.savefig('%sMDV_2D_xy/MDV2D_R_%s_t_%s.png' %(path,Ra2,t))
    plt.close()


for i in range (1,25):
    
    time = str(i*20)
    time = time+'   '
    print time
    
    c = 0.5
    
    while c <= 2.5 :
        
        Ra = round(c, 1)
        file_velocity = '%sVelocity_%s.csv' %(path,Ra)
        plot_dmx_velocity(file_velocity,Ra,time,path)
        c = c+0.5
    
    
    
    
    