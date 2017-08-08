import math
import numpy as np
import matplotlib.pyplot as plt
import sys


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/cell_velocity_vectors.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/ 
#python .py path-output-velocity 




#Path file 
path = sys.argv[1]



# Read file col  
def plot_velocity_vectors(file, time, Ra, path):
    
    file1  = open(file, 'r')
    lines = file1.readlines()


    # Get Vectors
    px = []
    py = []
    vx = []
    vy = []
    vrx = []
    vry = []

    #Read file, output from Velocity_all get velocity of cell and neighbourhood
    for line in lines:
        if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
        
            line = line.split()
        
            px.append(float(line[2]))
            py.append(float(line[3]))
        
            vx.append(float(line[5]))
            vy.append(float(line[6]))
        
            vrx.append(float(line[9]))
            vry.append(float(line[10]))

    #Plot vectors
    q = plt.quiver(px,py,vx,vy,angles='xy',color='r', scale=5)
    p = plt.quiver(px,py,vrx,vry,angles='xy',color='b', scale=5)
    plt.axis([-200, 200, -200, 200])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cell (r) and Neighbourhood (b) Velocity Vector;  Time: %s Radius: %s' %(time,Ra))
    
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    t = time[:3].replace(' ', '')
    file2 = file[:-4]
    plt.savefig('%sVelocityVectors/VV_R_%s_t_%s.png' %(path,Ra2,t))
    plt.clf()
    
    
    
for i in range (0,20):
    
    time = str(i*5 + 700)
    time = time+'   '
    print time
    
    c = 5
    
    while c <= 6 :
        
        Ra = round(c, 1)
        #The output folder VelocityVectors is needed
        file_velocity = '%sVelocity_%s.txt' %(path,Ra)
        plot_velocity_vectors(file_velocity,time,Ra,path)
        c = c+1
    
    
    
    
    