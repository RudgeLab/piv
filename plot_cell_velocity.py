import math
import numpy as np
import matplotlib.pyplot as plt
import sys


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/plot_velocity.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-03-27-21-20/
#python .py output-velocity time




#Path file 
path = sys.argv[1]



# Read file col  
def plot_velocity(file, time, Ra):
    
    file1  = open(file, 'r')
    lines = file1.readlines()


    # Get Vectors

    px = []
    py = []
    vx = []
    vy = []
    vrx = []
    vry = []


    for line in lines:
        if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
        
            line = line.split()
        
            px.append(float(line[2]))
            py.append(float(line[3]))
        
            vx.append(float(line[5]))
            vy.append(float(line[6]))
        
            vrx.append(float(line[9]))
            vry.append(float(line[10]))

    # Plot
    q = plt.quiver(px,py,vx,vy,angles='xy',color='r')
    p = plt.quiver(px,py,vrx,vry,angles='xy',color='b')
    plt.axis([-25, 25, -25, 25])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Time: %s Radius: %s' %(time,Ra))
    
    t = time[:3].replace(' ', '')
    file2 = file[:-4]
    plt.savefig('%s_t_%s.png' %(file2,t))
    plt.clf()
    
    
    
for i in range (1,20):
    
    time = str(i*10)
    time = time+'   '
    print time
    
    c = 0.5
    
    while c <= 5 :
        
        Ra = round(c, 1)
        file_velocity = '%sVelocity_%s.csv' %(path,Ra)
        plot_velocity(file_velocity,time,Ra)
        c = c+0.5
    
    
    
    
    