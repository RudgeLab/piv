import math
import numpy as np
import matplotlib.pyplot as plt
import sys


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/5_jun/plot_mdv_t.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-05-30-20-45/
#python .py output-velocity time




#Path file 
path = sys.argv[1]


# Read file col  
def plot_mdv(file, Ra,path):
    
    file1  = open(file, 'r')
    lines = file1.readlines()


    # Get Vectors
    t = []
    dmt = []
    
    #TIME 
    for i in range (0,20):
        time1 = str(i*5 + 700)
        time = time1+'   '
        
        #Read file, output from Velocity_all get velocity of cell and neighbourhood and difference
        for line in lines:
            if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
                t.append(time1)
                line = line.split()
                
                v = [float(line[5]), float(line[6]), float(line[7])]
                v = np.asarray(v)
                
                v1 = [float(line[9]), float(line[10]), float(line[11])]
                v1 = np.asarray(v1)
        
                u = v - v1 
                d = np.linalg.norm(u)
                dmt.append(d)
    
    #Plot
    q = plt.plot(t,dmt,'m.', markersize=1)
    plt.xlabel('Time')
    plt.ylabel('MDV')
    plt.title('MDV vs Time; Radius: %s' %(Ra))
    plt.axis([700, 800, 0, 0.35])
    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    plt.savefig('%sMDV_Time/MDVt_R_%s.png' %(path,Ra2))
    plt.close()




    

c = 5.0

while c <= 6 :
    
    print c
    Ra = round(c, 1)
    file_velocity = '%sVelocity_%s.txt' %(path,Ra)
    #The output folder MDV_Time is needed
    plot_mdv(file_velocity,Ra,path)
    c = c + 1.0
    

    
    
    
    
    