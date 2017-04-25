import math
import numpy as np
import matplotlib.pyplot as plt
import sys


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/plot_magdvelocity_time.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-03-27-21-20/
#python .py output-velocity time




#Path file 
path = sys.argv[1]



# Read file col  
def plot_dm_velocity(file, Ra,path):
    
    file1  = open(file, 'r')
    lines = file1.readlines()


    # Get Vectors


    t = []
    dmt = []
    
    
    for i in range (2,500):
        
        
        time = str(i)
        time = time+'   '
        for line in lines:
            if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
                t.append(i)
                line = line.split()
                
                v = [float(line[5]), float(line[6]), float(line[7])]
                v = np.asarray(v)
                
                v1 = [float(line[9]), float(line[10]), float(line[11])]
                v1 = np.asarray(v1)
        
                u = v - v1 
            
                d = np.linalg.norm(u)
            
                dmt.append(d)

        

        
    #print t
    #print dmt
                 
    #Plot
    q = plt.plot(t,dmt,'m.', markersize=1)

    plt.xlabel('Time')
    plt.ylabel('Mag Diff')
    plt.title('Time vs Vel Mag Diff; Radius: %s' %(Ra))
    plt.axis([0, 500, 0, 0.25])
    #plt.show()
    Ra2 = str(Ra)
    Ra2 = Ra2[0]+Ra2[2]
    plt.savefig('%sMDV_Time/MDVt_R_%s.png' %(path,Ra2))
    plt.close()




    

c = 0.5

while c <= 2.5 :
    
    Ra = round(c, 1)
    file_velocity = '%sVelocity_%s.csv' %(path,Ra)
    plot_dm_velocity(file_velocity,Ra,path)
    c = c + 0.5
    print c

    
    
    
    
    