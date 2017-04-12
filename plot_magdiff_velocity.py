import math
import numpy as np
import matplotlib.pyplot as plt
import sys


# RUN: python /Users/MariaGabriela/cellmodeller/Scripts/plot_magdiff_velocity.py /Users/MariaGabriela/cellmodeller/data/ex1_simpleGrowth-17-03-27-21-20/
#python .py output-velocity time




#Path file 
path = sys.argv[1]



# Read file col  
def plot_dm_velocity(file, Ra):
    
    file1  = open(file, 'r')
    lines = file1.readlines()


    # Get Vectors


    t = []
    dmt = []
    
    
    for i in range (2,200):
        
        mc= []
        mr = []
        dm = []
        
        time = str(i)
        time = time+'   '
        for line in lines:
            if line[0]==time[0] and line[1]==time[1] and line[2]== time[2]:
                t.append(i)
                line = line.split()

                m = float(line[8])
                r = float(line[12])
                mc.append(m)
                mr.append(r)

                a = abs(m-r)
               # dm.append(a)
                dmt.append(a)
        
        # n = sum(dm)
        # l = len(dm)
        # p = n/l
        # dmt.append(p)
        
    print t
    print dmt
                 
    #Plot
    q = plt.plot(t,dmt,'m.', markersize=1)

    plt.xlabel('Time')
    plt.ylabel('Mag Diff')
    plt.title('Time vs Vel Mag Diff; Radius: %s' %(Ra))
    #plt.show()
    file2 = file[:-4]
    plt.savefig('%s_magdiffvel.png' %(file2))
    plt.clf()
    plt.show()



    

c = 0.5

while c <= 5 :
    
    Ra = round(c, 1)
    file_velocity = '%sVelocity_%s.csv' %(path,Ra)
    plot_dm_velocity(file_velocity,Ra)
    c = c + 0.5

    
    
    
    
    