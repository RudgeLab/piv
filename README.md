# piv
Particle image velocimetry for tracking movement and growth of cells.

Cell_velocity.py :From a folder of .pickles from cellmodeller, returns .csv with velocity (x,y,z and magnitude) of each cell and velocity of the neighborhood (x,y,z and magnitude) for a given Radius
run: python path_to_folder_pickles step(ej: 1) stop(ej: 200) Radius

plot_cell_velocity.py: Plots the velocity vector (cell and neighborhood) for each time given.  
run: python plot_cell_velocity.py path-output-velocity 

plot_magdiff_velocity.py: Plots the magnitude of the difference from the cell-neighbothood velocities for each time.  
run: python plot_magdiff_velocity.py path-output-velocity 
