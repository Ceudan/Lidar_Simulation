### Setting up environment
conda create --name lidar_simulation python=3.10
conda install -c conda-forge pyvista
conda install pyvista
#### you have to change to older numpy version to prevent errors
 conda install numpy==1.23.1 