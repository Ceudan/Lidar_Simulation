# import shapely as sp
import stat
from turtle import color
from weakref import ref
import pyvista as pv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
np.random.seed(0)

# Create a PyVista plotter
plotter = pv.Plotter()
# Create a PyVista plotter with off-screen rendering enabled
plotter = pv.Plotter(off_screen=True)

def delete_existing_plots():
    # Use glob to find all files starting with 'plot_time_' and ending with '.png' in the current directory
    files = glob.glob("./plot_time_*.png")
    
    # Loop through the files and remove each one
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

# Function to create a rectangle plane and add it to the plot
def create_rectangle(name,center, normal, width, height, color, plotter):
    plane = pv.Plane(center=center, direction=normal, i_size=width, j_size=height)
    if(name == "floor"):
        plotter.add_mesh(plane, color="gray")
    else:
        plotter.add_mesh(plane, color=color)

# Function to sample random points from within a rectangle
def sample_points_from_rectangle(center, normal, width, height, points_per_m2):
    # Calculate number of points to generate
    area = width * height
    num_points = int(area * points_per_m2)

    # Get two orthogonal vectors (t1, t2) to define the rectangle's local coordinate system
    t1, t2 = orthogonal_vectors(normal)
    
    # Sample random x, y coordinates in the local 2D system (width and height of rectangle)
    t1_lens = np.random.uniform(-height / 2, height / 2, size=num_points)
    t2_lens = np.random.uniform(-width / 2, width / 2, size=num_points)
    
    # Generate points in 3D space by adding scaled t1 and t2 to the center
    points = np.array([center + t1_len * t1 + t2_len * t2 for t1_len, t2_len in zip(t1_lens, t2_lens)])
    
    return points.copy()

# Helper function to calculate two orthogonal vectors (t1, t2) to the normal
def orthogonal_vectors(normal):
    # Assume z-up normal if close to z-axis (normalizing to handle slanted planes)
    normal = np.array(normal) / np.linalg.norm(normal)
    if np.isclose(np.abs(normal[2]), 1.0):  # If normal is close to z-axis
        t1 = np.array([1.0, 0.0, 0.0])  # Choose x-direction
    else:
        t1 = np.cross(normal, [0.0, 0.0, 1.0])
        t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 = t2 / np.linalg.norm(t2)
    return t1, t2

# helper functions for estimating target state
def calculate_angle(points, direction):
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    # Normalize the points (shifted points after subtracting lidar position)
    points_norm = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    # Compute the dot product between each point and the direction vector
    dot_products = np.dot(points_norm, direction)
    # Compute the angle using arccos (element-wise)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Clip to avoid numerical issues
    return angles
# calculate brightness of each point

# def brightness(lidar_position, filtered_points, effective_range, reflectivity_constant, ):
#   #1/d^2 * reflectivity constant
def calculate_intensity(distances, reflectivity, rainfall_intensity):
    # Apply inverse-square law for distance and Beer-Lambert attenuation for fog
    exponent = -0.02*(rainfall_intensity**0.6)*distances
    normalalized_power = (reflectivity/(distances**2))*np.exp(exponent)
    normalized_max_power = 1/(1**2)*np.exp(-0.02*(0**0.6)*1)
    rel_power = normalalized_power/normalized_max_power
    return rel_power

def get_lidar_detections(point_cloud, lidar_position, direction, rainfall_intensity, reflectivity, fov, detection_threshold):
    # calculate which points are in FOV
    shifted_points = point_cloud - lidar_position
    distances = np.linalg.norm(shifted_points, axis=1)
    angles = calculate_angle(shifted_points, direction)
    within_fov = np.abs(angles) <= fov / 2
    point_cloud = point_cloud[within_fov]
    distances = distances[within_fov]
    # filter points by intensity of the return signal
    rel_intensities = calculate_intensity(distances=distances, reflectivity=reflectivity,rainfall_intensity=rainfall_intensity)
    detectable = rel_intensities >= detection_threshold
    point_cloud = point_cloud[detectable]
    return point_cloud.copy()

def plot_path(plotter, path, rainfall_intensity, reflectivity, fov, detection_threshold) -> tuple[float,float]:
    # Sample random points from each rectangle and visualize them
    points_per_m2 = 3

    # Camera position and orientation
    camera_focal_point = (13, 0, 5)
    camera_view_up = (0,0,1)
    camera_pos = [-20, -30, 40]
    # direction robot is facing
    robot_dir = (1,0,0)

    # one coorinate of our statistics plot
    statistic = (-1,-1)

    # Iterate through path
    for i in range(0,len(path)):
        # clear the plot after each timestep
        plotter.clear()
        # Iterate through objects
        for name in Objects.keys():
            # Sample points from object
            rect = Objects[name]
            points = sample_points_from_rectangle(rect["center"], rect["normal"], rect["width"], rect["height"], points_per_m2)
            
            # Get the lidar detections
            if(i+1<len(path)):
                robot_dir = path[i+1] - path[i]
            if(name=="floor"):
                actual_reflectivity = 0.5
            else:
                actual_reflectivity = reflectivity
            filt_points = get_lidar_detections(points.copy(), path[i], robot_dir, rainfall_intensity=rainfall_intensity, reflectivity=actual_reflectivity, fov=fov, detection_threshold=detection_threshold)

            # Get performance statistics
            per_detected = len(filt_points)/len(points)
            if(name in ["house_face_11","house_face_21"]):
                statistic = (reflectivity,per_detected)

            # Plot objects, points and robot position
            if(per_detected>=0.5):
                color = "green"
            else:
                color = "navajowhite"
            create_rectangle(name,rect["center"], rect["normal"], rect["width"], rect["height"],color, plotter)
            if(len(filt_points)>0):
                plotter.add_points(filt_points, color="red", point_size=8)
            plotter.add_points(path[i], color="black", point_size=20)

        # Finalize image and save
        plotter.camera_position = [camera_pos, camera_focal_point, camera_view_up]
        # plotter.show(auto_close=False)
        filename = f'plot_time_{i}.png'
        plotter.screenshot(filename)

    return statistic



"""
World Objects:
    - Below are the shapes that represent our world.
    - All shapes are stored as shapely surface objects (a 2d plane).
    - The coordinates of the world are in the form (x metres, y metres, z metres).
"""
Objects = {}
# The floor of the world
Objects["floor"] = {"center": (13, 0, 0), "normal": (0, 0, 1), "width": 50, "height": 50}
# Walls of House 1
Objects["house_face_11"] = {"center": (10, -5, 3), "normal": (-1, 0, 0), "width": 6, "height": 6}
Objects["house_face_12"] = {"center": (13, -2, 3), "normal": (0, 1, 0), "width": 6, "height": 6}
Objects["house_face_13"] = {"center": (16, -5, 3), "normal": (1, 0, 0), "width": 6, "height": 6}
Objects["house_face_14"] = {"center": (13, -8, 3), "normal": (0, -1, 0), "width": 6, "height": 6}
Objects["house_roof_11"] = {"center": (11.5, -5, 7.5), "normal": (-1, 0, 1), "width": 4.23, "height": 6}
Objects["house_roof_12"] = {"center": (14.5, -5, 7.5), "normal": (1, 0, 1), "width": 4.23, "height": 6}
# Walls of House 2
Objects["house_face_21"] = {"center": (10, 5, 3), "normal": (-1, 0, 0), "width": 6, "height": 6}
Objects["house_face_22"] = {"center": (13, 8, 3), "normal": (0, 1, 0), "width": 6, "height": 6}
Objects["house_face_23"] = {"center": (16, 5, 3), "normal": (1, 0, 0), "width": 6, "height": 6}
Objects["house_face_24"] = {"center": (13, 2, 3), "normal": (0, -1, 0), "width": 6, "height": 6}
Objects["house_roof_21"] = {"center": (11.5, 5, 7.5), "normal": (-1, 0, 1), "width": 4.23, "height": 6}
Objects["house_roof_22"] = {"center": (14.5, 5, 7.5), "normal": (1, 0, 1), "width": 4.23, "height": 6}



# path = np.array([(-10,0,2),(-2,0,2),(6,0,2),(14,0,2),(22,-6,2),(22,-14,2),(18,-18,2),(12.5,-18,2),(12,-17.5,2),(12,-17,2)])
path = np.array([(-4,0,2)])
rainfall_intensity=10
reflectivity=0.5
fov=np.pi/2
detection_threshold=0.0001
statistics = [] # an array of size (num data points, 2), where each data point containts (x,y) for plotting
for reflectivity in np.arange(0.05,0.95,0.5):
    statistic = plot_path(plotter, path, rainfall_intensity=rainfall_intensity, reflectivity=reflectivity, fov=fov, detection_threshold=detection_threshold)
    statistic = np.expand_dims(np.array(statistic),axis=0)
    if(len(statistics)==0):
        statistics = statistic
    else:
        statistics = np.concatenate((statistics,statistic),axis=0)

# plot the 

plt.figure(figsize=(7,4))
plt.plot(statistics[:, 0], statistics[:, 1], marker='o', linestyle='-', color='b', markersize=5)
# plt.xscale('log')
# plt.yscale('log')
plt.title("Backscatter Coefficient vs. Detection Rate")
plt.xlabel("Backscatter Coefficient")
plt.ylabel("Detection Rate")
plt.grid(True)

# Save the plot if save_path is provided
if True:
    save_name = "Backscatter_vs_Detection_Rate.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"Plot saved as {save_name}")
    
plt.show()




