# import shapely as sp
import pyvista as pv
import numpy as np
from shapely import area
np.bool = np.bool_

# Create a PyVista plotter
plotter = pv.Plotter()

# Function to create a rectangle plane and add it to the plot
def create_rectangle(center, normal, width, height, plotter):
    plane = pv.Plane(center=center, direction=normal, i_size=width, j_size=height)
    plotter.add_mesh(plane, color="lightblue")

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
    
    return points

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


# Add objects (rectangles) to the plotter
for name in Objects.keys(): 
    rect = Objects[name]
    create_rectangle(rect["center"], rect["normal"], rect["width"], rect["height"], plotter)

# Sample random points from each rectangle and visualize them
points_per_m2 = 3
for name in Objects.keys(): 
    rect = Objects[name]
    points = sample_points_from_rectangle(rect["center"], rect["normal"], rect["width"], rect["height"], points_per_m2)
    plotter.add_points(points, color="red", point_size=8)


def plot_from_viewpoint(plotter, path):
    camera_focal_point = None
    camera_view_up = (0,0,1)
    for i in range(len(path)):
        cpoint = path[i]
        if(i<=len(path)-2):
            npoint = path[i+1]
            dir = npoint - cpoint
            camera_focal_point = npoint+dir
        plotter.camera_position = [cpoint, camera_focal_point, camera_view_up]
        plotter.show()
        print("i",i)


path = np.array([(0,0,10),(2,0,10),(4,0,10),(6,0,10),(8,0,10),(10,0,10),(12,0,10),(14,0,10),(16,0,10),(18,0,10)])
plot_from_viewpoint(plotter, path)


