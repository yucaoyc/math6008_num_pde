import numpy as np
import meshpy.triangle as triangle
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def check_bc(pt, bc_points, segments):
    """ 
        check if pt lies on the boundary specified by bc_points and segments
    """
    for i in range(len(segments)):
        l_idx, r_idx = segments[i]
        l_pt = bc_points[l_idx]
        r_pt = bc_points[r_idx]
        v1 = pt - l_pt
        v2 = r_pt - pt
        if np.abs(np.dot(v1, v2) - np.linalg.norm(v1) * np.linalg.norm(v2)) < 1.0e-5:
            return True

    return False

def area_of_triangle(pts):
    """
        Compute the area given three vertices.
        using Heron's formula.
    """
    a = np.linalg.norm(pts[1] - pts[0])
    b = np.linalg.norm(pts[2] - pts[1])
    c = np.linalg.norm(pts[0] - pts[2])
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

####################################
# Generate vertices and triangles
# written by LLM
####################################

def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]

def generate_mesh(points, segments, max_volume=0.02, min_angle=20):
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(segments)
    mesh = triangle.build(info, max_volume=max_volume, min_angle=min_angle)
    return mesh

def extract_mesh_data(mesh):
    triangles = np.array(mesh.elements)
    vertices = np.array(mesh.points)
    return vertices, triangles

def plot_mesh(vertices, triangles, figsize=(6, 6)):
    triangulation = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    plt.figure(figsize=figsize)
    plt.triplot(triangulation, 'o-')
    plt.title('2D Mesh for FEM')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.show()
