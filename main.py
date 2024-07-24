from pymesh import mesh
import numpy as np
from scipy.spatial import ConvexHull

def identify_holes(mesh):
    def is_convex(vertices):
        hull = ConvexHull(vertices)
        return len(hull.simplices) == len(vertices) - 1

    holes = []

    # Iterate over each triangle in the mesh
    for i, triangle in enumerate(mesh.faces):
        triangle_vertices = mesh.vertices[triangle]

        # Check if the triangle is part of a potential hole
        if is_convex(triangle_vertices):
            continue

        # Collect vertices of potential hole
        hole_vertices = set(triangle)
        boundary_edges = set(tuple(sorted((triangle[0], triangle[1]))),
                             tuple(sorted((triangle[1], triangle[2]))),
                             tuple(sorted((triangle[2], triangle[0]))))

        # Explore the potential hole by traversing connected triangles
        while boundary_edges:
            edge = boundary_edges.pop()
            for j, other_triangle in enumerate(mesh.faces):
                if j != i and any(v in edge for v in other_triangle):
                    hole_vertices.update(other_triangle)
                    new_edges = [tuple(sorted((other_triangle[0], other_triangle[1]))),
                                 tuple(sorted((other_triangle[1], other_triangle[2]))),
                                 tuple(sorted((other_triangle[2], other_triangle[0])))]
                    boundary_edges.update(new_edges)

        # Check if the identified vertices form a closed loop
        if len(hole_vertices) > 2 and hole_vertices == set(sum(mesh.faces, [])):
            holes.append(mesh.vertices[list(hole_vertices)])

    return holes

# Load STL file
stl_file = 'your_file.stl'
mesh = Mesh.from_file(stl_file)

# Identify holes
holes = identify_holes(mesh)

# Print information about identified holes
for i, hole_vertices in enumerate(holes):
    hole_area = 0.5 * np.linalg.norm(np.cross(hole_vertices[1] - hole_vertices[0], hole_vertices[2] - hole_vertices[0]))
    hole_centroid = np.mean(hole_vertices, axis=0)
    print(f'Hole {i + 1}:')
    print(f'  Area: {hole_area:.4f} square units')
    print(f'  Centroid: {hole_centroid}')
    print(f'  Vertices: {hole_vertices}')
    print()
