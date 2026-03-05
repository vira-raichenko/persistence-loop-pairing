"""Utility functions for reading cycle geometry files and computing barycenters."""


def read_poly_into_list(file):
    """
    Read a .poly file and return a list of 3D vertex coordinates.

    Expected format:
        POINTS
        1: x y z [optional color]
        2: x y z [optional color]
        ...
        POLYS
        ...
        END

    Returns: list of [x, y, z] integer coordinates.
    """
    vertices = []
    with open(file) as f:
        for line in f:
            parts = [elt.strip() for elt in line.split(' ')]
            if len(parts) == 4 or len(parts) == 8:
                vertices.append([
                    int(float(parts[1])),
                    int(float(parts[2])),
                    int(float(parts[3]))
                ])
    return vertices


def calculate_barycenter(points):
    """
    Calculate the barycenter (centroid) of a 3D point cloud.

    :param points: List of 3D points [[x, y, z], ...]
    :return: Tuple (x, y, z) of the barycenter.
    """
    if not points:
        raise ValueError("The list of points is empty")
    num_points = len(points)
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    sum_z = sum(point[2] for point in points)
    return (sum_x / num_points, sum_y / num_points, sum_z / num_points)
