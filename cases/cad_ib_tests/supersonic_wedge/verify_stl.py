#!/usr/bin/env python3
import math

def parse_stl(filename):
    triangles = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_normal = None
    current_verts = []
    for line in lines:
        line = line.strip()
        if line.startswith('facet normal'):
            parts = line.split()
            current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
            current_verts = []
        elif line.startswith('vertex'):
            parts = line.split()
            current_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('endfacet'):
            if len(current_verts) == 3:
                triangles.append({'normal': current_normal, 'v0': current_verts[0], 'v1': current_verts[1], 'v2': current_verts[2]})
    return triangles

def ray_triangle_intersection(orig, dir_vec, tri):
    EPS = 1e-10
    v0, v1, v2 = tri['v0'], tri['v1'], tri['v2']
    e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]]
    e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]]
    pvec = [
        dir_vec[1]*e2[2] - dir_vec[2]*e2[1],
        dir_vec[2]*e2[0] - dir_vec[0]*e2[2],
        dir_vec[0]*e2[1] - dir_vec[1]*e2[0]
    ]
    det = e1[0]*pvec[0] + e1[1]*pvec[1] + e1[2]*pvec[2]
    if abs(det) < EPS:
        return False, 0.0
    inv_det = 1.0 / det
    tvec = [orig[0]-v0[0], orig[1]-v0[1], orig[2]-v0[2]]
    u = (tvec[0]*pvec[0] + tvec[1]*pvec[1] + tvec[2]*pvec[2]) * inv_det
    if u < -EPS or u > 1.0 + EPS:
        return False, 0.0
    qvec = [
        tvec[1]*e1[2] - tvec[2]*e1[1],
        tvec[2]*e1[0] - tvec[0]*e1[2],
        tvec[0]*e1[1] - tvec[1]*e1[0]
    ]
    v = (dir_vec[0]*qvec[0] + dir_vec[1]*qvec[1] + dir_vec[2]*qvec[2]) * inv_det
    if v < -EPS or u + v > 1.0 + EPS:
        return False, 0.0
    t = (e2[0]*qvec[0] + e2[1]*qvec[1] + e2[2]*qvec[2]) * inv_det
    if t > EPS:
        return True, t
    return False, 0.0

def query_inside_raycast(x, y, z, triangles):
    orig = [x, y, z]
    # Test multiple ray directions for robustness against degenerate edge hits
    ray_dirs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0]
    ]
    inside_votes = 0
    for rdir in ray_dirs:
        intersections = 0
        for tri in triangles:
            hit, t = ray_triangle_intersection(orig, rdir, tri)
            if hit:
                intersections += 1
        if intersections % 2 == 1:
            inside_votes += 1
    return (inside_votes >= 2)

def main():
    tris = parse_stl('wedge_3d.stl')
    print("Testing Ray-Casting Topological Parity across domain points:")
    test_points = [
        (0.2, 0.5, "In front of wedge (Fluid)"),
        (1.0, 0.5, "Above top slanted face (Fluid)"),
        (1.0, 0.05, "Inside wedge (Solid)"),
        (1.8, 0.30, "Inside wedge near trailing top (Solid)"),
        (1.8, 0.40, "Above wedge near trailing top (Fluid)"),
        (2.5, 0.5, "Behind wedge (Fluid)"),
        (2.5, 0.2, "Behind wedge back wall (Fluid)")
    ]
    for x, y, desc in test_points:
        inside = query_inside_raycast(x, y, 0.0, tris)
        status = "SOLID (phi <= 0)" if inside else "FLUID (phi > 0)"
        print(f"  Point ({x:.2f}, {y:.2f}) [{desc}]: -> {status}")

if __name__ == '__main__':
    main()
