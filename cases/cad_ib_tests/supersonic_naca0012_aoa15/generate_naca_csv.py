import numpy as np

def generate_naca0012_aoa15(num_points=200, aoa_deg=15.0, rot_center=(0.25, 0.0)):
    # Cosine spacing for high leading edge resolution
    beta = np.linspace(0, np.pi, num_points // 2)
    x = 0.5 * (1.0 - np.cos(beta))
    
    # NACA 0012 thickness distribution (closed trailing edge)
    t = 0.12
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    
    # Upper surface (from trailing edge to leading edge)
    x_upper = x[::-1]
    y_upper = yt[::-1]
    
    # Lower surface (from leading edge to trailing edge)
    x_lower = x[1:]
    y_lower = -yt[1:]
    
    # Combined closed polygon
    x_poly = np.concatenate([x_upper, x_lower])
    y_poly = np.concatenate([y_upper, y_lower])
    
    # Rotate around quarter chord by AoA (15 deg counter-clockwise)
    rad = np.radians(aoa_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    xc, yc = rot_center
    dx = x_poly - xc
    dy = y_poly - yc
    
    x_rot = xc + dx * cos_a - dy * sin_a
    y_rot = yc + dx * sin_a + dy * cos_a
    
    # Save to CSV
    output_path = "naca0012_aoa15.csv"
    with open(output_path, "w") as f:
        f.write("# x, y\n")
        for px, py in zip(x_rot, y_rot):
            f.write(f"{px:.8f}, {py:.8f}\n")
            
    print(f"Generated '{output_path}' with {len(x_rot)} boundary points rotated at {aoa_deg} deg AoA.")

if __name__ == "__main__":
    generate_naca0012_aoa15()
