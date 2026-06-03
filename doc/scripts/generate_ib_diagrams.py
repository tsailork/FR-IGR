import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure assets directory exists
os.makedirs('../assets', exist_ok=True)

def generate_sbm_diagram():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Draw Cartesian Grid
    for x in np.arange(0, 1.1, 0.2):
        ax.axvline(x, color='#e0e0e0', linestyle='-', linewidth=1, zorder=1)
        ax.axhline(x, color='#e0e0e0', linestyle='-', linewidth=1, zorder=1)
        
    # 2. Draw physical boundary (a quarter circle)
    theta = np.linspace(0, np.pi/2, 100)
    R = 0.55
    cx, cy = 0.0, 0.0
    bx = cx + R * np.cos(theta)
    by = cy + R * np.sin(theta)
    
    ax.plot(bx, by, color='black', linewidth=3, label='Physical Boundary ($\Gamma$)', zorder=2)
    ax.fill_between(bx, 0, by, color='#d3d3d3', alpha=0.5, label='Solid Domain ($\Omega_s$)', zorder=1)
    
    # 3. Draw surrogate boundary (stair-step approximation, inside the solid)
    surrogate_pts = [
        (0.4, 0.0), (0.4, 0.2), 
        (0.2, 0.2), (0.2, 0.4),
        (0.0, 0.4)
    ]
    sx, sy = zip(*surrogate_pts)
    ax.plot(sx, sy, color='red', linewidth=2, linestyle='--', label='Surrogate Boundary ($\~{\Gamma}$)', zorder=3)
    
    # 4. Draw face points for P=2 (3 points) on the chosen face
    # Choose the face at x=0.4, y in [0.0, 0.2].
    face_x = 0.4
    face_y_coords = [0.0 + 0.2*(0.5 * (1.0 + z)) for z in [-0.7745966692, 0.0, 0.7745966692]] # Gauss-Legendre roots for P=2
    for fy in face_y_coords:
        ax.plot(face_x, fy, 'mo', markersize=6, label='P=2 Face Points' if fy == face_y_coords[0] else "", zorder=4)
        
    # We draw the ray from the middle face point (which is at y=0.1)
    face_y = 0.1
    
    # Calculate normal to the circle at the intersection
    # The ray goes from physical boundary out into the fluid.
    mag = np.sqrt(face_x**2 + face_y**2)
    nx = face_x / mag
    ny = face_y / mag
    
    # Physical boundary point
    px = R * nx
    py = R * ny
    ax.plot(px, py, 'bo', markersize=8, label='Physical Boundary Point ($D$)', zorder=4)
    
    # Draw ray from surrogate face point to the last donor point
    ray_len_total = 0.4
    last_d = 0.3
    last_dx = px + (ray_len_total/0.4)*last_d * nx
    last_dy = py + (ray_len_total/0.4)*last_d * ny
    
    # Arrow from face_x, face_y to last_dx, last_dy
    ax.arrow(face_x, face_y, last_dx - face_x, last_dy - face_y, head_width=0.02, head_length=0.03, fc='blue', ec='blue', zorder=3, length_includes_head=True)
    
    # Draw donor stencil points along the ray (for P=2, we use P+1=3 points)
    donor_dists = [0.1, 0.2, 0.3]
    donor_cells = set()
    
    for i, d in enumerate(donor_dists):
        dx = px + (ray_len_total/0.4)*d * nx
        dy = py + (ray_len_total/0.4)*d * ny
        if i == 0:
            ax.plot(dx, dy, 'gs', markersize=8, label='1D Stencil Points', zorder=5)
        else:
            ax.plot(dx, dy, 'gs', markersize=8, zorder=5)
            
        # Determine which 0.2x0.2 Cartesian cell this donor point falls into
        cx = np.floor(dx / 0.2) * 0.2
        cy = np.floor(dy / 0.2) * 0.2
        donor_cells.add((cx, cy))
            
    # Draw P=2 solution points in EVERY donor element identified
    gl_roots = [-0.7745966692, 0.0, 0.7745966692]
    first_el = True
    for (cx, cy) in donor_cells:
        ax.add_patch(patches.Rectangle((cx, cy), 0.2, 0.2, fill=True, color='#ffffcc', alpha=0.5, label='Donor Element(s)' if first_el else "", zorder=1))
        for zx in gl_roots:
            for zy in gl_roots:
                sx = cx + 0.1 * (1.0 + zx)
                sy = cy + 0.1 * (1.0 + zy)
                ax.plot(sx, sy, 'cx', markersize=6, markeredgewidth=2, label='P=2 Solution Points' if first_el and zx == gl_roots[0] and zy == gl_roots[0] else "", zorder=4)
        first_el = False
            
    # Labels
    ax.text(0.1, 0.9, 'Fluid Domain ($\Omega_f$)', fontsize=14, fontweight='bold')
    ax.text(0.15, 0.15, 'Solid\n($\Omega_s$)', fontsize=14, fontweight='bold', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Shifted Boundary Method (SBM) Conceptual Diagram', fontsize=16)
    ax.axis('off')
    ax.legend(loc='upper right', framealpha=1.0)
    
    plt.tight_layout()
    plt.savefig('../assets/sbm_diagram.svg', format='svg', bbox_inches='tight')
    plt.close()

def generate_vpm_diagram():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid
    nx, ny = 50, 50
    x = np.linspace(-0.1, 1.1, nx)
    y = np.linspace(-0.1, 1.1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Physical boundary (circle)
    R = 0.4
    cx, cy = 0.5, 0.5
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    
    # VPM Mask function (chi) with a small smoothing layer
    epsilon = 0.03
    chi = 0.5 * (1.0 - np.tanh((dist - R) / epsilon))
    
    # Plot heatmap
    c = ax.pcolormesh(X, Y, chi, cmap='Blues', shading='auto', zorder=1)
    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Penalization Mask ($\chi$)', rotation=270, labelpad=20, fontsize=12)
    
    # Draw physical boundary line
    theta = np.linspace(0, 2*np.pi, 100)
    bx = cx + R * np.cos(theta)
    by = cy + R * np.sin(theta)
    ax.plot(bx, by, color='black', linewidth=2, linestyle='--', label='Physical Boundary ($\Gamma$)', zorder=2)
    
    # Draw a velocity field overlay to show it stopping
    # Background velocity going right to left, wrapping around the circle
    # We'll just draw some stylized streamlines that avoid the circle
    # For a real feel, just plot vectors that go to 0 inside
    vx = np.ones_like(X)
    vy = np.zeros_like(Y)
    
    # Deflect around circle (very crude analytical potential flow)
    for i in range(nx):
        for j in range(ny):
            dx = X[i,j] - cx
            dy = Y[i,j] - cy
            r2 = dx**2 + dy**2
            if r2 > R**2:
                # Flow past cylinder
                vx[i,j] = 1.0 - R**2 * (dx**2 - dy**2) / (r2**2)
                vy[i,j] = - R**2 * (2 * dx * dy) / (r2**2)
            else:
                vx[i,j] = 0.0
                vy[i,j] = 0.0
                
    # Multiply by (1-chi) to show VPM effect explicitly damping it
    vx_damped = vx * (1.0 - chi)
    vy_damped = vy * (1.0 - chi)
    
    # Subsample for quiver
    skip = 4
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              vx_damped[::skip, ::skip], vy_damped[::skip, ::skip], 
              color='red', alpha=0.6, scale=15, zorder=3, label='Velocity Field')
              
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Volume Penalization Method (VPM)', fontsize=16)
    ax.axis('off')
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    # add quiver to legend manually if not there
    ax.legend(handles, labels, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('../assets/vpm_diagram.svg', format='svg', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_sbm_diagram()
    generate_vpm_diagram()
    print("SVGs generated successfully in doc/assets/")
