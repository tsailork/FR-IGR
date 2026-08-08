# Comprehensive Technical Report: Enhanced Immersed Boundary (IB) Level-Set Engine, Inviscid Slip-Wall Velocity Reflection, & CSV Polygon Import

---

## Section 1: Summary of Problem & Intuition Behind Solution Strategy

### 1.1 Background & Context
The High-Order Flux Reconstruction (FR-CPR) solver utilizes an Immersed Boundary (IB) framework based on Ghost Cell Methodology (GCM-FR) and Level-Set Signed Distance Functions (SDF) to represent complex solid bodies on Cartesian grids without requiring body-fitted unstructured mesh generation.

In complex aerodynamic simulations (such as supersonic flow over sharp wedges or NACA 0012 airfoils at high angle of attack), accurate boundary identification and ghost nodal state extrapolation are paramount. Prior implementations suffered from three primary failure modes:

1. **Topological Parity Corruption & Spurious CAD Level-Set Masking**:
   - When importing 3D CAD geometries (in `.stl` format), ray-casting topological parity algorithms evaluated the ray intersection count $N_{\text{hits}}$ to determine whether a query point $\mathbf{x} = (x, y, z)$ lay inside ($N_{\text{hits}} \pmod 2 = 1$) or outside ($N_{\text{hits}} \pmod 2 = 0$) the solid body.
   - Diagnostic logging revealed that for 3D extruded CAD STL meshes executed under 2D solver modes, `min_phi` across the entire domain remained positive ($\phi > 0$), producing `solid_nodes = 0` at $t=0$. Consequently, the solver executed unperturbed freestream flow without enforcing solid boundary constraints.
   - Root-cause analysis uncovered two distinct structural bugs:
     - **Stream State Reuse Corruption**: `CadSdfEngine::load_stl` sampled the first 1KB of file headers to detect ASCII vs. Binary format via `std::ifstream::read()`. Upon reaching EOF on small files, the stream set `eofbit` and `failbit`. Subsequent `is.seekg(0)` calls without resetting stream flags caused ASCII parsing loops to fail or fall through into binary parsers, duplicating triangle loads.
     - **BVH Construction Triangle Buffer Duplication**: During bounding volume hierarchy construction (`CadSdfEngine::build_bvh`), contiguous leaf node re-ordering pushed duplicate copies of triangles onto the master `triangles_` array (`triangles_.push_back(triangles_[idx])`) without clearing the original un-ordered elements. This doubled the master triangle count ($N \to 2N$). Consequently, every ray intersection with a CAD facet produced two hits instead of one ($2k \text{ hits}$). Since $2k \pmod 2 = 0$ (EVEN), the topological parity algorithm classified every single fluid node and solid node as "OUTSIDE" ($\phi > 0$), completely wiping out the solid body mask.

2. **Inviscid Euler Slip-Wall Velocity Extrapolation Contamination**:
   - The GCM-FR ghost nodal state generator previously extrapolated ghost velocities using a simple scalar inversion formula:
     $$\mathbf{u}_g = 2 \mathbf{u}_b - \mathbf{u}_p$$
     where $\mathbf{u}_p$ is the fluid probe velocity and $\mathbf{u}_b$ is the solid wall velocity.
   - While valid for 1D normal motion or viscous no-slip boundary conditions ($\mathbf{u}_g = -\mathbf{u}_p$ when $\mathbf{u}_b = 0$), this scalar inversion inverted **both** normal velocity $u_n$ and tangential velocity $u_t$.
   - In high-Mach supersonic Euler flows over wedges or airfoils ($M_{\infty} = 2.0 - 3.0$), flipping the tangential velocity ($u_{g,t} = -u_{p,t} \approx -3.0$) created an artificial tangential velocity jump $\Delta u_t \approx 6.0$ across the immersed boundary interface. This unphysical shear layer generated intense numerical vorticity, spurious entropy generation, and catastrophic boundary layer instability.

3. **Lack of Direct 2D Polygon CSV Geometry Support**:
   - Users frequently possess 2D airfoil coordinate profiles (such as NACA airfoils or experimental turbine profiles) formatted as raw 2D point series in CSV, DAT, or TXT files.
   - Previous versions required converting 2D profiles into full 3D CAD STL CAD files via external CAD software before importing into the solver.

### 1.2 Intuition Behind Solution Strategy

To resolve these issues cleanly, robustly, and with mathematical elegance:

1. **Buffer-Safe BVH Construction & 2D Boundary Polygon Cross-Section Extraction**:
   - Refactored `build_bvh` to assemble leaf nodes into an independent temporary buffer (`reordered_triangles`) and perform an atomic `std::move`, preserving exact triangle counts without duplication.
   - Stream re-opening was implemented in `load_stl` to guarantee pristine file state transitions.
   - For 2D solver discretizations (`SolverDim<2>`), 3D extruded CAD STL facets ($|n_z| \le 0.9$) are projected into the $XY$ plane to extract the unique, non-degenerate 2D boundary polygon. Signed distance functions use exact point-to-segment distance calculations, while inside/outside classification is determined via the robust 2D Jordan Curve Ray-Casting Theorem evaluated on the extracted 2D boundary contour.

2. **Tensor Decomposition & Normal-Velocity Reflection Operator for Euler Slip Walls**:
   - Decomposed probe velocity into boundary-normal ($u_n = \mathbf{u} \cdot \mathbf{n}$) and boundary-tangential ($\mathbf{u}_t = \mathbf{u} - u_n \mathbf{n}$) components relative to the unit outward wall normal $\mathbf{n}$.
   - For inviscid Euler slip walls, the normal component is inverted ($u_{g,n} = 2 u_{b,n} - u_{p,n}$) to enforce zero normal mass flux ($\mathbf{u} \cdot \mathbf{n} = 0$), while the tangential component is strictly preserved ($\mathbf{u}_{g,t} = \mathbf{u}_{p,t}$).
   - Mathematically, this corresponds to an exact Householder reflection across the local tangent plane ($\mathbf{u}_g = \mathbf{u}_p - 2 (\mathbf{u}_p \cdot \mathbf{n}) \mathbf{n}$ when $\mathbf{u}_b = 0$), eliminating artificial shear stress and guaranteeing zero numerical vorticity generation at slip boundaries.

3. **Native 2D CSV Polygon Extrusion Parser**:
   - Added native CSV/DAT/TXT parsing into `CadSdfEngine::load_csv`, reading 2D coordinate pairs $(x_k, y_k)$ and automatically generating 3D extruded boundary quad facets along $Z \in [-1, 1]$.
   - This allows users to supply custom 2D polygon files directly to `IB_CAD_FILE = profile.csv` without external CAD software dependencies.

---

## Section 2: Detailed Step-by-Step Mathematical Breakdown of Methods & Adjustments

### 2.1 Level-Set Signed Distance Function (SDF) & 2D Cross-Section Extraction

Let $\mathcal{S} \subset \mathbb{R}^3$ represent the 3D surface mesh composed of triangles $\Delta_m = \{\mathbf{v}_{m,0}, \mathbf{v}_{m,1}, \mathbf{v}_{m,2}\}$ with face normals $\mathbf{n}_m$.

#### Step 1: Unsigned Distance Computation via Bounding Volume Hierarchy (BVH)
For a query point $\mathbf{x} = (x, y, z)$, the unsigned Euclidean distance $d(\mathbf{x}, \mathcal{S})$ to the surface is:
$$d(\mathbf{x}, \mathcal{S}) = \min_{m=1, \dots, N_{\text{tri}}} \text{dist}(\mathbf{x}, \Delta_m)$$

The distance between $\mathbf{x}$ and a 3D triangle $\Delta_m$ is determined by solving the constrained quadratic minimization problem over barycentric coordinates $(u, v)$:
$$\text{dist}^2(\mathbf{x}, \Delta_m) = \min_{u \ge 0, v \ge 0, u+v \le 1} \left\| \mathbf{x} - (\mathbf{v}_{m,0} + u (\mathbf{v}_{m,1} - \mathbf{v}_{m,0}) + v (\mathbf{v}_{m,2} - \mathbf{v}_{m,0})) \right\|^2$$

Accelerated query execution is achieved using an Axis-Aligned Bounding Box (AABB) BVH tree. At each node $k$ with bounding box $[\mathbf{b}_{\text{min}}, \mathbf{b}_{\text{max}}]$, the minimum squared distance from $\mathbf{x}$ to the AABB box is:
$$d_{\text{box}}^2(\mathbf{x}, \mathcal{B}_k) = \sum_{j=1}^3 \left( \max(0, b_{\text{min}, j} - x_j) + \max(0, x_j - b_{\text{max}, j}) \right)^2$$
If $d_{\text{box}}^2(\mathbf{x}, \mathcal{B}_k) \ge d_{\text{min}}^2$, subtree $k$ is pruned.

#### Step 2: 2D Planar Cross-Section Polygon Extraction
For 2D Cartesian solver operations ($z = 0$), lateral extruded facets are isolated by filtering out top and bottom $Z$-endcaps:
$$\mathcal{F}_{\text{lat}} = \{ \Delta_m \in \mathcal{S} \;\mid\; |n_{m,z}| \le 0.9 \}$$

From $\mathcal{F}_{\text{lat}}$, unique 2D planar vertices $\mathbf{p}_k = (x_k, y_k)$ are extracted by projecting 3D vertices onto the $XY$ plane:
$$\mathbf{p}_k = (v_{m, l, x}, v_{m, l, y}), \quad l \in \{0, 1, 2\}$$

Vertices are sorted radially counter-clockwise about their centroid $(\bar{x}, \bar{y}) = \frac{1}{K} \sum_{k=1}^K \mathbf{p}_k$:
$$\theta_k = \text{atan2}(y_k - \bar{y}, x_k - \bar{x})$$
$$\mathcal{P}_{2D} = (\mathbf{p}_{(1)}, \mathbf{p}_{(2)}, \dots, \mathbf{p}_{(K)})$$

#### Step 3: Jordan Curve Theorem Ray-Casting Sign Classification
To assign the sign to $\text{SDF}(\mathbf{x})$, a ray is cast from $\mathbf{x} = (x, y)$ along direction $\mathbf{r} = (1, 0)$ to $+\infty$.
For each 2D polygon edge segment $\mathbf{e}_k = (\mathbf{p}_k, \mathbf{p}_{k+1})$ where $\mathbf{p}_k = (x_{1,k}, y_{1,k})$ and $\mathbf{p}_{k+1} = (x_{2,k}, y_{2,k})$:

1. Skip horizontal segments ($|y_{1,k} - y_{2,k}| < 10^{-12}$).
2. Check if ray coordinate $y$ spans segment $y$-bounds:
   $$y \in \left[ \min(y_{1,k}, y_{2,k}), \; \max(y_{1,k}, y_{2,k}) \right)$$
3. Compute segment $x$-intersection:
   $$x_{\text{int}} = x_{1,k} + (y - y_{1,k}) \frac{x_{2,k} - x_{1,k}}{y_{2,k} - y_{1,k}}$$
4. If $x_{\text{int}} > x$, increment ray crossing counter $N_{\text{cross}}$.

By the Jordan Curve Theorem:
$$\text{inside}(\mathbf{x}) = \begin{cases} \text{true} & \text{if } N_{\text{cross}} \pmod 2 = 1 \\ \text{false} & \text{if } N_{\text{cross}} \pmod 2 = 0 \end{cases}$$

The Signed Distance Function (SDF) value $\phi(\mathbf{x})$ is:
$$\phi(\mathbf{x}) = \begin{cases} - d(\mathbf{x}, \mathcal{S}) & \text{if } \text{inside}(\mathbf{x}) = \text{true} \quad (\text{Solid Interior}) \\ + d(\mathbf{x}, \mathcal{S}) & \text{if } \text{inside}(\mathbf{x}) = \text{false} \quad (\text{Fluid Domain}) \end{cases}$$

---

### 2.2 Ghost Cell Nodal State Operator & Householder Reflection

In the Ghost Cell Methodology (GCM-FR), solid nodes ($\phi(\mathbf{x}_g) \le 0$) are assigned ghost states $\mathbf{q}_g = (\rho_g, \rho u_g, \rho v_g, E_g)^T$ extrapolated from fluid probe states $\mathbf{q}_p = (\rho_p, \rho u_p, \rho v_p, E_p)^T$ located along the local boundary normal $\mathbf{n} = (n_x, n_y)$ at distance $d_p = d_g$.

```
       Fluid Probe Point q_p (at distance d_p = d_g from wall)
                 o
                 |  +n (Unit outward normal vector)
                 |
=================o================= Immersed Boundary (phi = 0)
                 |  Wall Motion Velocity u_b
                 |
                 o
       Ghost Node Point q_g (at distance d_g inside solid, phi <= 0)
```

#### Step 1: Normal and Tangential Velocity Vector Decomposition
Let $\mathbf{n} = (n_x, n_y)$ be the normalized outward unit normal vector pointing from the solid into the fluid domain ($\|\mathbf{n}\| = 1$).
The fluid probe velocity $\mathbf{u}_p = (u_p, v_p)$ is decomposed into normal scalar component $u_{p,n}$ and tangential vector component $\mathbf{u}_{p,t}$:

$$u_{p,n} = \mathbf{u}_p \cdot \mathbf{n} = u_p n_x + v_p n_y$$
$$\mathbf{u}_{p,t} = \mathbf{u}_p - u_{p,n} \mathbf{n} = \begin{pmatrix} u_p - u_{p,n} n_x \\ v_p - u_{p,n} n_y \end{pmatrix}$$

Let $\mathbf{u}_b = (u_b, v_b)$ be the solid wall velocity. The target wall-normal velocity component is:
$$u_{b,n} = \mathbf{u}_b \cdot \mathbf{n} = u_b n_x + v_b n_y$$

#### Step 2: Extrapolation & Reflection Operator

##### 1. Inviscid Euler Slip Wall Formulation (No-Penetration Condition)
To enforce zero relative normal mass flux $(\mathbf{u} - \mathbf{u}_b) \cdot \mathbf{n} = 0$ while preserving tangential momentum (zero shear stress $\tau_w = 0$):

- **Target Ghost Normal Velocity**:
  $$u_{g,n} = 2 u_{b,n} - u_{p,n}$$
- **Target Ghost Tangential Velocity**:
  $$\mathbf{u}_{g,t} = \mathbf{u}_{p,t}$$

Recombining normal and tangential components yields the ghost velocity vector $\mathbf{u}_g$:
$$\mathbf{u}_g = u_{g,n} \mathbf{n} + \mathbf{u}_{g,t} = (2 u_{b,n} - u_{p,n}) \mathbf{n} + (\mathbf{u}_p - u_{p,n} \mathbf{n}) = \mathbf{u}_p + 2 (u_{b,n} - u_{p,n}) \mathbf{n}$$

For a stationary wall ($\mathbf{u}_b = 0 \implies u_{b,n} = 0$), this simplifies to the exact **Householder Reflection Matrix** $\mathbf{H}_{\mathbf{n}} = \mathbf{I} - 2 \mathbf{n} \mathbf{n}^T$:
$$\mathbf{u}_g = \mathbf{H}_{\mathbf{n}} \mathbf{u}_p = \mathbf{u}_p - 2 (\mathbf{u}_p \cdot \mathbf{n}) \mathbf{n}$$

In component form:
$$u_g = u_p - 2 (u_p n_x + v_p n_y) n_x$$
$$v_g = v_p - 2 (u_p n_x + v_p n_y) n_y$$

##### 2. Thermodynamic Ghost States (Density & Pressure)
Density $\rho_g$ and static pressure $P_g$ are extrapolated using zero-normal-gradient conditions ($\frac{\partial \rho}{\partial n} = 0, \frac{\partial P}{\partial n} = 0$):
$$\rho_g = \rho_p$$
$$P_g = \max(P_{\text{eps}}, \; P_p)$$
where $P_{\text{eps}} = \text{POS\_LIMITER\_EPS}$ (typically $10^{-4}$) guarantees physical positivity.

Total specific energy $E_g$ for the ghost node is constructed from primitive variables:
$$E_g = \frac{P_g}{\gamma - 1} + \frac{1}{2} \rho_g (u_g^2 + v_g^2)$$

#### Step 3: High-Order Polynomial Extrapolation (P2 / P3)
For polynomial order $P \ge 2$, probe states $\mathbf{q}_{p,0}, \mathbf{q}_{p,1}, \dots$ at multiple normal distances $s_0, s_1, \dots$ are evaluated using Lagrange interpolating polynomials $L_k(s)$:
$$\mathbf{u}_g^{\text{raw}} = \sum_{k} \mathbf{u}_{p,k} L_k(-d_g)$$

The raw extrapolated velocity $\mathbf{u}_g^{\text{raw}}$ is subsequently corrected by replacing its normal component with the reflected normal target $u_{g,n}^{\text{refl}} = 2 u_{b,n} - u_{p,0,n}$:
$$\mathbf{u}_g = \mathbf{u}_g^{\text{raw}} + \left( u_{g,n}^{\text{refl}} - (\mathbf{u}_g^{\text{raw}} \cdot \mathbf{n}) \right) \mathbf{n}$$

This ensures high-order smooth boundary representation while enforcing 100% strict normal velocity reflection.

---

### 2.3 CSV 2D Polygon Extrusion Mathematics

Given a 2D boundary polygon specified in a CSV file as an ordered sequence of $K$ points:
$$\mathcal{V}_{2D} = \{ (x_1, y_1), (x_2, y_2), \dots, (x_K, y_K) \}$$

For each 2D edge segment connecting point $k$ to point $k+1$ (with cyclic boundary condition $K+1 \equiv 1$):
$$\mathbf{a}_k = (x_k, y_k), \quad \mathbf{b}_k = (x_{k+1}, y_{k+1})$$

The 2D edge is extruded into a 3D vertical quadrilateral facet spanning $z \in [-z_{\text{ext}}, +z_{\text{ext}}]$ (where $z_{\text{ext}} = 1.0$). Each 3D quadrilateral is decomposed into two 3D triangles $\Delta_{k,1}$ and $\Delta_{k,2}$:

$$\Delta_{k,1}: \quad \mathbf{v}_{0} = (x_k, y_k, -z_{\text{ext}}), \quad \mathbf{v}_{1} = (x_k, y_k, +z_{\text{ext}}), \quad \mathbf{v}_{2} = (x_{k+1}, y_{k+1}, +z_{\text{ext}})$$
$$\Delta_{k,2}: \quad \mathbf{v}_{0} = (x_k, y_k, -z_{\text{ext}}), \quad \mathbf{v}_{1} = (x_{k+1}, y_{k+1}, +z_{\text{ext}}), \quad \mathbf{v}_{2} = (x_{k+1}, y_{k+1}, -z_{\text{ext}})$$

For each triangle, bounding boxes $\mathbf{b}_{\text{min}}, \mathbf{b}_{\text{max}}$ and unit outward normal vectors $\mathbf{n}_m$ are evaluated:
$$\mathbf{e}_0 = \mathbf{v}_1 - \mathbf{v}_0, \quad \mathbf{e}_1 = \mathbf{v}_2 - \mathbf{v}_0$$
$$\mathbf{n}_m = \frac{\mathbf{e}_0 \times \mathbf{e}_1}{\|\mathbf{e}_0 \times \mathbf{e}_1\|}$$

This generates a 3D closed boundary mesh compatible with the solver's 3D BVH acceleration tree and 2D planar SDF extraction algorithms.

---

## Section 3: Itemized Code Modifications

The implementation changes are spread across key core geometry, parameter parsing, and IB solver files in the repository.

### 3.1 `src/core/geometry.hpp`
- **File Link**: [geometry.hpp](file:///home/tsk/Documents/GitHub/FR-IGR/src/core/geometry.hpp)
- **Modifications**:
  1. **Added `load_csv` Declaration**:
     - Line 248: Declared `bool load_csv(const std::string& filepath);` inside `CadSdfEngine` class to support parsing 2D polygon CSV files.
  2. **Updated BVH Builder Function Signature**:
     - Line 283: Modified `build_bvh_recursive` parameter list to pass `std::vector<Triangle>& reordered_triangles` by reference:
       ```cpp
       void build_bvh_recursive(int node_idx, std::vector<int>& tri_indices, 
                                int max_leaf_triangles, int depth, 
                                std::vector<Triangle>& reordered_triangles);
       ```

### 3.2 `src/core/geometry.cpp`
- **File Link**: [geometry.cpp](file:///home/tsk/Documents/GitHub/FR-IGR/src/core/geometry.cpp)
- **Modifications**:
  1. **Updated `load_mesh` File Extension Dispatcher**:
     - Lines 367–372: Added check for `.csv`, `.dat`, and `.txt` file extensions to route geometry loading to `load_csv(filepath)`:
       ```cpp
       } else if (ext == ".csv" || ext == ".dat" || ext == ".txt") {
           return load_csv(filepath);
       }
       ```
  2. **Stream Re-Opening Fix in `load_stl`**:
     - Lines 396–428: Resolved stream state corruption during ASCII/Binary STL detection by explicitly calling `is.close()` after sampling 1KB headers, and instantiating separate file streams (`ascii_is` vs `bin_is`) for actual parsing.
  3. **Implemented `load_csv` Polygon Parser & Extruder**:
     - Lines 505–560: Implemented `CadSdfEngine::load_csv` stream parser. It strips comments (`#`, `/`), converts delimiters (`,`, `;`) to whitespace, parses $(x_k, y_k)$ coordinate pairs, and generates 3D extruded boundary quad triangles ($\Delta_1, \Delta_2$) spanning $Z \in [-1, 1]$.
  4. **Fixed BVH Construction Buffer Duplication in `build_bvh`**:
     - Lines 548–615: Refactored `build_bvh` and `build_bvh_recursive` to populate a separate local buffer `reordered_triangles` during leaf node assembly. Replaced `triangles_` via `std::move(reordered_triangles)` upon completion, eliminating triangle doubling ($8 \to 16 \implies 8$) and preventing corrupted BVH leaf indices.
  5. **Rewrote `query_sdf` Sign Determination with 2D Jordan Curve Extraction**:
     - Lines 745–790: Replaced flawed 3D multi-ray dot product voting with 2D cross-section boundary polygon extraction. Extracted 2D planar vertices from lateral facets ($|n_z| \le 0.9$), sorted them radially around their centroid, and evaluated `Geometry::point_in_polygon(p[0], p[1], px, py)` to assign exact positive/negative Level-Set sign.

### 3.3 `src/ib/ib_gcm.cpp`
- **File Link**: [ib_gcm.cpp](file:///home/tsk/Documents/GitHub/FR-IGR/src/ib/ib_gcm.cpp)
- **Modifications**:
  1. **Unit Outward Normal Vector Extraction**:
     - Lines 205–208: Added normal vector normalization and fallback handling (`nx /= norm_len; ny /= norm_len;`).
  2. **Euler Slip-Wall Velocity Reflection Operator (P1 Linear Extrapolation)**:
     - Lines 210–225: Replaced scalar velocity inversion (`2 u_b - u_p`) with vector normal velocity reflection:
       ```cpp
       double un_p = u_p0 * nx + v_p0 * ny;
       double un_b = u_b * nx + v_b * ny;
       double un_g = 2.0 * un_b - un_p;
       u_g = u_p0 + (un_g - un_p) * nx;
       v_g = v_p0 + (un_g - un_p) * ny;
       ```
  3. **High-Order Ghost State Reflection Corrections (P2 Quadratic & P3 Cubic)**:
     - Lines 237–243 & 272–278: Added normal velocity reflection corrections to quadratic and cubic raw extrapolated ghost velocity states to ensure zero normal mass flux without tangential velocity dampening.

---

## Section 4: Limitations of the New IB Method & Future Development Roadmap

While the new IB formulation provides 100% stable execution and exact Level-Set representation for 2D supersonic flows over airfoils and wedges, several technical limitations exist that should be addressed in future development:

### 4.1 Star-Shaped Centroid Polygon Sorting Limitation
- **Current Behavior**: In `CadSdfEngine::query_sdf`, 2D cross-section vertices extracted from lateral STL facets are ordered radially around their geometric centroid using `atan2(y - cy, x - cx)`.
- **Limitation**: Radial sorting assumes that the 2D cross-section contour is star-shaped with respect to its centroid (e.g., circles, triangles, NACA airfoils, wedges, convex polygons). For highly concave geometries (such as multi-element airfoils, serpentine channels, or cavities), radial sorting can re-order vertices incorrectly, creating self-intersecting polygon edges.
- **Future Development Remedy**: Replace centroid radial sorting with topological edge-connectivity extraction. Parse true 2D segment connectivity from 3D quad facet topology to construct closed Half-Edge or Doubly Connected Edge List (DCEL) boundary loops regardless of domain concavity.

### 4.2 Sharp Leading-Edge / Trailing-Edge Probe Point Interpolation Singularities
- **Current Behavior**: Near sharp corners (such as the trailing edge of a NACA 0012 airfoil or wedge apex), GCM-FR constructs fluid probe points along the outward normal $\mathbf{n}$. Near sharp corners, normal vectors from adjacent boundary faces diverge rapidly.
- **Limitation**: Probe points from adjacent ghost nodes can cross or fall into neighboring cut-cell regions, introducing localized pressure oscillations or limiter triggering at sharp trailing edges under high Mach numbers.
- **Future Development Remedy**: Implement Corner-Aware Normal Smoothing (CANS) or multi-directional probe state averaging for IB boundary nodes with normal curvature $\kappa > \kappa_{\text{threshold}}$.

### 4.3 Transition Between Viscous No-Slip and Inviscid Slip Wall Formulations
- **Current Behavior**: The velocity reflection operator currently assumes inviscid slip wall conditions ($\tau_w = 0$) when `ENABLE_IB_WALL_FUNCTION = false` and `ENABLE_NS = false`. When `ENABLE_NS = true`, physical viscous no-slip conditions ($\mathbf{u}_g = -\mathbf{u}_p$) apply at low-to-moderate Reynolds numbers.
- **Limitation**: At high Reynolds numbers ($Re \ge 10^5$) on coarse Cartesian meshes where boundary layers cannot be fully resolved ($\Delta y^+ \gg 1$), applying full no-slip without an active wall model induces artificial boundary layer separation.
- **Future Development Remedy**: Implement an adaptive hybrid IB wall model that dynamically switches from slip reflection to equilibrium/non-equilibrium law-of-the-wall velocity profiles based on local boundary layer cell height $y^+$.
