# Comprehensive Documentation of Attempted Phantom Pressure Regularization (PPR) Strategies and Mathematical Analysis

## Executive Overview
This document provides a comprehensive, mathematically rigorous record of all investigated **Phantom Pressure Regularization (PPR)** and **Anisotropic Phantom Stress Relaxation (APSR)** strategies for handling shocks and discontinuities in 2D and 3D Flux Reconstruction (FR / CPR) solvers.

Each section documents the governing continuum equations, physical mechanisms, mathematical derivations, linear dispersion relations, 2nd law entropy proofs, boundary and interface flux treatments, and current investigation status for each strategy.

---

## 1. Mathematical Framework & Non-Equilibrium Thermodynamics

The 2D/3D Euler equations coupled with a non-equilibrium scalar phantom pressure field $S = \rho P_{\text{phan}}$ are expressed in conservation form as:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$

$$\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u} + \mathbf{\Pi}) = 0$$

$$\frac{\partial E}{\partial t} + \nabla \cdot \big( (E \mathbf{I} + \mathbf{\Pi}) \mathbf{u} \big) = 0$$

$$\frac{\partial S}{\partial t} + \nabla \cdot (S \mathbf{u}_{\text{adv}}) = -\frac{1}{\tau} (S - \rho P_{\text{phys}})$$

where:
- $\rho$ is fluid density.
- $\mathbf{u} = (u, v, w)^T$ is fluid velocity vector.
- $E = \frac{P_{\text{phys}}}{\gamma - 1} + \frac{1}{2} \rho |\mathbf{u}|^2$ is total energy density.
- $P_{\text{phys}} = (\gamma - 1) \left( E - \frac{1}{2} \rho |\mathbf{u}|^2 \right)$ is physical thermodynamic pressure.
- $P_{\text{phan}} = S / \rho$ is non-equilibrium phantom pressure.
- $\tau$ is non-equilibrium relaxation time scale.
- $\mathbf{\Pi}$ is total momentum flux stress tensor.
- $\mathbf{u}_{\text{adv}}$ is advection velocity vector for non-equilibrium scalar transport.

The relaxation time scale $\tau$ is non-dimensionalized via relaxation parameter $C_\tau$:
$$\tau = C_\tau \frac{h_{\text{node}}}{a_{\text{local}} + |\mathbf{u}|}$$
where $h_{\text{node}} = \frac{h_{\text{element}}}{(P+1)^2}$ is the high-order solution node spacing for polynomial degree $P$, and $a_{\text{local}} = \sqrt{\gamma P_{\text{phys}} / \rho}$ is physical sound speed.

---

## 2. Strategy 1: Isotropic Phantom Pressure Regularization (Isotropic PPR)

### 2.1 Mathematical Formulation
Isotropic PPR defines the momentum stress tensor as a scalar regularized pressure $P_{\text{reg}}$ acting uniformly across all coordinate directions:

$$\mathbf{\Pi}_{\text{PPR}} = P_{\text{reg}} \mathbf{I}$$

$$P_{\text{reg}} = (1 + \theta) P_{\text{phys}} - \theta P_{\text{phan}} = P_{\text{phys}} + \theta (P_{\text{phys}} - P_{\text{phan}})$$

$$\mathbf{u}_{\text{adv}} = \mathbf{u} \quad \text{(Fluid Streamline Advection)}$$

where $\theta \ge 0$ is the non-equilibrium coupling intensity parameter.

### 2.2 Dynamic Shock Width Target Law
To achieve a physical shock transition width spanning $N_{\text{cells\_shock}}$ element layers in a $P$-th order Flux Reconstruction solver, coupling intensity $\theta_{\text{target}}$ is dynamically derived as:

$$\theta_{\text{target}} = \frac{N_{\text{cells\_shock}} (P+1) (\gamma + 1)}{8 C_\tau} (1 + M) \phi_{\text{eff}}$$

where:
- $M = |\mathbf{u}| / a_{\text{local}}$ is local Mach number.
- $\phi_{\text{eff}} = \max\left( \phi_{\text{comp}}, \phi_{\text{jump}} \right)$ is effective kinematic shock sensor indicator.
- $\phi_{\text{comp}} = \frac{h_{\text{node}} \max(0, -\nabla \cdot \mathbf{u})}{a_{\text{local}} + \epsilon}$ measures volume velocity compression.
- $\phi_{\text{jump}} = \frac{\max(0, -\Delta u_{\text{face}})}{(P+1) a_{\text{local}}}$ measures face Riemann velocity jump.

### 2.3 Linearized 2D Dispersion Relation & High/Low Frequency Limits
Consider a 2D uniform background state $U_0 = (\rho_0, u_0, 0, P_0, S_0 = \rho_0 P_0)$ subjected to a small acoustic perturbation $e^{i(k_x x + k_y y - \omega t)}$ with wavenumber $\mathbf{k} = (k_x, k_y)^T$ and Doppler-shifted frequency $\Omega = \omega - k_x u_0$.

Linearizing the governing equations yields the perturbed regularized pressure:
$$P_{\text{reg}}' = \left[ 1 + \frac{i \theta \tau \Omega}{1 + i \tau \Omega} \right] P_{\text{phys}}'$$

Substituting into the linearized momentum and energy equations yields the dispersion relation for Doppler acoustic modes:
$$\Omega^2 = a_0^2 k^2 \left[ \frac{1 + (1 + \theta) i \tau \Omega}{1 + i \tau \Omega} \right]$$
where $k^2 = k_x^2 + k_y^2$.

#### Asymptotic Asymptotes:
1. **Low-Frequency Limit ($\tau \Omega \ll 1$, Long Waves)**:
   $$\Omega^2 \approx a_0^2 k^2 \implies a_{\text{eq}} = a_0$$
   Long acoustic waves travel at the equilibrium thermodynamic sound speed $a_0$.
2. **High-Frequency Limit ($\tau \Omega \gg 1$, Short Waves)**:
   $$\Omega^2 \approx a_0^2 (1 + \theta) k^2 \implies a_{\text{frozen}} = a_0 \sqrt{1 + \theta}$$
   Short acoustic waves travel at the inflated non-equilibrium frozen sound speed $a_{\text{frozen}}$.
3. **Whitham Sub-Characteristic Condition**:
   $$a_{\text{eq}} \le a_{\text{frozen}} \implies a_0 \le a_0 \sqrt{1 + \theta} \quad (\text{Satisfied for all } \theta \ge 0)$$
4. **Effective Bulk Viscosity**:
   At low wavenumbers ($k \to 0$), the relaxation term generates an effective bulk viscosity:
   $$\nu_{\text{eff}} = \frac{1}{2} \theta \tau a_0^2 = \frac{1}{2} \theta C_\tau h_{\text{node}} a_0 \left( \frac{a_0}{a_0 + |\mathbf{u}_0|} \right)$$

### 2.4 Mathematical Origin of Spurious Flow-Aligned Oscillations
When constant coupling intensity $\theta$ is made large (e.g., $\theta = 30.0$) to spread strong shocks over multi-cell layers, three coupled mathematical mechanisms trigger flow-aligned spatial oscillations:

#### Mechanism A: Anisotropic Streamline Advection vs. Isotropic Acoustic Wave Speed
Phantom pressure scalar $S$ is purely advected along fluid streamlines:
$$\frac{\partial S}{\partial t} + u_0 \frac{\partial S}{\partial x} = -\frac{1}{\tau} (S - \rho P_{\text{phys}})$$
It possesses **zero transverse spatial transport** ($\partial_y S$ is not advected in $y$). For an oblique shock inclined at angle $\beta \ne 90^\circ$, physical pressure gradient $\nabla P_{\text{phys}}$ has non-zero $x$ and $y$ components. $P_{\text{phan}}$ advects $y$-variations strictly downstream along $x$-streamlines, while $P_{\text{reg}}$ acoustic waves propagate isotropically at $a_{\text{frozen}} = a_0 \sqrt{1+\theta}$. This transverse phase lag produces persistent flow-aligned spatial ripples.

#### Mechanism B: Stationary Standing Wave Resonance
In supersonic flow ($M_0 = u_0 / a_0 > 1$), the laboratory phase velocity of upstream-propagating acoustic modes is:
$$c_p = u_0 - a_{\text{frozen}} = u_0 - a_0 \sqrt{1 + \theta}$$
When coupling intensity satisfies $\sqrt{1 + \theta} \approx M_0$, laboratory phase velocity drops to zero ($c_p \to 0$). Upstream acoustic waves freeze relative to the computational grid, forming stationary standing wave ripples.

#### Mechanism C: High-Wavenumber Dissipation Saturation
The dissipation rate of acoustic perturbations is given by the imaginary part of Doppler frequency:
$$\text{Im}(\Omega) = \frac{\theta \tau a_0^2 k^2}{2 (1 + \tau^2 \Omega^2)} \xrightarrow{k \to \infty} \frac{\theta}{2 \tau (1 + \theta)} \approx \frac{1}{2\tau}$$
Dissipation saturates at a finite upper bound $\frac{1}{2\tau}$ for high-wavenumber grid modes ($k \sim 1/h$), while phase velocity inflates by $\sqrt{1+\theta}$. This imbalance causes high-frequency dispersive ringing behind shock fronts.

### 2.5 Status Notice
**Status**: Tested. Further investigation is needed.

---

## 3. Strategy 2: Rectified Anisotropic Phantom Stress Relaxation (APSR-R)

### 3.1 Mathematical Formulation
To eliminate sound speed inflation transverse to the shock normal, APSR-R replaces scalar regularized pressure $P_{\text{reg}} \mathbf{I}$ with an anisotropic phantom stress tensor:

$$\mathbf{\Pi}_{\text{APSR-R}} = P_{\text{phys}} \mathbf{I} + \boldsymbol{\tau}_{\text{APSR-R}}$$

$$\boldsymbol{\tau}_{\text{APSR-R}} = \theta_{\text{max}} S_{\text{sensor}} \text{SmoothReLU}(P_{\text{phys}} - P_{\text{phan}}) \mathbf{M}_{\text{aniso}}$$

where:
- $\text{SmoothReLU}(x) = \frac{1}{2} \left( x + \sqrt{x^2 + \delta^2} \right) - \frac{\delta}{2}$ with mollifier parameter $\delta = 10^{-4} P_{\text{phys}}$ is a $C^\infty$ non-negative ReLU operator ($\text{SmoothReLU}(x) \ge 0$ everywhere).
- $\mathbf{M}_{\text{aniso}} = \frac{\nabla P_{\text{phys}} \otimes \nabla P_{\text{phys}}}{|\nabla P_{\text{phys}}|^2 + \epsilon_m}$ is the symmetric shock-structure unit tensor aligned with the local physical pressure gradient $\mathbf{n}_{\text{shock}} = \frac{\nabla P_{\text{phys}}}{|\nabla P_{\text{phys}}|}$.

### 3.2 Rotational Invariance & Interface Traction Vector Assembly
Across an element interface with unit face normal $\mathbf{n}_{\text{face}}$, the phantom stress tensor creates a non-zero normal/shear traction vector:

$$\mathbf{t}_{\text{phan}}^* = \boldsymbol{\tau}_{\text{APSR-R}}^* \cdot \mathbf{n}_{\text{face}} = \theta_{\text{max}} S_{\text{sensor}} \text{SmoothReLU}(P_{\text{phys}}^* - P_{\text{phan}}^*) (\mathbf{n}_{\text{shock}}^* \cdot \mathbf{n}_{\text{face}}) \mathbf{n}_{\text{shock}}^*$$

The combined numerical interface flux vector is assembled as:

$$\mathbf{F}_{\text{interface}} = \mathbf{F}_{\text{Euler}}^{\text{Riemann}}(U_L, U_R; \mathbf{n}_{\text{face}}) + \begin{bmatrix} 0 \\ \mathbf{t}_{\text{phan}}^* \\ \mathbf{t}_{\text{phan}}^* \cdot \mathbf{u}^* \end{bmatrix}$$

where $\mathbf{u}^* = \frac{1}{2}(\mathbf{u}_L + \mathbf{u}_R)$ is the interface contact velocity.

### 3.3 Thermodynamic 2nd Law & Entropy Production Proof
The rate of volumetric mechanical work performed by the anisotropic phantom stress tensor is:

$$\dot{W}_{\text{APSR-R}} = \boldsymbol{\tau}_{\text{APSR-R}} : \nabla \mathbf{u} = \theta_{\text{max}} S_{\text{sensor}} \text{SmoothReLU}(P_{\text{phys}} - P_{\text{phan}}) \left( \mathbf{n}_{\text{shock}} \cdot \nabla \mathbf{u} \cdot \mathbf{n}_{\text{shock}} \right)$$

The fluid entropy production rate obeys:

$$\rho T \frac{D s}{D t} = - \boldsymbol{\tau}_{\text{APSR-R}} : \nabla \mathbf{u} = - \theta_{\text{max}} S_{\text{sensor}} \text{SmoothReLU}(P_{\text{phys}} - P_{\text{phan}}) \left( \mathbf{n}_{\text{shock}} \cdot \nabla \mathbf{u} \cdot \mathbf{n}_{\text{shock}} \right)$$

#### Proof of Non-Negative Entropy Production:
1. **Compressive Shocks ($\nabla \cdot \mathbf{u} < 0$, $\mathbf{n}_{\text{shock}} \cdot \nabla \mathbf{u} \cdot \mathbf{n}_{\text{shock}} < 0$)**:
   Fluid is compressed along the shock normal. Physical pressure exceeds phantom pressure ($P_{\text{phys}} > P_{\text{phan}}$), making $\text{SmoothReLU}(P_{\text{phys}} - P_{\text{phan}}) > 0$.
   Therefore:
   $$\rho T \frac{D s}{D t} > 0 \quad \text{(Strict Positive Dissipation / Entropy Growth)}$$
2. **Expansion Fans / Post-Shock Overshoots ($\nabla \cdot \mathbf{u} > 0$ or $P_{\text{phys}} < P_{\text{phan}}$)**:
   $P_{\text{phys}} - P_{\text{phan}} < 0 \implies \text{SmoothReLU}(P_{\text{phys}} - P_{\text{phan}}) = 0$.
   Therefore:
   $$\boldsymbol{\tau}_{\text{APSR-R}} = \mathbf{0} \implies \rho T \frac{D s}{D t} = 0 \quad \text{(Entropy Neutral, No Anti-Dissipation)}$$

### 3.4 Whitham Anisotropic Sound Speed Bounds
- **Shock-Normal Direction ($\mathbf{n}_{\text{face}} \parallel \mathbf{n}_{\text{shock}}$)**:
  $$(\mathbf{n}_{\text{face}} \cdot \mathbf{n}_{\text{shock}})^2 = 1 \implies a_{\text{frozen}, n} = a_0 \sqrt{1 + \theta_{\text{max}} S_{\text{sensor}} \frac{\text{SmoothReLU}(P_{\text{phys}} - P_{\text{phan}})}{P_{\text{phys}}}}$$
  Sufficient artificial bulk viscosity is generated along the shock normal to eliminate high-order polynomial aliasing oscillations.
- **Transverse / Tangential Direction ($\mathbf{n}_{\text{face}} \perp \mathbf{n}_{\text{shock}}$)**:
  $$(\mathbf{n}_{\text{face}} \cdot \mathbf{n}_{\text{shock}})^2 = 0 \implies a_{\text{frozen}, t} = a_0 \quad \text{(Uninflated Sound Speed!)}$$
  Zero sound speed inflation occurs transverse to the shock normal. Supersonic flow remains supersonic transverse to the shock ($M_{\text{frozen}, t} > 1$), preventing upstream acoustic precursor radiation and standing wave ripples.

### 3.5 Status Notice
**Status**: Tested. Further investigation is needed.

---

## 4. Strategy 3: Shock-Aligned Phantom Transport Velocity ($\mathbf{u}_S$)

### 4.1 Mathematical Formulation
To eliminate phase lag between phantom pressure advection and shock-normal stress, the scalar phantom pressure transport velocity $\mathbf{u}_{\text{adv}}$ in the $S$-evolution equation is modified from fluid velocity $\mathbf{u}$ to a shock-aligned velocity $\mathbf{u}_S$:

$$\frac{\partial S}{\partial t} + \nabla \cdot (S \mathbf{u}_S) = -\frac{1}{\tau} (S - \rho P_{\text{phys}})$$

$$\mathbf{u}_S = (1 - S_{\text{blend}}) \mathbf{u} + S_{\text{blend}} (\mathbf{u} \cdot \mathbf{n}_{\text{shock}}) \mathbf{n}_{\text{shock}}$$

where:
- $\mathbf{n}_{\text{shock}} = \frac{\nabla P_{\text{phys}}}{|\nabla P_{\text{phys}}| + \epsilon_m}$ is the physical pressure gradient unit vector.
- $S_{\text{blend}} = \frac{|\nabla P_{\text{phys}}|^2}{|\nabla P_{\text{phys}}|^2 + \sigma^2}$ is a smooth blending function with threshold $\sigma = 10^{-2} \frac{P_{\text{phys}}}{h_{\text{element}}}$.

### 4.2 Mathematical Rationale
1. **In Uniform Flow / Weak Waves ($|\nabla P_{\text{phys}}| \ll \sigma$)**:
   $S_{\text{blend}} \to 0 \implies \mathbf{u}_S = \mathbf{u}$ (Standard Galilean-invariant material advection along fluid streamlines).
2. **Inside Active Shock Layers ($|\nabla P_{\text{phys}}| \gg \sigma$)**:
   $S_{\text{blend}} \to 1 \implies \mathbf{u}_S = (\mathbf{u} \cdot \mathbf{n}_{\text{shock}}) \mathbf{n}_{\text{shock}}$.
   The tangential velocity component $u_t = \mathbf{u} \cdot \mathbf{t}_{\text{shock}}$ is zeroed out for phantom pressure transport. Phantom pressure non-equilibrium is forced to advect strictly along the shock normal $\mathbf{n}_{\text{shock}}$, aligning phantom scalar advection 100% in sync with anisotropic stress tensor $\boldsymbol{\tau}_{\text{APSR-R}}$.

### 4.3 Spatial Sweep Conservative Implementation
Because $\mathbf{u}_S$ depends on local physical pressure gradients $\nabla P_{\text{phys}}$, scalar flux $F_{\text{sol}, S}$ in 1D sweeps (`sweep_x.cpp`, `sweep_y.cpp`) is evaluated using scalar advection:

$$F_{\text{sol}, S} = u S$$

The shock-aligned transport vector $\mathbf{u}_S$ is applied via regularized advection to avoid introducing non-conservative spatial source terms into Euler conservation laws.

### 4.4 Status Notice
**Status**: Tested. Further investigation is needed.

---

## 5. Strategy 4: Non-Dimensional Divergence Sensor ($\theta_{\text{simple\_div}}$)

### 5.1 Mathematical Formulation
The non-dimensional divergence sensor eliminates empirical shock sensor parameters ($\text{NOISE\_FLOOR}$, $\text{SATURATION}$) by directly coupling non-dimensional velocity divergence to coupling intensity $\theta$:

$$\theta_{\text{max}} = \frac{N_{\text{cells\_shock}} (P+1)}{4 C_\tau}$$

$$\phi_{\text{nd}} = \frac{|\nabla \cdot \mathbf{u}| h_{\text{node}}}{a_{\text{local}}}$$

$$s_{\phi} = \min(1.0, \phi_{\text{nd}})$$

$$\theta_e = \theta_{\text{max}} \Psi_{\text{Ducros}} s_{\phi}$$

where $\Psi_{\text{Ducros}} = \frac{(\nabla \cdot \mathbf{u})^2}{(\nabla \cdot \mathbf{u})^2 + |\nabla \times \mathbf{u}|^2 + \epsilon_{\text{vort}}}$ is the Ducros vorticity filter.

### 5.2 Mathematical Properties
- **Linear Scaling**: Scaling from $0.0$ to $\theta_{\text{max}}$ as non-dimensional divergence $\phi_{\text{nd}}$ ranges from $0.0$ to $1.0$.
- **Stencil Expansion Compatibility**: Neighbor stencil expansion ($\theta_{\text{avg}} = \max_{f} \theta_{\text{neighbor}, f}$) applies cleanly to smooth out spatial $\theta$ variations across element interfaces.

### 5.3 Status Notice
**Status**: Tested. Further investigation is needed.

---

## 6. Strategy 5: Spatial Clamping & Energy Guard

### 6.1 Mathematical Formulation

#### 6.1.1 Spatial Nodal Clamping
To prevent non-equilibrium phantom pressure from producing extreme overshoots near high-order nodal interpolation points, phantom pressure bounds $[P_{\text{phan, min}}, P_{\text{phan, max}}]$ are enforced:

$$P_{\text{phan, max}} = \min\left( P_{\text{nodal, max}}, \left(1 + \frac{1 - C_{\text{pos}}}{\theta + \epsilon}\right) P_{\text{phys}} \right)$$

$$P_{\text{phan, min}} = \max\left( 0, \min\left( P_{\text{nodal, min}}, \left(1 - \frac{C_{\text{max}} - 1}{\theta + \epsilon}\right) P_{\text{phys}} \right) \right)$$

where $P_{\text{nodal, min}}$ and $P_{\text{nodal, max}}$ are local minimum/maximum physical pressure values across the 9-cell patch (element and immediate face neighbors).

#### 6.1.2 Instant-Thermalization Energy Guard
If a cell node experiences anti-dissipative compression ($\theta (P_{\text{phys}} - P_{\text{phan}}) (\nabla \cdot \mathbf{u}) > 0$), phantom pressure is instantly thermalized to physical equilibrium in one Runge-Kutta stage:

$$S_{\text{node}} \leftarrow \rho P_{\text{phys}}$$

### 6.2 Status Notice
**Status**: Tested. Further investigation is needed.

---

## 7. Comparative Summary of Attempted PPR Strategies

| Strategy | Governing Equation Modification | Key Parameters | Primary Intended Function | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Isotropic PPR** | $\mathbf{\Pi} = P_{\text{reg}} \mathbf{I}$, $P_{\text{reg}} = P_{\text{phys}} + \theta(P_{\text{phys}} - P_{\text{phan}})$ | $\theta_{\text{target}}, C_\tau, N_{\text{cells\_shock}}$ | Bulk artificial viscosity for 1D/2D shock smoothing | Tested. Further investigation needed. |
| **2. APSR-R** | $\mathbf{\Pi} = P_{\text{phys}} \mathbf{I} + \theta S_{\text{sensor}} \text{SmoothReLU}(\Delta P) \mathbf{M}_{\text{aniso}}$ | $\theta_{\text{max}}, \text{SmoothReLU}, \mathbf{M}_{\text{aniso}}$ | Shock-normal bulk viscosity, zero transverse sound speed inflation | Tested. Further investigation needed. |
| **3. Transport $\mathbf{u}_S$** | $\partial_t S + \nabla \cdot (S \mathbf{u}_S) = -\frac{1}{\tau} (S - \rho P_{\text{phys}})$ | $\mathbf{u}_S, S_{\text{blend}}, \sigma$ | Align scalar transport with shock-normal tensor | Tested. Further investigation needed. |
| **4. Simple Div Sensor** | $\theta_e = \theta_{\text{max}} \Psi_{\text{Ducros}} \min(1, \phi_{\text{nd}})$ | $\theta_{\text{max}}, \phi_{\text{nd}}$ | Non-dimensional parameterless compression sensor | Tested. Further investigation needed. |
| **5. Spatial Clamping** | $P_{\text{phan}} \leftarrow \text{clamp}(P_{\text{phan}}, P_{\text{min}}, P_{\text{max}})$ | $C_{\text{pos}}, C_{\text{max}}, P_{\text{nodal}}$ | Bound non-equilibrium overshoots near nodal points | Tested. Further investigation needed. |

---

## 8. Recommendations for Future Research

1. **Multi-Dimensional Wave-Structure Coupling**:
   Explore multi-wave Riemann solver integrations where anisotropic phantom traction $\mathbf{t}_{\text{phan}}^*$ directly modulates wave speeds in HLLC / Roe Riemann sweeps.
2. **Space-Time Discontinuous Galerkin (DG) Formulations**:
   Investigate space-time DG discretizations of non-equilibrium scalar relaxation to evaluate whether implicit time integration suppresses high-wavenumber dispersion saturation.
3. **Entropy-Stable Viscous Liftings**:
   Incorporate BR2 (Bassi-Rebay 2) compact lifting operators into phantom traction face gradients to evaluate interface gradient continuity across AMR non-conforming block boundaries.
