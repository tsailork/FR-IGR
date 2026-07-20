# Phantom Pressure Regularization (PPR) - Advanced Theoretical Ideas & Alternative Formulations

This document records alternative mathematical formulations and physical concepts for expanding the Phantom Pressure Regularization (PPR) method in the 2D high-order Flux Reconstruction solver. 

---

## Executive Summary & Background

In standard PPR, the phantom pressure state variable $S = \rho P_{phan}$ is advected as a passive scalar using the local fluid velocity $\mathbf{u} = (u, v)$:
\[
\frac{\partial S}{\partial t} + \nabla \cdot (S \mathbf{u}) = \frac{S^{eq} - S}{\tau}
\]

### The Directionality Problem
Because fluid discontinuities and shock waves propagate at **shock speed $W_s$** (governed by acoustic characteristic speeds $u \pm a$) rather than fluid velocity $\mathbf{u}$:
1. **Downstream Shocks ($W_s \cdot \mathbf{u} > 0$):** Fluid velocity carries $S$ into and across the shock front, creating effective pre-conditioning and smoothing of the regularized pressure $P_{reg} = P_{phys} + \theta (P_{phys} - P_{phan})$.
2. **Upstream / Standing Shocks ($W_s \cdot \mathbf{u} \le 0$):** Fluid velocity $\mathbf{u}$ sweeps $S$ *downstream*, moving phantom pressure away from the incoming shock front. The regularized pressure $P_{reg}$ at the leading shock edge lacks adequate pre-conditioning, causing localized Gibbs oscillations or solver instability.

The following three options provide potential long-term solutions for future developers.

---

## Option 2: Characteristic Wave-Speed Guided Advection

### Concept & Mathematical Formulation
Instead of advecting $S$ strictly along fluid streamlines $\mathbf{u}$, modify the advection velocity vector $\mathbf{u}_{adv}$ to include an acoustic characteristic component directed along the local pressure gradient $\nabla P$:
\[
\mathbf{u}_{adv} = \mathbf{u} - a \, \hat{\mathbf{n}}_{\nabla P}
\]
where:
* $a = \sqrt{\gamma P / \rho}$ is the local speed of sound.
* $\hat{\mathbf{n}}_{\nabla P} = \frac{\nabla P}{\|\nabla P\| + \epsilon}$ is the unit normal vector pointing towards higher pressure (into the shock front).
* $\epsilon > 0$ is a small regularization constant to prevent division by zero in smooth flow regions.

### Governing Equation
\[
\frac{\partial S}{\partial t} + \nabla \cdot \left( S \left[ \mathbf{u} - a \, \frac{\nabla P}{\|\nabla P\| + \epsilon} \right] \right) = \frac{S^{eq} - S}{\tau}
\]

### Advantages & Implementation Notes
* **Physical Alignment:** Directly aligns phantom pressure advection with the direction of acoustic wave propagation and compression gradients.
* **Shock-Targeted Transport:** Automatically drives $S$ upstream into incoming shock waves, regardless of whether the flow is subsonic or supersonic.
* **Code Integration:** Requires evaluating the element-local pressure gradient $\nabla P$ (which can be obtained via Lagrange derivative matrix $D$).

---

## Option 3: Isotropic Spatial Diffusion (Diffusion-Augmented PPR)

### Concept & Mathematical Formulation
Augment or replace the hyper-directional advection term with a spatial diffusion operator $\nu_S \nabla^2 S$:
\[
\frac{\partial S}{\partial t} + \nabla \cdot (S \mathbf{u}) = \nabla \cdot (\nu_S \nabla S) + \frac{S^{eq} - S}{\tau}
\]
where $\nu_S = C_{\nu} h \cdot \max(0, -\nabla \cdot \mathbf{u})$ is a localized, resolution-dependent diffusion coefficient active only in compressive flow regions ($\nabla \cdot \mathbf{u} < 0$).

### Advantages & Implementation Notes
* **Isotropic Smoothing:** Laplacian diffusion $\nabla^2 S$ spreads phantom pressure equally in all spatial directions (upstream, downstream, transverse), completely eliminating flow-velocity directional bias.
* **Decoupled Helmholtz / BR2 Solver:** Can be discretized using the existing Bassi-Rebay 2 (BR2) parabolic framework used for IGR in `parabolic.cpp`.

---

## Option 4: Local Shock-Sensor Adaptive Multiplier

### Concept & Mathematical Formulation
Make the advection velocity multiplier $\kappa$ a dynamic function of the local flow compression $\nabla \cdot \mathbf{u}$ or pressure jump $J_P$:
\[
\mathbf{u}_{adv} = \kappa(\nabla \cdot \mathbf{u}) \, \mathbf{u}
\]
where:
\[
\kappa(\nabla \cdot \mathbf{u}) = \begin{cases}
1.0 & \text{if } \nabla \cdot \mathbf{u} \ge 0 \quad (\text{smooth or expanding flow}) \\
1.0 - \chi \cdot \tanh\left( \frac{-\nabla \cdot \mathbf{u}}{\omega_0} \right) & \text{if } \nabla \cdot \mathbf{u} < 0 \quad (\text{compressive flow / shock})
\end{cases}
\]
with tuning parameter $\chi \in [1.0, 2.0]$.

### Advantages & Implementation Notes
* **Targeted Intervention:** Preserves physical passive scalar transport ($\kappa = 1.0$) in smooth flow fields while automatically dampening or reversing advection ($\kappa \le 0.0$) at shock locations.
* **Low Overhead:** Computes $\nabla \cdot \mathbf{u}$ using existing velocity gradient routines in `gradient.cpp`.
