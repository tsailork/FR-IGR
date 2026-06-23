/**
 * @file sbm_geometry.cpp
 * @brief Implementations of shifted boundary face detection, 1D normal ray stencils, and state extrapolations.
 */

#include "sbm_geometry.hpp"
#include "../core/solver.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>

namespace ImmersedBoundary {

std::vector<SurrogateFluxPoint> sbm_registry;
std::unordered_map<uint64_t, const SurrogateFluxPoint*> sbm_lookup;

SBMDiagnostics current_sbm_diags;

void reset_sbm_diagnostics() {
    double saved_dist_ratio = current_sbm_diags.max_dist_ratio;
    double saved_d_dl_ratio = current_sbm_diags.max_d_dl_ratio;
    double saved_lebesgue   = current_sbm_diags.max_lebesgue;
    
    current_sbm_diags = SBMDiagnostics();
    
    current_sbm_diags.max_dist_ratio = saved_dist_ratio;
    current_sbm_diags.max_d_dl_ratio = saved_d_dl_ratio;
    current_sbm_diags.max_lebesgue   = saved_lebesgue;
}

SBMDiagnostics get_sbm_diagnostics() {
    return current_sbm_diags;
}

inline uint64_t get_sbm_hash(int b_id, int ey, int ex, int face, int node) {
    uint64_t h = 0;
    h |= ((uint64_t)b_id & 0xFFFF) << 48;
    h |= ((uint64_t)ey & 0xFFFF) << 32;
    h |= ((uint64_t)ex & 0xFFFF) << 16;
    h |= ((uint64_t)face & 0xF) << 8;
    h |= ((uint64_t)node & 0xFF);
    return h;
}

// Helper to evaluate 1D Lagrange polynomial basis weights
void compute_lagrange_weights(int P, double xi_eval, const std::vector<double>& nodes, std::vector<double>& weights) {
    weights.assign(P + 1, 0.0);
    double sum_abs_weights = 0.0;
    for (int i = 0; i <= P; ++i) {
        double L = 1.0;
        for (int j = 0; j <= P; ++j) {
            if (i != j) {
                L *= (xi_eval - nodes[j]) / (nodes[i] - nodes[j]);
            }
        }
        weights[i] = L;
        sum_abs_weights += std::abs(L);
    }
    
    #pragma omp critical
    {
        if (sum_abs_weights > current_sbm_diags.max_lebesgue) {
            current_sbm_diags.max_lebesgue = sum_abs_weights;
        }
    }
}

void initialize_sbm_geometry(Solver& solver) {
    if (!solver.p.ENABLE_IB) return;
    
    sbm_registry.clear();
    std::cout << "[SBM] Initializing Shifted Boundary Geometry..." << std::endl;
    
    int total_sbm_faces = 0;

    for (int b_idx = 0; b_idx < solver.blocks.size(); ++b_idx) {
        Block& b = solver.blocks[b_idx];
        
        // 3D arrays: [ey][ex][face]
        std::vector<std::vector<std::vector<bool>>> prelim(b.ny, std::vector<std::vector<bool>>(b.nx, std::vector<bool>(4, false)));
        std::vector<std::vector<bool>> fully_fluid(b.ny, std::vector<bool>(b.nx, false));
        std::vector<std::vector<bool>> fully_solid(b.ny, std::vector<bool>(b.nx, false));
        
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                // f_inside[face] = true iff ALL quadrature nodes on that face lie inside the body (phi < 0)
                bool f_inside[4] = {true, true, true, true};
                
                int num_pts_inside = 0;
                int num_pts_total = (solver.p.N_PTS + 2) * (solver.p.N_PTS + 2);
                
                // Check all volume DOFs and element corners/boundaries for classification
                for (int iy = 0; iy <= solver.p.N_PTS + 1; ++iy) {
                    double zy = (iy == 0) ? -1.0 : (iy == solver.p.N_PTS + 1) ? 1.0 : solver.basis.z[iy - 1];
                    for (int ix = 0; ix <= solver.p.N_PTS + 1; ++ix) {
                        double zx = (ix == 0) ? -1.0 : (ix == solver.p.N_PTS + 1) ? 1.0 : solver.basis.z[ix - 1];
                        double x = b.x_min + (ex + 0.5 * (1.0 + zx)) * b.dx;
                        double y = b.y_min + (ey + 0.5 * (1.0 + zy)) * b.dy;
                        if (solver.get_ib_sdf_at_time(x, y, 0.0) < 0) {
                            num_pts_inside++;
                        }
                    }
                }
                
                if (num_pts_inside == 0) fully_fluid[ey][ex] = true;
                if (num_pts_inside == num_pts_total) fully_solid[ey][ex] = true;

                // Tag face j as "all-inside" if every GL node AND the corners on it have phi < 0.
                // Faces: 0=Left, 1=Right, 2=Bottom, 3=Top.
                for (int iy = 0; iy <= solver.p.N_PTS + 1; ++iy) {
                    double z = (iy == 0) ? -1.0 : (iy == solver.p.N_PTS + 1) ? 1.0 : solver.basis.z[iy - 1];
                    double y = b.y_min + (ey + 0.5 * (1.0 + z)) * b.dy;
                    if (solver.get_ib_sdf_at_time(b.x_min + ex * b.dx,       y, 0.0) >= 0.0) f_inside[0] = false;
                    if (solver.get_ib_sdf_at_time(b.x_min + (ex+1) * b.dx,   y, 0.0) >= 0.0) f_inside[1] = false;
                }
                for (int ix = 0; ix <= solver.p.N_PTS + 1; ++ix) {
                    double z = (ix == 0) ? -1.0 : (ix == solver.p.N_PTS + 1) ? 1.0 : solver.basis.z[ix - 1];
                    double x = b.x_min + (ex + 0.5 * (1.0 + z)) * b.dx;
                    if (solver.get_ib_sdf_at_time(x, b.y_min + ey * b.dy,       0.0) >= 0.0) f_inside[2] = false;
                    if (solver.get_ib_sdf_at_time(x, b.y_min + (ey+1) * b.dy,   0.0) >= 0.0) f_inside[3] = false;
                }
                
                // Step 2: Preliminary shifted-face criterion (Colombo Sec. 3.1)
                // A face is preliminary-shifted iff ALL nodes on F are inside the body.
                prelim[ey][ex][0] = f_inside[0];
                prelim[ey][ex][1] = f_inside[1];
                prelim[ey][ex][2] = f_inside[2];
                prelim[ey][ex][3] = f_inside[3];

                // Constraint 3.a: The OPPOSITE face is NOT entirely inside.
                // If an element is fully inside the body, both faces will be unmarked.
                prelim[ey][ex][0] = prelim[ey][ex][0] && !f_inside[1];
                prelim[ey][ex][1] = prelim[ey][ex][1] && !f_inside[0];
                prelim[ey][ex][2] = prelim[ey][ex][2] && !f_inside[3];
                prelim[ey][ex][3] = prelim[ey][ex][3] && !f_inside[2];
            }
        }
        
        // Constraint 3.b: Remove tag if face is tagged twice
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx - 1; ++ex) {
                if (prelim[ey][ex][1] && prelim[ey][ex+1][0]) {
                    prelim[ey][ex][1] = false;
                    prelim[ey][ex+1][0] = false;
                }
            }
        }
        for (int ex = 0; ex < b.nx; ++ex) {
            for (int ey = 0; ey < b.ny - 1; ++ey) {
                if (prelim[ey][ex][3] && prelim[ey+1][ex][2]) {
                    prelim[ey][ex][3] = false;
                    prelim[ey+1][ex][2] = false;
                }
            }
        }
        
        // Constraint 3.c: Remove tags if cell has two opposite faces flagged by adjacent cells
        if (solver.p.ENABLE_IB_3C) {
            for (int ey = 0; ey < b.ny; ++ey) {
                for (int ex = 0; ex < b.nx; ++ex) {
                    bool l_flagged = (ex > 0) && prelim[ey][ex-1][1];
                    bool r_flagged = (ex < b.nx - 1) && prelim[ey][ex+1][0];
                    if (l_flagged && r_flagged) {
                        prelim[ey][ex-1][1] = false;
                        prelim[ey][ex+1][0] = false;
                    }
                    
                    bool b_flagged = (ey > 0) && prelim[ey-1][ex][3];
                    bool t_flagged = (ey < b.ny - 1) && prelim[ey+1][ex][2];
                    if (b_flagged && t_flagged) {
                        prelim[ey-1][ex][3] = false;
                        prelim[ey+1][ex][2] = false;
                    }
                }
            }
        }

        // Synchronization / Union Pass
        // If a fluid/mixed element needs a shifted face, ensure the interface is marked.
        // We do a logical OR across the interface: if Cell A marks its Right face,
        // we keep that mark. If Cell B marks its Left face, we keep that mark.
        // Since 3.b removed double-marks, this just ensures the ownership is solid.
        // (No explicit code needed here as the arrays already store directional flags correctly,
        // but this serves as a conceptual marker for the step requested).
        
        // ----------------------------------------------------------------
        // Populate solid_mask and initialise solid cells to freestream.
        // Elements that are FULLY inside the IB body are "blanked":
        //   - Their state is set to the freestream conserved variables.
        //   - compute_rhs() will zero their RHS so they are never updated.
        // ----------------------------------------------------------------
        b.solid_mask.assign(b.ny * b.nx, false);

        // Freestream conserved state
        const double gm1   = solver.p.GAMMA - 1.0;
        const double rho_f = solver.p.RHO_INF;
        const double rhou_f = rho_f * solver.p.U_INF;
        const double rhov_f = rho_f * solver.p.V_INF;
        const double E_f   = solver.p.P_INF / gm1
                           + 0.5 * rho_f * (solver.p.U_INF * solver.p.U_INF
                                           + solver.p.V_INF * solver.p.V_INF);

        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                if (fully_solid[ey][ex]) {
                    b.solid_mask[ey * b.nx + ex] = true;
                    // Override all DOFs to freestream
                    for (int iy = 0; iy < solver.p.N_PTS; ++iy)
                        for (int ix = 0; ix < solver.p.N_PTS; ++ix) {
                            b.U(0, ey, ex, iy, ix) = rho_f;
                            b.U(1, ey, ex, iy, ix) = 0.0;
                            b.U(2, ey, ex, iy, ix) = 0.0;
                            b.U(3, ey, ex, iy, ix) = E_f;
                        }
                }
            }
        }

        // Output Diagnostic File
        std::string diag_filename = "csv_outputs/sbm_diag_block" + std::to_string(b.id) + ".csv";
        std::ofstream diag(diag_filename);
        diag << "ey,ex,type,is_shifted,L,R,B,T,xc,yc,sdf\n";
        
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                std::string type = "mixed";
                if (fully_fluid[ey][ex]) type = "fluid";
                if (fully_solid[ey][ex]) type = "solid";
                
                bool is_shifted = prelim[ey][ex][0] || prelim[ey][ex][1] || prelim[ey][ex][2] || prelim[ey][ex][3];
                double xc = b.x_min + (ex + 0.5) * b.dx;
                double yc = b.y_min + (ey + 0.5) * b.dy;
                double sdf = solver.get_ib_sdf_at_time(xc, yc, 0.0);
                
                diag << ey << "," << ex << "," << type << "," << is_shifted << ","
                     << prelim[ey][ex][0] << "," << prelim[ey][ex][1] << ","
                     << prelim[ey][ex][2] << "," << prelim[ey][ex][3] << ","
                     << xc << "," << yc << "," << sdf << "\n";
            }
        }
        
        // Generate SBM Registry & ray reconstruction
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int f = 0; f < 4; ++f) {
                    if (prelim[ey][ex][f]) {
                        for (int k = 0; k < solver.p.N_PTS; ++k) {
                            SurrogateFluxPoint sfp;
                            sfp.block_id = b.id;
                            sfp.element_x = ex;
                            sfp.element_y = ey;
                            sfp.face_idx = f;
                            sfp.node_idx = k;
                            
                            // Find physical coordinate of this surrogate point
                            double px = 0.0, py = 0.0;
                            if (f == 0) { // L
                                px = b.x_min + ex * b.dx;
                                py = b.y_min + (ey + 0.5 * (1.0 + solver.basis.z[k])) * b.dy;
                            } else if (f == 1) { // R
                                px = b.x_min + (ex + 1) * b.dx;
                                py = b.y_min + (ey + 0.5 * (1.0 + solver.basis.z[k])) * b.dy;
                            } else if (f == 2) { // B
                                px = b.x_min + (ex + 0.5 * (1.0 + solver.basis.z[k])) * b.dx;
                                py = b.y_min + ey * b.dy;
                            } else if (f == 3) { // T
                                px = b.x_min + (ex + 0.5 * (1.0 + solver.basis.z[k])) * b.dx;
                                py = b.y_min + (ey + 1) * b.dy;
                            }
                            
                            // Iterate to find physical boundary point
                            // Start at surrogate point, walk along gradient
                            double curr_x = px;
                            double curr_y = py;
                            double nx = 0, ny = 0;
                            double dist = solver.get_ib_sdf_at_time(curr_x, curr_y, 0.0);
                            sfp.D = std::abs(dist); // Distance is magnitude
                            
                            for (int iter = 0; iter < 10; ++iter) {
                                solver.get_ib_sdf_gradient_at_time(curr_x, curr_y, 0.0, nx, ny);
                                double sdf = solver.get_ib_sdf_at_time(curr_x, curr_y, 0.0);
                                if (std::abs(sdf) < 1e-6) break;
                                curr_x -= sdf * nx;
                                curr_y -= sdf * ny;
                            }
                            
                            // The ray points inwards to the fluid: hat_ns
                            double ns_x = nx;
                            double ns_y = ny;
                            // Ensure ns points into the fluid (SDF > 0). If SDF increases along ns, good.
                            solver.get_ib_sdf_gradient_at_time(curr_x + 1e-4 * ns_x, curr_y + 1e-4 * ns_y, 0.0, nx, ny);
                            double sdf_test = solver.get_ib_sdf_at_time(curr_x + 1e-4 * ns_x, curr_y + 1e-4 * ns_y, 0.0);
                            if (sdf_test < 0) {
                                ns_x = -ns_x;
                                ns_y = -ns_y;
                            }
                            
                            sfp.nx_true = ns_x;
                            sfp.ny_true = ns_y;
                            
                            // ----------------------------------------------------------------
                            // Lagrange interpolation stencil (Colombo Sec. 3.2)
                            //
                            // Arc-length psi measured from the SHIFTED BOUNDARY going into fluid:
                            //   psi = 0            : shifted boundary      *** EVALUATION POINT ***
                            //   psi = D            : physical boundary      *** NODE 0 (BC) ***
                            //   psi in [L, L+dL]   : P_interp donor GL pts *** NODES 1..P_interp ***
                            //
                            //   P_interp = P_soln + 1                 (one order higher than solution)
                            //   alpha    = 1 + D/h                    (local, Colombo notation)
                            //   L        = D + alpha*sqrt(2)*h        (START of donor interval)
                            //   dL       = h * (z[1] - z[0])         (interval length)
                            //
                            // Donor GL nodes basis.z[j-1] (j=1..P_interp) mapped [-1,1] -> [L, L+dL]:
                            //   psi[j] = L + 0.5*dL*(1 + basis.z[j-1])
                            //   (z=-1 => psi=L, z=+1 => psi=L+dL)
                            //
                            // Polynomial is evaluated BACKWARD to psi=0 (extrapolation).
                            // ----------------------------------------------------------------
                            const int P_soln   = std::max(1, solver.p.P_DEG);
                            const int P_interp = P_soln + 1;  // one order higher than solution
                            
                            double h     = std::max(b.dx, b.dy);
                            double alpha = 1.0 + sfp.D / h;
                            double L     = sfp.D + solver.p.IB_L_SCALE * alpha * std::sqrt(2.0) * h;
                            double dL    = solver.p.IB_DL_SCALE * h * std::sqrt(2.0);
                            
                            double ratio = sfp.D / L;
                            double d_dl_ratio = sfp.D / dL;
                            #pragma omp critical
                            {
                                if (ratio > current_sbm_diags.max_dist_ratio) {
                                    current_sbm_diags.max_dist_ratio = ratio;
                                }
                                if (d_dl_ratio > current_sbm_diags.max_d_dl_ratio) {
                                    current_sbm_diags.max_d_dl_ratio = d_dl_ratio;
                                }
                            }
                            
                            std::vector<double> psi(P_interp + 1);
                            psi[0] = sfp.D;  // physical boundary node
                            for (int j = 1; j <= P_interp; ++j) {
                                psi[j] = L + 0.5 * dL * (1.0 + solver.basis.z[j - 1]);
                            }
                            
                            sfp.donor_points.resize(P_interp);
                            for (int j = 1; j <= P_interp; ++j) {
                                double ray_x = curr_x + psi[j] * ns_x;
                                double ray_y = curr_y + psi[j] * ns_y;
                                
                                int dex = (int)std::floor((ray_x - b.x_min) / b.dx);
                                int dey = (int)std::floor((ray_y - b.y_min) / b.dy);
                                dex = std::max(0, std::min(b.nx - 1, dex));
                                dey = std::max(0, std::min(b.ny - 1, dey));
                                
                                DonorPoint dp;
                                dp.b_id = b.id;
                                dp.ex   = dex;
                                dp.ey   = dey;
                                dp.xi   = 2.0 * (ray_x - (b.x_min + dex * b.dx)) / b.dx - 1.0;
                                dp.eta  = 2.0 * (ray_y - (b.y_min + dey * b.dy)) / b.dy - 1.0;
                                sfp.donor_points[j - 1] = dp;
                            }
                            
                            // Weights evaluated at psi=0: u_sb = l[0]*u_Gamma + sum_j l[j]*u_donor[j-1]
                            compute_lagrange_weights(P_interp, 0.0, psi, sfp.l_weights);
                            sbm_registry.push_back(sfp);
                        }
                        total_sbm_faces++;
                    }
                }
            }
        }
        
        // ----------------------------------------------------------------
        // Output Shifted Faces VTK PolyData (.vtp) for easy visualization
        // (Even though it's .vtp, putting it in csv_outputs as requested)
        std::string vtp_filename = "csv_outputs/sbm_faces_block" + std::to_string(b.id) + ".vtp";
        std::ofstream vtp(vtp_filename);
        int num_faces = 0;
        for(int ey = 0; ey < b.ny; ++ey)
            for(int ex = 0; ex < b.nx; ++ex)
                for(int f = 0; f < 4; ++f)
                    if(prelim[ey][ex][f]) num_faces++;
        
        vtp << "<?xml version=\"1.0\"?>\n";
        vtp << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        vtp << "  <PolyData>\n";
        vtp << "    <Piece NumberOfPoints=\"" << num_faces * 2 << "\" NumberOfLines=\"" << num_faces << "\">\n";
        vtp << "      <Points>\n";
        vtp << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int ey = 0; ey < b.ny; ++ey) {
            for (int ex = 0; ex < b.nx; ++ex) {
                for (int f = 0; f < 4; ++f) {
                    if (prelim[ey][ex][f]) {
                        double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                        if (f == 0) { // Left
                            x1 = x2 = b.x_min + ex * b.dx;
                            y1 = b.y_min + ey * b.dy;
                            y2 = b.y_min + (ey + 1) * b.dy;
                        } else if (f == 1) { // Right
                            x1 = x2 = b.x_min + (ex + 1) * b.dx;
                            y1 = b.y_min + ey * b.dy;
                            y2 = b.y_min + (ey + 1) * b.dy;
                        } else if (f == 2) { // Bottom
                            y1 = y2 = b.y_min + ey * b.dy;
                            x1 = b.x_min + ex * b.dx;
                            x2 = b.x_min + (ex + 1) * b.dx;
                        } else if (f == 3) { // Top
                            y1 = y2 = b.y_min + (ey + 1) * b.dy;
                            x1 = b.x_min + ex * b.dx;
                            x2 = b.x_min + (ex + 1) * b.dx;
                        }
                        vtp << x1 << " " << y1 << " 0.0  " << x2 << " " << y2 << " 0.0\n";
                    }
                }
            }
        }
        vtp << "        </DataArray>\n";
        vtp << "      </Points>\n";
        vtp << "      <Lines>\n";
        vtp << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < num_faces; ++i) vtp << (2 * i) << " " << (2 * i + 1) << "\n";
        vtp << "        </DataArray>\n";
        vtp << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 0; i < num_faces; ++i) vtp << (2 * i + 2) << "\n";
        vtp << "        </DataArray>\n";
        vtp << "      </Lines>\n";
        vtp << "    </Piece>\n";
        vtp << "  </PolyData>\n";
        vtp << "</VTKFile>\n";
    }
    
    // Populate lookup map
    sbm_lookup.clear();
    for (const auto& sfp : sbm_registry) {
        sbm_lookup[get_sbm_hash(sfp.block_id, sfp.element_y, sfp.element_x, sfp.face_idx, sfp.node_idx)] = &sfp;
    }
    
    std::cout << "[SBM] Registered " << total_sbm_faces << " surrogate faces." << std::endl;
}

const SurrogateFluxPoint* get_sbm_face(int b_id, int ey, int ex, int face, int node) {
    auto it = sbm_lookup.find(get_sbm_hash(b_id, ey, ex, face, node));
    if (it != sbm_lookup.end()) return it->second;
    return nullptr;
}

void compute_sbm_state(const Solver& solver, const SurrogateFluxPoint* sfp, double u_sb[4]) {
    if (sfp->donor_points.empty()) return;
    
    // Evaluate the 2D solution polynomial at a donor point (dp.xi, dp.eta) using
    // the background element's solution DOFs.
    auto evaluate_solution = [&](const DonorPoint& dp, double u_out[4]) {
        const Block& b = solver.blocks[dp.b_id];
        const int P = solver.p.P_DEG;
        const int N = solver.p.N_PTS;
        std::vector<double> L_xi(N), L_eta(N);
        compute_lagrange_weights(P, dp.xi, solver.basis.z, L_xi);
        compute_lagrange_weights(P, dp.eta, solver.basis.z, L_eta);
        
        for (int v = 0; v < 4; ++v) u_out[v] = 0.0;
        for (int iy = 0; iy < N; ++iy)
            for (int ix = 0; ix < N; ++ix) {
                double w = L_xi[ix] * L_eta[iy];
                for (int v = 0; v < 4; ++v)
                    u_out[v] += b.U(v, dp.ey, dp.ex, iy, ix) * w;
            }
    };
    
    // Use the nearest donor (donor_points[0]) to estimate density and pressure
    // at the physical boundary. Both Dirichlet and Neumann wall conditions
    // extrapolate rho and p from the adjacent fluid state.
    double u_dp0[4];
    evaluate_solution(sfp->donor_points[0], u_dp0);
    
    const double gm1   = solver.p.GAMMA - 1.0;
    const double rho_b = u_dp0[0];
    const double u_vel = u_dp0[1] / rho_b;
    const double v_vel = u_dp0[2] / rho_b;
    const double p_b   = gm1 * (u_dp0[3] - 0.5 * rho_b * (u_vel*u_vel + v_vel*v_vel));
    
    // ----------------------------------------------------------------
    // Construct wall boundary state u_b at the PHYSICAL BOUNDARY (psi=D)
    //
    // Inviscid / Euler  (ENABLE_NS == false):  Neumann slip wall
    //   - Zero NORMAL velocity (u·n = 0), tangential velocity free.
    //   - Density and pressure: zeroth-order extrapolation from nearest donor.
    //
    // Viscous / NS      (ENABLE_NS == true):   Dirichlet no-slip wall
    //   - Zero velocity: u_wall = v_wall = 0 (no-slip + no-penetration).
    //   - Density and pressure: zeroth-order extrapolation from nearest donor.
    //   - Internal energy: E = p/(gamma-1)  (zero kinetic energy).
    // ----------------------------------------------------------------
    double u_b[4];
    u_b[0] = rho_b;
    
    if (solver.p.ENABLE_NS) {
        // Dirichlet: full no-slip -- both velocity components are zero at the wall.
        u_b[1] = 0.0;
        u_b[2] = 0.0;
        u_b[3] = p_b / gm1;   // zero kinetic energy
    } else {
        // Neumann slip wall: project out the normal component of velocity.
        const double v_n      = u_vel * sfp->nx_true + v_vel * sfp->ny_true;
        const double u_tang   = u_vel - v_n * sfp->nx_true;
        const double v_tang   = v_vel - v_n * sfp->ny_true;
        u_b[1] = rho_b * u_tang;
        u_b[2] = rho_b * v_tang;
        u_b[3] = p_b / gm1 + 0.5 * rho_b * (u_tang*u_tang + v_tang*v_tang);
    }
    
    // ----------------------------------------------------------------
    // 1D Lagrange extrapolation to the SHIFTED BOUNDARY (psi=0):
    //   u_sb = l_weights[0]*u_b + sum_{j=1}^{P_interp} l_weights[j]*u_donor[j-1]
    // ----------------------------------------------------------------
    for (int v = 0; v < 4; ++v) u_sb[v] = u_b[v] * sfp->l_weights[0];
    
    for (size_t j = 0; j < sfp->donor_points.size(); ++j) {
        double u_dp[4];
        evaluate_solution(sfp->donor_points[j], u_dp);
        for (int v = 0; v < 4; ++v)
            u_sb[v] += u_dp[v] * sfp->l_weights[j + 1];
    }
    
    // ----------------------------------------------------------------
    // Pressure & Density Limiter for Extrapolated State
    // ----------------------------------------------------------------
    if (u_sb[0] < solver.p.POS_LIMITER_EPS) {
        u_sb[0] = solver.p.POS_LIMITER_EPS;
        solver.sbm_nonphysical_count++;
        #pragma omp atomic
        current_sbm_diags.limiter_count++;
    }
    
    double press = (solver.p.GAMMA - 1.0) * (u_sb[3] - 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / u_sb[0]);
    if (press < solver.p.POS_LIMITER_EPS) {
        solver.sbm_nonphysical_count++;
        #pragma omp atomic
        current_sbm_diags.limiter_count++;
        u_sb[3] = solver.p.POS_LIMITER_EPS / (solver.p.GAMMA - 1.0) + 0.5 * (u_sb[1]*u_sb[1] + u_sb[2]*u_sb[2]) / u_sb[0];
    }
}

} // namespace ImmersedBoundary
