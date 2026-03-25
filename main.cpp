#include "solver.hpp"
#include "outputs.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

// Sigmoid function for smoothing transitions
inline double sigmoid(double x, double x0, double delta) {
    return 1.0 / (1.0 + std::exp(-(x - x0) / delta));
}

int main() {
    // 1. Load Parameters
    Parameters params;
    params.load_from_file("inputs.dat");

    // 2. Setup Solver
    Solver solver(params);
    std::cout << "Initializing 2D Riemann Problem (Configuration 3) with smoothed ICs...\n";

    // 3. Initialize Output Directory and Writer
    ensure_output_directory("pv_outputs");
    VTKWriter writer("solution");

    // 4. Initial Conditions (Smoothed)
    double delta = 0.5 * std::min(solver.dx, solver.dy);
    for(int ey=0; ey<params.N_ELEM_Y; ++ey) {
        for(int ex=0; ex<params.N_ELEM_X; ++ex) {
            for(int iy=0; iy<params.N_PTS; ++iy) {
                for(int ix=0; ix<params.N_PTS; ++ix) {
                    double x = params.X_MIN + (ex + 0.5*(1+solver.basis.z[ix])) * solver.dx;
                    double y = params.Y_MIN + (ey + 0.5*(1+solver.basis.z[iy])) * solver.dy;

                    // Centered at 0.5, 0.5
                    double x0 = 0.5, y0 = 0.5;
                    
                    // Quadrant values
                    double rTR = 1.5,    uTR = 0.0,   vTR = 0.0,   pTR = 1.5;
                    double rTL = 0.5323, uTL = 1.206, vTL = 0.0,   pTL = 0.3;
                    double rBL = 0.138,  uBL = 1.206, vBL = 1.206, pBL = 0.029;
                    double rBR = 0.5323, uBR = 0.0,   vBR = 1.206, pBR = 0.3;

                    // Blending weights
                    double wx = sigmoid(x, x0, delta);
                    double wy = sigmoid(y, y0, delta);

                    // Interpolate values
                    double rho = (1-wy)*((1-wx)*rBL + wx*rBR) + wy*((1-wx)*rTL + wx*rTR);
                    double u   = (1-wy)*((1-wx)*uBL + wx*uBR) + wy*((1-wx)*uTL + wx*uTR);
                    double v   = (1-wy)*((1-wx)*vBL + wx*vBR) + wy*((1-wx)*vTL + wx*vTR);
                    double p   = (1-wy)*((1-wx)*pBL + wx*pBR) + wy*((1-wx)*pTL + wx*pTR);

                    solver.U(0, ey, ex, iy, ix) = rho;
                    solver.U(1, ey, ex, iy, ix) = rho*u;
                    solver.U(2, ey, ex, iy, ix) = rho*v;
                    solver.U(3, ey, ex, iy, ix) = p/(params.GAMMA-1.0) + 0.5*rho*(u*u+v*v);
                }
            }
        }
    }

    // 5. Time Stepping Loop
    double t = 0.0;
    int step = 0;
    double next_output = 0.0;
    int output_count = 0;

    // Output Initial State
    writer.write_snapshot(solver, output_count++, t);
    next_output += params.OUTPUT_DT;
    
    while(t < params.T_FINAL) {
        // Dynamic time step calculation based on CFL
        double dt = solver.compute_dt();
        
        // Ensure we don't overshoot Output or T_FINAL
        if (t + dt > next_output) dt = next_output - t;
        if (t + dt > params.T_FINAL) dt = params.T_FINAL - t;

        solver.step_rk3(dt);
        t += dt;
        step++;
        
        if (std::abs(t - next_output) < 1e-12) {
            std::cout << "Step " << step << " t=" << t << " dt=" << dt << "\n";
            writer.write_snapshot(solver, output_count++, t);
            next_output += params.OUTPUT_DT;
        }
    }

    return 0;
}
