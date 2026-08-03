} else if (p.IC_TYPE == "SHOCK_VORTEX_WORKSHOP" || p.IC_TYPE == "SHOCK_VORTEX_HIOCFD") {
    // 2D Shock-Vortex Interaction (HiOCFD Case CI2 / Inoue & Hattori 1999)
    const double xs    = 0.5;   // Standing shock position
    const double xv    = 0.25;  // Vortex center X
    const double yv    = 0.5;   // Vortex center Y
    const double ra    = 0.075; // Inner core radius
    const double rb    = 0.175; // Outer core radius
    const double Mv    = 0.9;   // Max tangential Mach number
    const double Ms    = 1.5;   // Upstream shock Mach number
    const double gamma = p.GAMMA;

    double rx = x - xv;
    double ry = y - yv;
    double r  = std::sqrt(rx * rx + ry * ry);

    // Compute tangential velocity and temperature profile
    double v_theta = 0.0;
    double T_r     = 1.0;

    const double C = Mv * ra / (ra * ra - rb * rb);

    if (r <= ra) {
        v_theta = Mv * (r / ra);
        
        // Temperature at r_a
        double T_ra = 1.0 - (gamma - 1.0) * C * C * 
                      (2.0 * rb * rb * std::log(ra / rb) - 0.5 * ra * ra + 0.5 * std::pow(rb, 4) / (ra * ra));
        
        // Inner core temperature profile
        T_r = T_ra - 0.5 * (gamma - 1.0) * Mv * Mv * (1.0 - (r * r) / (ra * ra));

    } else if (r < rb) {
        v_theta = C * (r - (rb * rb) / r);

        // Outer region temperature profile
        T_r = 1.0 - (gamma - 1.0) * C * C * 
              (2.0 * rb * rb * std::log(r / rb) - 0.5 * r * r + 0.5 * std::pow(rb, 4) / (r * r));
    }

    double u_vort = (r > 1e-12) ? -v_theta * (ry / r) : 0.0;
    double v_vort = (r > 1e-12) ?  v_theta * (rx / r) : 0.0;

    // Upstream vortex thermodynamic fields
    double rho_vort = std::pow(T_r, 1.0 / (gamma - 1.0));
    double p_vort   = (1.0 / gamma) * std::pow(T_r, gamma / (gamma - 1.0));

    // Rankine-Hugoniot standing shock jump across x = xs
    double p1   = 1.0 / gamma;
    double rho1 = 1.0;
    double u1   = Ms * std::sqrt(gamma * p1 / rho1); // u1 = 1.5

    double rho2 = rho1 * ((gamma + 1.0) * Ms * Ms) / ((gamma - 1.0) * Ms * Ms + 2.0);
    double p2   = p1 * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (Ms * Ms - 1.0));
    double u2   = u1 * (rho1 / rho2);

    // Exact Heaviside step jump across shock front
    if (x < xs) {
        rho   = rho1 * rho_vort;
        u     = u1 + u_vort;
        v     = v_vort;
        press = p_vort;
    } else {
        rho   = rho2;
        u     = u2;
        v     = 0.0;
        press = p2;
    }
}