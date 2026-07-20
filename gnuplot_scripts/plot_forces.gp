# CSV setup
set datafile separator ","

# General styling options
set grid lc rgb "#E5E5E5" lt 1 lw 1.5
set border 3 lc rgb "#4F5D65" lw 1.5
set tics nomirror tc rgb "#4F5D65"
set key right top box lt 1 lc rgb "#E5E5E5" spacing 1.3

while (1) {
    # Enable multiplot
    set multiplot layout 2,1 title "Aerodynamic Forces & Coefficients" font "DejaVu Sans Bold,14" tc rgb "#2F3E46"

    # --- Subplot 1: Aerodynamic Coefficients ---
    set title "Force Coefficients (C_d, C_l)" font "DejaVu Sans Bold,12" tc rgb "#2F3E46"
    set ylabel "Coefficient Value" font "DejaVu Sans,11" tc rgb "#2F3E46"
    unset xlabel
    set format x ""

    plot "csv_outputs/forces.csv" using 1:4 title "C_d (Drag Coefficient)" with lines lw 2.5 lc rgb "#E76F51", \
         "csv_outputs/forces.csv" using 1:5 title "C_l (Lift Coefficient)" with lines lw 2.5 lc rgb "#0077B6"

    # --- Subplot 2: Physical Forces ---
    set title "Integrated Physical Forces" font "DejaVu Sans Bold,12" tc rgb "#2F3E46"
    set xlabel "Simulation Time" font "DejaVu Sans,11" tc rgb "#2F3E46"
    set ylabel "Force Value" font "DejaVu Sans,11" tc rgb "#2F3E46"
    set format x "%g" # Restore x-axis labels

    plot "csv_outputs/forces.csv" using 1:2 title "Drag Force" with lines lw 2.5 lc rgb "#D81159", \
         "csv_outputs/forces.csv" using 1:3 title "Lift Force" with lines lw 2.5 lc rgb "#4F5D65"

    unset multiplot
    
    pause 10
}
