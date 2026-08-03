# CSV setup
set datafile separator ","

# Styling
set title "Normalized Residuals History" font "DejaVu Sans Bold,14" tc rgb "#2F3E46"
set xlabel "Simulation Time" font "DejaVu Sans,12" tc rgb "#2F3E46"
set ylabel "L2 Residual (Normalized)" font "DejaVu Sans,12" tc rgb "#2F3E46"

# Grid style
set grid lc rgb "#E5E5E5" lt 1 lw 1.5
set border 3 lc rgb "#4F5D65" lw 1.5
set tics nomirror tc rgb "#4F5D65"

# Y-axis logscale for residuals
set logscale y
set format y "10^{%L}"

# Legend (Key) styling
set key right top box lt 1 lc rgb "#E5E5E5" spacing 1.3 font "DejaVu Sans,10"

# Colors
array colors[4]
colors[1] = "#0077B6" # Blue
colors[2] = "#2A9D8F" # Teal/Green
colors[3] = "#E76F51" # Orange/Red
colors[4] = "#8338EC" # Purple

# Normalization helper function
isnan(x) = (x != x)

while (1) {
    # Dynamic column detection (skipping first column 'Time')
    num_cols = int(system("grep -v '^#' csv_outputs/residuals.csv 2>/dev/null | head -n 1 | awk -F, '{print NF}'"))

    array init_vals[10]
    do for [i=1:10] { init_vals[i] = NaN }
    
    norm_val(col, val) = (isnan(init_vals[col]) ? (init_vals[col] = (val == 0.0 ? 1.0 : val)) : 0, val / init_vals[col])

    plot for [i=2:num_cols] "csv_outputs/residuals.csv" using 1:(norm_val(i, column(i))) title columnhead(i) with lines lw 2.5 lc rgb colors[i-1]
    
    pause 10
}
