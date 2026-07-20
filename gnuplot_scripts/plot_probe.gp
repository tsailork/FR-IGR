# CSV setup
set datafile separator ","

# Styling
set title "Probe Pressure History" font "DejaVu Sans Bold,14" tc rgb "#2F3E46"
set xlabel "Simulation Time" font "DejaVu Sans,12" tc rgb "#2F3E46"
set ylabel "Pressure" font "DejaVu Sans,12" tc rgb "#2F3E46"

# Grid style
set grid lc rgb "#E5E5E5" lt 1 lw 1.5
set border 3 lc rgb "#4F5D65" lw 1.5
set tics nomirror tc rgb "#4F5D65"

# Legend (Key) styling
set key right top box lt 1 lc rgb "#E5E5E5" spacing 1.3 font "DejaVu Sans,10"

# Color palette (up to 8, loops around if more)
array colors[8]
colors[1] = "#0077B6" # Blue
colors[2] = "#E76F51" # Orange
colors[3] = "#2A9D8F" # Teal
colors[4] = "#8338EC" # Purple
colors[5] = "#F4A261" # Light Orange
colors[6] = "#E63946" # Red
colors[7] = "#457B9D" # Steel Blue
colors[8] = "#1D3557" # Dark Navy

get_color(i) = colors[((i-2) % 8) + 1]

while (1) {
    # Dynamic column detection
    num_cols = int(system("grep -v '^#' csv_outputs/probe.csv 2>/dev/null | head -n 1 | awk -F, '{print NF}'"))

    plot for [i=2:num_cols] "csv_outputs/probe.csv" using 1:i title columnhead(i) with lines lw 2.5 lc rgb get_color(i)
    
    pause 10
}
