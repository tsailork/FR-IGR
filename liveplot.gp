# 1. Basic configuration for CSV
set datafile separator ","
set autoscale y

# Adds 10% empty space above and below your data points
set offsets 0, 0, graph 0.1, graph 0.1

# 2. Plotting command
# Using 1:2 means Column 1 is X and Column 2 is Y
plot for [i=2:5] "< tail -n 500 csv_outputs/residuals.csv" using 1:i with lines title "Residual "
### plot for [i=2:4] "< tail -n 500 csv_outputs/probe.csv" using 1:i with lines title "Probe "

# 3. Real-time Loop (Every 1 second)
while (1) {
    pause 1
    replot
}
