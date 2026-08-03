# 1. Basic configuration for CSV
set datafile separator ","
set autoscale y

# Adds 10% empty space above and below your data points
set offsets 0, 0, graph 0.1, graph 0.1

 
# 2. Plotting command (Update 'data.csv' to your filename)
# Using 1:2 means Column 1 is X and Column 2 is Y
plot for [i=2:4] "< tail -n 999999 probe.csv" using 1:i with lines title "Probe "

while (1) {
  pause 1
}
