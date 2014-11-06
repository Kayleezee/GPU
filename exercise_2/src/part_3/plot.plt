reset;
set title "PCIe Data Movements"
set key left
set xlabel "datasize in KB"
set ylabel "everage time of 10000 in microseconds"
plot 'data.dat' u 1:2 w lp title "H2D pageable", 'data.dat' u 1:3 w lp title "H2D pinnend", 'data.dat' u 1:4 w lp title "D2H pageable", 'data.dat' u 1:5 w lp title "D2H pinnend"