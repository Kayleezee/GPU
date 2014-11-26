reset


# epslatex
set terminal png #size 10.4cm,6.35cm color colortext standalone 'phv,9' \
#header '\definecolor{t}{rgb}{0.5,0.5,0.5}'
set output 'matrix_multiply.png'

# define axis
# remove border on top and right and set color to gray
set datafile separator ";"
#set style line 11 lc rgb '#808080' lt 1
#set border 3 front ls 11
#set tics nomirror
# define grid
#set style line 12 lc rgb'#808080' lt 0 lw 1
#set grid back ls 12

# color definitions
#set style line 1 lc rgb '#8b1a0e' pt 1 ps 1.5 lt 1 lw 5 # --- red
#set style line 2 lc rgb '#5e9c36' pt 6 ps 1.5 lt 1 lw 5 # --- green
#set key bottom right

#set format '\color{t}$%g$'
#set title 'run time'
set xlabel 'time [s]'
set ylabel 'size'
#set xrange auto
set logscale x 
#set logscale y
set xtics (16,32,64,128,256,512,1024,2048,256,512,1024)
set yrange [0.:10.]

set ylabel 'time [s]'
set xlabel 'size'

plot 'matrix_multiply.dat' u 1:2 t 'run time' w lp ls 1


