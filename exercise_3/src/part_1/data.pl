#
#
#
#
#
#
set terminal png
set output 'image.png'
set xlabel "Memory size in Byte"
set ylabel "Bandwidth in GB/s"
set logscale x
set logscale y
set xrange [1e3:1e9]
f(x) = a + x*b
#fit f(x) 'data_fix.txt' via a,b
plot 'data.txt' title "Messwerte" #, f(x) title "linearer fit"