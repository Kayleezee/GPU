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
set key left
set logscale y
#fit f(x) 'data_fix.txt' via a,b
plot '1Thread.dat' with linespoints ls 1 title "1 Threads" , \
     '2Thread.dat' with linespoints title "2 Threads" , \
     '4Thread.dat' with linespoints title "4 Threads", \
     '8Thread.dat' with linespoints title "8 Threads", \
     '16Thread.dat' with linespoints title "16 Threads", \
     '32Thread.dat' with linespoints title "32 Threads", \
     '64Thread.dat' with linespoints title "64 Threads", \
     '128Thread.dat' with linespoints title "128 Threads", \
     '256Thread.dat' with linespoints title "256 Threads", \
     '512Thread.dat' with linespoints title "512 Threads", \
     '1024Thread.dat' with linespoints title "1024 Threads"