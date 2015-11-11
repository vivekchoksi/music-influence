#
# In-degree distribution. G(19741, 78405). 3990 (0.2021) nodes with in-deg > avg deg (7.9), 1367 (0.0692) with >2*avg.deg (Tue Nov 10 01:42:34 2015)
#

set title "In-degree distribution. G(19741, 78405). 3990 (0.2021) nodes with in-deg > avg deg (7.9), 1367 (0.0692) with >2*avg.deg"
set key bottom right
set logscale xy 10
set format x "10^{%L}"
set mxtics 10
set format y "10^{%L}"
set mytics 10
set grid
set xlabel "In-degree"
set ylabel "Count"
set tics scale 2
set terminal svg size 1000,800
set output 'inDeg.in_degree_distr.svg'
plot 	"inDeg.in_degree_distr.tab" using 1:2 title "" with linespoints pt 6
