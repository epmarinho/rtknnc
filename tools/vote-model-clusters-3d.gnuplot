# Standalone 3-D RTKNNC layer figure.  Variables are provided by the Python
# runner so that GNUplot remains the renderer and the existing layer is reused.

if (!exists("truth_mode")) truth_mode = 0
if (!exists("point_size")) point_size = 0.52

set encoding utf8
set datafile separator whitespace
set key off
set tics out nomirror scale 0.55
set border 4095 linewidth 1.15 linecolor rgb "#4A4A4A"
set grid xtics ytics ztics linecolor rgb "#E4E7EB" linewidth 0.65
set xlabel "x_{0}" font ",15" offset 0,0.6
set ylabel "x_{1}" font ",15" offset 1.0,0
set zlabel "x_{2}" font ",15" offset 1.2,0
set title plot_title font ",18" offset 0,-0.4
set view 62,32,0.92,1.0
set xyplane relative 0.06
set palette defined ( \
    0 "#9A9A9A", \
    1 "#0072B2", \
    2 "#D55E00", \
    3 "#009E73", \
    4 "#CC79A7", \
    5 "#E69F00", \
    6 "#56B4E9", \
    7 "#F0E442", \
    8 "#332288", \
    9 "#88CCEE", \
   10 "#44AA99", \
   11 "#117733", \
   12 "#AA4499")
set cbrange [0:12]
unset colorbox

cluster_colour(cluster_id) = cluster_id < 0.5 ? 0 : 1 + (int(cluster_id) - 1) % 12
truth_colour(label_id) = 1 + int(label_id) % 12

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pngcairo enhanced truecolor size 1800,1350 \
            font "Liberation Sans,14" linewidth 1.2
        set output outbase . ".png"
    } else {
        set terminal pdfcairo enhanced color size 10.5in,7.9in \
            font "Liberation Sans,12" linewidth 1.0
        set output outbase . ".pdf"
    }

    if (truth_mode) {
        splot input_file using 3:4:5:(truth_colour(column(6))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle
    } else {
        splot input_file using 3:4:5:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle
    }
    unset output
}
