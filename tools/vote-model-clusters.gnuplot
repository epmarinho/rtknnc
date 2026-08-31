# Article-ready RTKNNC cluster panels rendered directly from existing layers.
# Required variables are supplied by render-vote-model-gnuplot.py.

if (!exists("dataset_title")) print "dataset_title is required"
if (!exists("is3d")) is3d = 0
if (!exists("point_size")) point_size = 0.46

set encoding utf8
set datafile separator whitespace
set key off
set tics out nomirror scale 0.55
set border linewidth 1.15 linecolor rgb "#4A4A4A"
set grid xtics ytics linecolor rgb "#E4E7EB" linewidth 0.65
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

# Layer cluster 0 is the historical unclustered sentinel.  The remaining
# identifiers are cycled through a colour-blind-friendly categorical palette.
cluster_colour(cluster_id) = cluster_id < 0.5 ? 0 : 1 + (int(cluster_id) - 1) % 12
truth_colour(label_id) = 1 + int(label_id) % 12

do for [render_index=1:2] {
    if (render_index == 1) {
        set terminal pngcairo enhanced truecolor size 2100,1350 \
            font "Liberation Sans,13" linewidth 1.2
        set output outbase . ".png"
    } else {
        set terminal pdfcairo enhanced color size 13in,8.35in \
            font "Liberation Sans,11" linewidth 1.0
        set output outbase . ".pdf"
    }

    # The sixth documentation panel fixes both ranges to [0:1].  GNUplot
    # retains that state between terminal passes, so restore data-driven
    # ranges before plotting the next PNG/PDF pass.
    set autoscale x
    set autoscale y
    set autoscale z

    set multiplot layout 2,3 rowsfirst title dataset_title font ",19" \
        margins 0.055,0.985,0.075,0.925 spacing 0.055,0.105

    set xlabel "x_{0}" font ",13"
    set ylabel "x_{1}" font ",13"
    if (is3d) {
        set zlabel "x_{2}" font ",13" offset 1.2,0
        set ztics
        set grid xtics ytics ztics linecolor rgb "#E4E7EB" linewidth 0.65
        # GNUplot's 3-D projection extends beyond the nominal plot box; the
        # reduced camera scale keeps all three labelled axes inside each cell.
        set view 62,32,0.50,0.92
        set xyplane relative 0.08
        set title reference_title font ",12"
        splot reference_file using 3:4:5:(truth_colour(column(6))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title uniform_title font ",12"
        splot uniform_file using 3:4:5:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title inv_r_title font ",12"
        splot inv_r_file using 3:4:5:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title inv_r2_title font ",12"
        splot inv_r2_file using 3:4:5:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title sigmoid_title font ",12"
        splot sigmoid_file using 3:4:5:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle
    } else {
        unset zlabel
        unset ztics
        set grid xtics ytics linecolor rgb "#E4E7EB" linewidth 0.65
        set size ratio -1
        set title reference_title font ",12"
        plot reference_file using 3:4:(truth_colour(column(5))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title uniform_title font ",12"
        plot uniform_file using 3:4:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title inv_r_title font ",12"
        plot inv_r_file using 3:4:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title inv_r2_title font ",12"
        plot inv_r2_file using 3:4:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle

        set title sigmoid_title font ",12"
        plot sigmoid_file using 3:4:(cluster_colour(real(stringcolumn(1)))) \
            with points pointtype 7 pointsize point_size linecolor palette notitle
    }

    # Sixth panel documents the controlled comparison without adding a legend
    # whose colour-to-cluster mapping would be misleading across panels.
    unset title
    unset xlabel
    unset ylabel
    unset zlabel
    unset xtics
    unset ytics
    unset ztics
    unset border
    unset grid
    set size noratio
    set xrange [0:1]
    set yrange [0:1]
    set label 1 "Four budget-matched shapes; inverse-square remains historical." \
        at graph 0.04,0.78 left font ",13" textcolor rgb "#30343B"
    set label 2 "Colours identify clusters independently in each panel." \
        at graph 0.04,0.61 left font ",13" textcolor rgb "#30343B"
    set label 3 "Grey denotes an unclustered point." \
        at graph 0.04,0.44 left font ",13" textcolor rgb "#30343B"
    set label 4 "Every model uses the same KNN graph and source vote budget." \
        at graph 0.04,0.27 left font ",13" textcolor rgb "#30343B"
    plot NaN notitle
    unset label 1
    unset label 2
    unset label 3
    unset label 4

    unset multiplot
    unset output

    # Restore panel settings for the second terminal pass.
    set tics out nomirror scale 0.55
    set border linewidth 1.15 linecolor rgb "#4A4A4A"
    set grid xtics ytics linecolor rgb "#E4E7EB" linewidth 0.65
}
