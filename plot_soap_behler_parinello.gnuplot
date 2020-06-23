set terminal pdfcairo font ",20" siz 20cm,10cm

if (ARGC < 1) print "missing hash, specify it with gnuplot -c script.gnuplot 0xdeadbeef"; exit
hash=ARG1

set output 'plots/gfre-' . hash . '.pdf'

set xlabel "number of features"
set ylabel "gfr"
set yrange [0:]

set multiplot layout 1,2
    set title "SOAP -> BP"
    plot 'results/gfre-' . hash . '.dat' u 3:1 w lp lw 3 t "gfre", \
         'results/gfrd-' . hash . '.dat' u 3:1 w lp lw 3 t "gfrd"

    set title "BP -> SOAP"
    plot 'results/gfre-' . hash . '.dat' u 3:2 w lp lw 3 t "gfre", \
         'results/gfrd-' . hash . '.dat' u 3:2 w lp lw 3 t "gfrd"
unset multiplot
