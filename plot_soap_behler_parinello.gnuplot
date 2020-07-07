set terminal pdfcairo font ",20" siz 20cm,10cm

if (ARGC < 1) print "missing hash, specify it with gnuplot -c script.gnuplot 0xdeadbeef"; exit
hash=ARG1

set output 'plots/gfr-' . hash . '.pdf'

set xlabel "number of features"
set ylabel "gfr"
set yrange [0:]

dataset=system("cat results/metadata-" . hash . ".json | jq -r '.dataset + \" \" + (.features_hypers[0][0].feature_selection_parameters.type // \"\" | tostring) + \" \" + (.features_hypers[0][0].feature_selection_parameters.n_features // \"\" | tostring)'")

set multiplot layout 1,2
    set title "SOAP -> BP\n" . dataset
    plot 'results/gfre-' . hash . '.dat' u 3:1 w lp lw 3 t "gfre", \
         'results/gfrd-' . hash . '.dat' u 3:1 w lp lw 3 t "gfrd"

    set title "BP -> SOAP\n" . dataset
    plot 'results/gfre-' . hash . '.dat' u 3:2 w lp lw 3 t "gfre", \
         'results/gfrd-' . hash . '.dat' u 3:2 w lp lw 3 t "gfrd"
unset multiplot
