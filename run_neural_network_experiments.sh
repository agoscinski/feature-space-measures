echo "\n##########################"
echo "# GFRE soap 2-body convergence experiment #"
echo "##########################\n"
taskset -c 0 python experiment_soap_2_body_convergence_on_qm9.py --option

echo "\n##########################"
echo "# GFRE soap 2-body schnet comparisonexperiment #"
echo "##########################\n"
taskset -c 0 python experiment_soap_2_body_schnet_comparison_on_qm9.py --option

echo "\n##########################"
echo "# GFRE soap 3-body convergence experiment #"
echo "##########################\n"
taskset -c 0 python experiment_soap_3_body_convergence_on_qm9.py --option

#echo "\n##########################"
#echo "# GFRE soap schnet experiment #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_schnet.py --option
#
#echo "\n##########################"
#echo "# LFRE soap schnet experiment #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_schnet_lfre.py --option
