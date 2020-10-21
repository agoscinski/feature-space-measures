echo "\n##########################"
echo "# GFRE soap schnet experiment #"
echo "##########################\n"
taskset -c 0 python experiment_soap_schnet.py --option

echo "\n##########################"
echo "# LFRE soap schnet experiment #"
echo "##########################\n"
taskset -c 0 python experiment_soap_schnet_lfre.py --option
