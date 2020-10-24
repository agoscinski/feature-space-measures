echo "\n##########################"
echo "# GFRE/GFRD symmetrized 2-body atomic density function convergence experiment #"
echo "##########################\n"
taskset -c 4 python experiment_sym_2_body_adcf_convergence_on_U0_qm9.py 10000 --option

echo "\n##########################"
echo "# GFRE/GFRD schnet convergence experiment #"
echo "##########################\n"
taskset -c 4 python experiment_schnet_convergence_on_U0_qm9.py 10000 --option

echo "\n##########################"
echo "# GFRE/GFRD SOAP convergence experiment #"
echo "##########################\n"
taskset -c 4 python experiment_soap_convergence_on_U0_qm9.py 10000 --option

echo "\n##########################"
echo "# GFRE/GFRD symmetrized 2-body atomic density function poly kernel comparison with schnet experiment #"
echo "##########################\n"
taskset -c 4 python experiment_sym_2_body_adcf_poly_kernel_schnet_comparison_on_U0_qm9.py 10000 --option

# TODO
#echo "\n##########################"
#echo "# GFRE/GFRD dimenet convergence experiment #"
#echo "##########################\n"
#taskset -c 0 python experiment_dimenet_convergence_on_U0_qm9.py --option


#echo "\n##########################"
#echo "# GFRE/GFRD soap 3-body convergence experiment #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_schnet_on_all_properties_qm9.py --option


# TODO
#echo "\n##########################"
#echo "#  #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_poly_kernel_dimenet_comparison_on_U0_qm9.py --option

# TODO
#echo "\n##########################"
#echo "#  #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_dimenet_comparison_on_all_porperties_qm9.py --option

#echo "\n##########################"
#echo "#  #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_dimenet_comparison_on_all_porperties_qm9.py --option

########################### NICE

#taskset -c 0 python experiment_soap_nice_schnet_gschnet_comparison_on_qm9.py --option


########################### LFRE

#echo "\n##########################"
#echo "# LFRE soap schnet experiment #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_schnet_lfre.py --option
