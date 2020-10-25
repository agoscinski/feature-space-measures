out
#echo "\n##########################"
#echo "# GFRE/GFRD symmetrized 2-body atomic density function convergence experiment #"
#echo "##########################\n"
#taskset -c 4 python experiment_sym_2_body_adcf_convergence_on_U0_qm9.py 10000 --option
#
#echo "\n##########################"
#echo "# GFRE/GFRD schnet convergence experiment #"
#echo "##########################\n"
#taskset -c 4 python experiment_schnet_convergence_on_U0_qm9.py 10000 --option
#
# [(2, 2), (4, 3), (6, 4)]  (8,5) is missing
#echo "\n##########################"
#echo "# GFRE/GFRD SOAP convergence experiment #"
#echo "##########################\n"
#taskset -c 4 python experiment_soap_convergence_on_U0_qm9.py 10000 --option

##########################
out2

echo "\n##########################"
echo "# GFRE/GFRD soap schnet comparison all properties experiment #"
echo "##########################\n"
taskset --cpu-list 0,1,2,3,4,5,6,7,8 python experiment_soap_schnet_on_all_properties_qm9.py 10000 --option

##########################
out3
echo "\n##########################"
echo "# GFRE/GFRD symmetrized 2-body atomic density function poly kernel comparison with schnet experiment #"
echo "##########################\n"
taskset --cpu-list 33,34,35,36,37,38  python experiment_sym_2_body_adcf_poly_kernel_schnet_comparison_on_U0_qm9.py 10000 --option
##########################

out4 with updated totial variance of sparse kernel
echo "\n##########################"
echo "# GFRE/GFRD symmetrized 2-body atomic density function poly kernel comparison with schnet experiment #"
echo "##########################\n"
taskset --cpu-list 33,34,35,36,37,38  python experiment_sym_2_body_adcf_poly_kernel_schnet_comparison_on_U0_qm9.py 10000 --option
##########################

out5
echo "\n##########################"
echo "# GFRE/GFRD SOAP convergence experiment #"
echo "##########################\n"
taskset -c 4 python experiment_soap_pca_convergence_on_U0_qm9.py 10000 --option
##########################

echo "\n##########################"
echo "# experiment_dimenet_convergence_on_U0_qm9.py #"
echo "##########################\n"
taskset --cpu-list 0,1,2,3,4,5,6,7,8 python experiment_dimenet_convergence_on_U0_qm9.py 10000 --option

echo "\n##########################"
echo "# experiment_soap_poly_kernel_dimenet_comparison_on_U0_qm9.py  #"
echo "##########################\n"
taskset --cpu-list 0,1,2,3,4,5,6,7,8 python experiment_soap_poly_kernel_dimenet_comparison_on_U0_qm9.py 10000 --opti5on


#repeat
echo "\n##########################"
echo "# GFRE/GFRD SOAP convergence experiment #"
echo "##########################\n"
taskset --cpu-list 33,34,35,36,37,38 python experiment_soap_convergence_on_U0_qm9.py 10000 --option

#echo "\n##########################"
#echo "#  #"
#echo "##########################\n"
#taskset --cpu-list 33,34,35,36,37,38 python experiment_soap_dimenet_comparison_on_all_porperties_qm9.py 10000 --option



#taskset --cpu-list 29,30,31,32 python compute_dimenet_features.py 10000 --option &

########################### NICE

#taskset -c 0 python experiment_soap_nice_schnet_gschnet_comparison_on_qm9.py --option


########################### LFRE

#echo "\n##########################"
#echo "# LFRE soap schnet experiment #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_schnet_lfre.py --option
