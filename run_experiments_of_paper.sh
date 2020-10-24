#### SOAP CONVERGENCE EXPERIMENTS
#
## data for Fig. 1
#echo "\n##########################"
#echo "# radial convergence experiment #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_max_radial.py --option
#
## data for Fig. 1
#echo "\n###########################"
#echo "# angular convergence experiment #"
#echo "###########################\n"
#taskset -c 0 python experiment_soap_max_angular.py --option
#
## data for Fig. 1
#echo "\n###########################"
#echo "# radial basis convergence experiment #"
#echo "###########################\n"
#taskset -c 0 python experiment_soap_radial_basis.py --option
#
#### SOAP PARAMETERS COMPARISON EXPERIMENTS
#
## data for Fig. 2
#echo "\n###########################"
#echo "# smearing sigma experiment #"
#echo "###########################\n"
#taskset -c 0 python experiment_soap_sigma.py --option
#
## data for Fig. 2
#echo "\n###########################"
#echo "# radial scaling experiment #"
#echo "###########################\n"
#taskset -c 0 python experiment_soap_radial_scaling.py --option
#
## data for Fig. 2
#echo "\n###########################"
#echo "# cutoff experiment #"
#echo "###########################\n"
#taskset -c 0 python experiment_soap_cutoff.py --option
#
#### SOAP BPSF COMPARISON EXPERIMENTS
#
## data for Fig. 3
#echo "\n##########################"
#echo "# experiment_soap vs behler parinello comparison #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_behler_parinello.py --option
#
## data for Fig. 4
#echo "\n##########################"
#echo "# experiment_soap vs behler parinello comparison with feature selection #"
#echo "##########################\n"
#taskset -c 0 python experiment_soap_behler_parinello_self_feature_selection.py --option
#
#### SYMMETRIZED ATOM DENSITY BODY ORDER EXPERIMENTS
#
## data for Fig. 5
#echo "\n###########################"
#echo "# body order gfre experiment on methane dataset #"
#echo "###########################\n"
#taskset -c 0 python experiment_body_order_gfre_methane.py --option
#
## data for Fig. 6
#echo "###########################\n"
#echo "# body order lfre experiment on methane dataset #"
#echo "###########################\n"
#taskset -c 0 python experiment_body_order_lfre_methane.py --option
#
## data for Fig. 7
#echo "\n###########################"
#echo "# body order lfre experiment on degenerate manifold dataset #"
#echo "###########################\n"
#taskset -c 0 python experiment_body_order_lfre_degenerate_manifold.py --option


### RBF KERNEL EXPERIMENTS

# data for Fig. 8
echo "\n#################"
echo "# body order rbf gamma density #"
echo "###################\n"
taskset -c 0 python experiment_body_order_rbf_kernel_same_body_order.py --option

# data for Fig. 9
echo "\n#################"
echo "# body order rbf gamma linear #"
echo "###################\n"
taskset -c 0 python experiment_body_order_rbf_kernel_higher_body_order.py --option


#### DISTANCE EXPERIMENTS
#
## data for Fig. 10
#echo "\n###########################"
#echo "# pointwise gfre wasserstein vs euclidean comparison on displaced methane #"
#echo "###########################\n"
#taskset -c 0 python experiment_pair_correlation_distance_pointwise_gfre.py --option
#
## data for Fig. 11
#echo "\n###########################"
#echo "# distance density plots #"
#echo "###########################\n"
#taskset -c 0 python compute_density_mat.py --option
#
## data for Fig. 12
#echo "\n###########################"
#echo "# gfre wasserstein vs euclidean comparison on carbon dataset #"
#echo "###########################\n"
#taskset -c 0 python experiment_pair_correlation_distance_gfre_carbon.py --option
