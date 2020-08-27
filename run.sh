### SOAP CONVERGENCE

# data for Fig. 1
echo "\n##########################"
echo "# max radial experiments #"
echo "##########################\n"
taskset -c 0 python soap_ps_max_radial_experiments.py --option

# data for Fig. 1
echo "\n###########################"
echo "# max angular experiments #"
echo "###########################\n"
taskset -c 0 python soap_ps_max_angular_experiments.py --option

# data for Fig. 1
echo "\n###########################"
echo "# radial bases experiments #"
echo "###########################\n"
taskset -c 0 python soap_ps_radial_basis_experiments.py --option

### SOAP DENSITY PLOTS

# data for Fig. 2
echo "\n###########################"
echo "# ps sigma experiments #"
echo "###########################\n"
taskset -c 0 python soap_ps_sigma_experiments.py --option

# data for Fig. 2
echo "\n###########################"
echo "# radial scaling experiments #"
echo "###########################\n"
taskset -c 0 python soap_ps_radial_scaling_experiments.py --option

# data for Fig. 2
echo "\n###########################"
echo "# radial scaling experiments #"
echo "###########################\n"
taskset -c 0 python soap_ps_cutoff_experiments.py --option

### SOAP BPSF

# data for Fig. 3
echo "\n##########################"
echo "# behler parinello #"
echo "##########################\n"
taskset -c 0 python soap_ps_behler_parinello_experiments.py --option

# data for Fig. 4
echo "\n##########################"
echo "# behler parinello fs #"
echo "##########################\n"
taskset -c 0 python soap_ps_behler_parinello_self_feature_selection_experiments.py --option

### SOAP BODY ORDER

# data for Fig. 5
echo "\n###########################"
echo "# body order gfrm experiments #"
echo "###########################\n"
taskset -c 0 python soap_body_order_experiments.py --option

# data for Fig. 6
echo "\n#################"
echo "# body order lfre #"
echo "###################\n"
taskset -c 0 python soap_body_order_lfre_experiments.py --option


### DEGENERATED MANIFOLD 

# data for Fig. 7
echo "\n###########################"
echo "# lfre #"
echo "###########################\n"
taskset -c 0 python soap_degenerated_manifold_lfre_experiments.py --option


### RBF KERNEL

# data for Fig. 8
echo "\n#################"
echo "# body order rbf pair plot #"
echo "###################\n"
taskset -c 0 python soap_ps_kernel_gammas_experiments.py --option

# data for Fig. 9
echo "\n#################"
echo "# body order rbf gamma plot #"
echo "###################\n"
taskset -c 0 python soap_ps_kernel_gammas_body_order_experiments.py --option


### WASSERSTEIN DISTANCE

# data for Fig. 10
echo "\n###########################"
echo "# samplewise error plot#"
echo "###########################\n"
taskset -c 0 python soap_rs_distance_gfre_train_test_experiments.py --option

# data for Fig. 11
echo "\n###########################"
echo "# distance density plots #"
echo "###########################\n"
taskset -c 0 python compute_density_mat.py --option

# data for Fig. 12
echo "\n###########################"
echo "# gfrm distance comparison #"
echo "###########################\n"
taskset -c 0 python soap_rs_distance_gfrm_pair_experiments.py --option
