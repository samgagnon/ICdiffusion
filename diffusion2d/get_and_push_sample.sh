#!/bin/bash

eval $(ssh-agent)

step ssh login samuel.gagnonhartman@sns.it --provisioner cineca-hpc

scp sgagnonh@login.leonardo.cineca.it:~/ICdiffusion/diffusion2d/run/fiducial/sample_1.npy ~/ICdiffusion/diffusion2d/run/fiducial/sample.npy
scp sgagnonh@login.leonardo.cineca.it:~/diff_data/galaxies/xHI_maps/xH_nohalos_z008.00_nf0.950820_eff232.7_effPLindex0_HIIfilter1_Mmin1.0e+11_RHIImax50_200_300Mpc.npy ~/ICdiffusion/diffusion2d/run/fiducial/target.npy
scp sgagnonh@login.leonardo.cineca.it:~/diff_data/galaxies/galsmear/muv_cubes/xH_nohalos_z008.00_nf0.950820_eff232.7_effPLindex0_HIIfilter1_Mmin1.0e+11_RHIImax50_200_300Mpc.npy ~/ICdiffusion/diffusion2d/run/fiducial/source.npy

sshpass -p "tehl33TH4x0r!"  scp -r ~/ICdiffusion/diffusion2d/run/fiducial/sample.npy sgagnonhartman@trantor01.sns.it:~/
sshpass -p "tehl33TH4x0r!"  scp -r ~/ICdiffusion/diffusion2d/run/fiducial/source.npy sgagnonhartman@trantor01.sns.it:~/
sshpass -p "tehl33TH4x0r!"  scp -r ~/ICdiffusion/diffusion2d/run/fiducial/target.npy sgagnonhartman@trantor01.sns.it:~/
