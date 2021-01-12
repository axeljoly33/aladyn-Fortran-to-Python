#
# ------------------------------------------------------------------
# 12-10-2020
#
# Artificial Neural Network module for aladyn.f code.
# Open_MP version.
# Converted to Python.
#
# Yann Abou Jaoude - Axel Joly
# Ecole Supérieure d'Ingénieurs Léonard-de-Vinci
# 12 Avenue Léonard de Vinci,
# Courbevoie, 92400, FRANCE
# phone: (+33) 01 41 16 70 00
# fax:
# e-mail: yann.abou_jaoude@edu.devinci.fr - axel.joly@edu.devinci.fr
# ------------------------------------------------------------------
# 04-26-2019
# 
# Artificial Neural Network module for aladyn.f code
# Open_MP version
# 
# Vesselin Yamakov
# National Institute of Aerospace
# 100 Exploration Way,
# Hampton, VA 23666 
# phone: (757)-864-2850
# fax:   (757)-864-8911
# e-mail: yamakov@nianet.org
# ------------------------------------------------------------------
# Use a trained NN to get pot. param for a specific potential form
# ------------------------------------------------------------------
# 
# Notices:
# Copyright 2015, 2018, United States Government as represented by the 
# Administrator of the National Aeronautics and Space Administration. 
# All Rights Reserved.
# 
# Disclaimers:
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY
# WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, 
# INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE 
# WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM 
# INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
# FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM 
# TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT,IN ANY MANNER, 
# CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT 
# OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS 
# OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE. 
# FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES 
# REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE,
# AND DISTRIBUTES IT "AS IS." 
# 
# Waiver and Indemnity:
# RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES
# GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR 
# RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
# LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH
# USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, 
# RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND
# HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND 
# SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT
# PERMITTED BY LAW. RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL 
# BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 

import sys
import operator
import numpy as np
import random
import torch
import math
import time

#import aladyn_sys
import atoms
import sim_box
import constants
import pot_module
import node_conf
import group_conf
#import aladyn_MD
#import aladyn_IO
#import aladyn


#
# ------------------------------------------------------------------
#

# use sys_OMP #sys NO OMP / sys /sys ACC/sys omp
# use constants
# use sim_box
# use pot_module
# save
ireport=0

Max_net_layers = 8
n_set_ann = 0
net_atom_types = 0
iflag_ann = 0
net_layers = 0
net_in = 0
net_out = 0
mG_dim = 0
max_tri_index = 0

Rc_ann = 0.0
d_ann = 0.0
d4_ann = 0.0
Gauss_ann = 0.0
range_min_ann = 0.0
ActFunc_shift = 0.0

Nodes_of_layer = [0] * (Max_net_layers + 1)

r0_value = []           # 1 dim , double
r0G_value = []          # 1 dim , double

Gi_atom = [[[]]]        # 3 dim , double
dG_i = [[[]]]           # 3 dim , double

Gi_list = [[]]          # 2 dim , double
Gi_new = [[]]           # 2 dim , double

U0 = []                 # 1 dim , double
U1 = []                 # 1 dim , double
U2 = []                 # 1 dim , double

W1_ann = [[]]           # 2 dim , double
W3_ann = [[]]           # 2 dim , double
W2_ann = [[[]]]         # 3 dim , double

B1_ann = []             # 1 dim , double
B3_ann = []             # 1 dim , double
B2_ann = [[]]           # 2 dim , double

dBOP_param_dxij = [[[[]]]]    # 4 dim , double

buf_ann = []            # 1 dim , double

r0Rc = []               # 1 dim , double
r0pRc = []              # 1 dim , double
U1f1 = []               # 1 dim , double
U2f1 = []               # 1 dim , double
U1f2 = []               # 1 dim , double
U2f2 = []               # 1 dim , double
U1f3 = []               # 1 dim , double
U2f3 = []               # 1 dim , double
Gi_dev = [[]]           # 2 dim , double
xr_ij0 = [[]]           # 2 dim , double
xr_ij1 = [[]]           # 2 dim , double
xr_ij2 = [[]]           # 2 dim , double
xr_ij3 = [[]]           # 2 dim , double
xr_ij_dev = [[[]]]      # 3 dim , double
fsij_dev = [[[]]]       # 3 dim , double
dfuN_dev = [[[]]]       # 3 dim , double
Gi_3D_dev1 = [[[]]]     # 3 dim , double
Gi_3D_dev2 = [[[]]]     # 3 dim , double
Gi_3D_dev3 = [[[]]]     # 3 dim , double
dfs_rij_3D1 = [[[]]]    # 3 dim , double
dfs_rij_3D2 = [[[]]]    # 3 dim , double
dfs_rij_3D3 = [[[]]]    # 3 dim , double
dfs_rij_3D = [[[[]]]]   # 4 dim , double
Gi_3D_dev = [[[[]]]]    # 4 dim , double
dBOP_param_dxij_ = [[[[]]]]   # 4 dim , double


# ! --- TEST arrays ---
# !	 double precision, dimension(:,:), allocatable :: Gi_cp

# u	  CONTAINS

# !--------------------------------------------------------------------
# ! This subroutine reads in data from files set in pot.dat file
# ! describing a trained Neural Network for a specific potential format
# ! ipot_type = iPOT_func_type = 100:straight ANN; 106: BOP ANN;
# !
# ! FORMAT:
# ! iflag_ann,range_min_ann,Rc_ann,d_ann,Gauss_ann
# ! n_set_ann, (r0_value(i), i=1,n_set_ann)
# ! w1(1,1), w1(1,2),…w1(1,20), w1(2,1),w1(2,2),…w1(60,20)
# ! b1(1), b1(2),…b1(20)
# ! w2(1,1), w2(1,2),…w2(1,20), w2(2,1),w2(2,2),…w2(20,20)
# ! b2(1), b2(2),…b2(20)
# ! w3(1,1), w3(2,1), w3(3,1),…w3(20,1)
# ! b3(1)
# !--------------------------------------------------------------------

def input_pot_ANN(ipot_type):
    global Max_net_layers, n_set_ann, net_atom_types, iflag_ann, net_layers, net_in, net_out, mG_dim, max_tri_index
    global net_layers, net_in, net_out, mG_dim, max_tri_index, Rc_ann, d_ann, d4_ann, Gauss_ann, range_min_ann
    global ActFunc_shift, Nodes_of_layer, r0_value, r0G_value, Gi_atom, dG_i, Gi_list, Gi_new
    global U0, U1, U2, W1_ann, W3_ann, W2_ann, B1_ann, B3_ann, B2_ann, dBOP_param_dxij, buf_ann
    global r0Rc, r0pRc, U1f1, U2f1, U1f2, U2f2, U1f3, U2f3, Gi_dev, xr_ij0, xr_ij1, xr_ij2, xr_ij3, xr_ij_dev, fsij_dev, dfuN_dev
    global Gi_3D_dev1, Gi_3D_dev2, Gi_3D_dev3, dfs_rij_3D1, dfs_rij_3D2, dfs_rij_3D3, dfs_rij_3D, Gi_3D_dev, dBOP_param_dxij_
    global ireport

    ierror = 0
    ierr = 0
    WT_ann = [[]]  # double

    LINE = ""
    err_msg = ""
    elem_symb_pot = [""] * (112 + 1)  # ! maximum numb. elements possible !
    gram_mol_pot = [0.0] * (112 + 1)

    ierr = 0
    net_in = 0
    net_out = 0
    net_atom_types = 1

    Nodes_of_layer = [0] * (Max_net_layers + 1)
    nunit = 40  # ! ANN file !

    err_msg = ' '
    ierror = 0

    nunit = open(pot_module.filename, "r")

    print(' ')
    print('READING pot file: ' , pot_module.filename , '...')

    # ! *** Start reading the Artificial Neural Network file *.ann ***

    LINE = nunit.readline()  # ! 1st Line !
    pot_module.iPOT_file_ver = int(LINE[1])  # ! pot file version !
    # !	 write(6,*)' ANN version:',iPOT_file_ver

    LINE = nunit.readline()  # ! 2nd Line !
    net_atom_types = int(LINE[1])  # ! number of chemical elem. !

    for i in range(1, net_atom_types + 1):
        myLine = nunit.readline()
        myData = myLine.split()
        myData[0] = myData[0].split('\'')[1]

        elem_symb_pot[i] = myData[0]
        gram_mol_pot[i] = float(myData[1])

    print('')
    print('Elements in ANN potential file:')
    print(elem_symb_pot[1], ' ', net_atom_types)

    for i in range(1, atoms.iatom_types + 1):   # ! element from pot.dat file !
        ierror = 1                              # ! assume element not found  !
        for n in range(1, net_atom_types + 1):  # ! element from lammps pot file !
            nelem = pot_module.numb_elem_Z(elem_symb_pot[n])
            if nelem == pot_module.ielement[i]:
                pot_module.ipot[n] = i  # ! so that ielement(ipot(n)) = ielement(i) !
                ierror = 0
        if ierror != 0:
            print('ERROR: Element ', pot_module.elem_symb[i], ' from pot.dat not in ', pot_module.filename)
    print(' ')

    err_msg = 'ERROR elements in ANN pot file do not match those in the pot.dat file!'

    sim_box.error_check(ierror, err_msg)

    if sim_box.mynod == 0:
        LINE = nunit.readline()  # ! 1st NN-data Line !
        myData = LINE.split()
        iflag_ann = int(myData[0])
        range_min_ann = float(myData[1])
        Rc_ann = float(myData[2])
        d_ann = float(myData[3])
        Gauss_ann = float(myData[4])
        # 1 is missing?

        LINE = nunit.readline()  # ! 2nd NN-data Line !
        myData = LINE.split()
        if ierr == 0:
            n_set_ann = int(myData[0])
            if n_set_ann > 0:
                r0_value = [0.0] * (n_set_ann + 1)
                if ierr == 0:
                    for i in range(1, n_set_ann + 1):
                        r0_value[i] = float(myData[i])
            else:
                ierr = 1
                print('ERROR: No Gaussian positions in line 2 in ', pot_module.filename)
            if ierr == 0:
                print('Gaussian positions:', n_set_ann, ':', end='')
                for i in range(1, n_set_ann + 1):
                    print(' ', r0_value[i], end='')
                print('')

        LINE = nunit.readline()  # ! 3rd NN-data Line !
        if ierr == 0:
            myData = LINE.split()
            net_layers = int(myData[0])
            if (0 < net_layers) and (net_layers <= Max_net_layers):
                for i in range(1, net_layers + 1):
                    Nodes_of_layer[i] = int(myData[i])
                net_in = Nodes_of_layer[1]
                net_out = Nodes_of_layer[net_layers]
                net_in_check = 5 * n_set_ann * atoms.iatom_types
                if net_in != net_in_check:
                    ierr = 1
                    print('ERROR: Inconsistency b/n the number of input net nodes:', net_in,
                           ' and Structure Parameters:', net_in_check, ' = 5 *', n_set_ann, ' * ', atoms.iatom_types)
            else:
                ierr = 1
                print('ERROR: Incorrect Net layers in line 3 in ', pot_module.filename)
                print('Number of Net layers = ', net_layers, ' must be between 1 and ', Max_net_layers)

            if ierr == 0:
                print('NN layers=:', net_layers, 'of nodes:', end='')
                for i in range(1, net_layers + 1):
                    print(' ', Nodes_of_layer[i], end='')
                print('')

                # !	write(6,20) net_atom_types, net_in, net_out, Rc_ann, d_ann !

    sim_box.error_check(ierr, 'ERROR reading ANN file in input_pot_ANN')
    sim_box.error_check((net_atom_types - atoms.iatom_types), 'ERROR: Elements in ANN file and pot.dat do not match!')

    if pot_module.iPOT_file_ver == 0 or pot_module.iPOT_file_ver == 1:
        ActFunc_shift = 0.0  # ! f(x) = 1/(1+exp(-x)) !
    elif pot_module.iPOT_file_ver == 2:
        ActFunc_shift = -0.5  # ! f(x) = 1/(1+exp(-x)) - 0.5 !
    else:
        ActFunc_shift = 0.0  # ! f(x) = 1/(1+exp(-x)) !

    r0G_value = [0.0] * (n_set_ann + 1)
    sim_box.error_check(ierr, 'ERROR allocate r0G_value in input_pot_ANN')
    for i in range(1, n_set_ann + 1):
        r0G_value[i] = r0_value[i] / Gauss_ann

    alloc_types_ANN()
    sim_box.error_check(ierr, 'ERROR alloc_types_ANN in input_pot_ANN')

    ierr = 0
    ww, bb = 0.0, 0.0

    # ! --- Read Input Layer Parameters for atom of type itype ---

    Ncolumns = Nodes_of_layer[1]  # ! 60 !
    Nraws = Nodes_of_layer[2]  # ! 20 !
    for icol in range(1, Ncolumns + 1):  # ! 1.. 60 !
        for iraw in range(1, Nraws + 1):  # ! 1.. 20: w(1,1), w(1,2), w(1,3)... !
            LINE = nunit.readline()  # ! ANN Line ! 7-1206
            myData = LINE.split()
            if ierr == 0:
                W1_ann[icol][iraw] = float(myData[0])
                dumb = float(myData[1])
    for iraw in range(1, Nraws + 1): # ! 1.. 20 !
        LINE = nunit.readline()  # ! ANN Line ! 1207-1226
        myData = LINE.split()
        if ierr == 0:
            B1_ann[iraw] = float(myData[0])
            dumb = float(myData[1])

    # --- Read Hidden Layers Parameters for atom of type itype ---

    for layer in range(2, net_layers - 2 + 1): # 4 - 2 + 1 = 3
        Ncolumns = Nodes_of_layer[layer]  # ! 20 !
        Nraws = Nodes_of_layer[layer + 1]  # ! 20 !
        for icol in range(1, Ncolumns + 1):  # ! 1.. 20 !
            for iraw in range(1, Nraws + 1):  # ! 1.. 20: w(1,1), w(1,2), w(1,3)... !
                LINE = nunit.readline()  # ! ANN Line ! 1227-1626
                myData = LINE.split()
                if ierr == 0:
                    ww = float(myData[0])
                    dumb = float(myData[1])
                W2_ann[icol][iraw][layer] = ww
        for iraw in range(1, Nraws + 1):
            LINE = nunit.readline()  # ! ANN Line ! 1627-1646
            myData = LINE.split()
            if ierr == 0:
                bb = float(myData[0])
                dumb = float(myData[1])
            B2_ann[iraw][layer] = bb

    # ! --- Read Output Layer Parameters for atom of type itype ---

    Ncolumns = Nodes_of_layer[net_layers - 1]  # ! 20 !
    Nraws = Nodes_of_layer[net_layers]  # ! 1 !
    for icol in range(1, Ncolumns + 1):  # ! 1.. 1 !
        for iraw in range(1, Nraws + 1):  # ! 1.. 20: w(1,1), w(1,2), w(1,3)... !
            LINE = nunit.readline()  # ! ANN Line ! 1647-1666
            myData = LINE.split()
            if ierr == 0:
                W3_ann[icol][iraw] = float(myData[0])
                dumb = float(myData[1])
    for iraw in range(1, Nraws + 1):
        LINE = nunit.readline()  # ! ANN Line ! 1667-1667
        myData = LINE.split()
        if ierr == 0:
            B3_ann[iraw] = float(myData[0])
            dumb = float(myData[1])

    nunit.close()

    sim_box.error_check(ierr, 'ERROR reading W_ann and B_ann arrays')

    # ! Swap index order from (n_set,l,...) to (l,n_set,...) of
    # ! the FIRST layer weights - FORTRAN order.

    Ncolumns = Nodes_of_layer[1]  # ! 60 !
    Nraws = Nodes_of_layer[2]  # ! 20 !

    WT_ann = [[0.0] * (Nraws + 1) for _ in range(Ncolumns + 1)]

    for i in range(1, len(W1_ann)):
        for j in range(1, len(W1_ann[0])):
            WT_ann[i][j] = W1_ann[i][j]

    for l in range(0, 4 + 1):
        for n_set in range(1, n_set_ann + 1):
            ind = l * n_set_ann + n_set
            icol = (n_set - 1) * 5 + l + 1
            for i in range(1, len(W1_ann[0])):
                W1_ann[icol][i] = WT_ann[ind][i]

    WT_ann.clear()

    ierror = ierr

    return ierror
    # ! End of input_pot_ANN !

# !--------------------------------------------------------------------
# ! Called from calc_pot_param() in aladyn.f
# !--------------------------------------------------------------------

def init_param_ANN():
    global Max_net_layers, n_set_ann, net_atom_types, iflag_ann, net_layers, net_in, net_out, mG_dim, max_tri_index
    global net_layers, net_in, net_out, mG_dim, max_tri_index, Rc_ann, d_ann, d4_ann, Gauss_ann, range_min_ann
    global ActFunc_shift, Nodes_of_layer, r0_value, r0G_value, Gi_atom, dG_i, Gi_list, Gi_new
    global U0, U1, U2, W1_ann, W3_ann, W2_ann, B1_ann, B3_ann, B2_ann, dBOP_param_dxij, buf_ann
    global r0Rc, r0pRc, U1f1, U2f1, U1f2, U2f2, U1f3, U2f3, Gi_dev, xr_ij0, xr_ij1, xr_ij2, xr_ij3, xr_ij_dev, fsij_dev, dfuN_dev
    global Gi_3D_dev1, Gi_3D_dev2, Gi_3D_dev3, dfs_rij_3D1, dfs_rij_3D2, dfs_rij_3D3, dfs_rij_3D, Gi_3D_dev, dBOP_param_dxij_
    global ireport

    # ! Get the maximum cut-off distance, r_cut_off and r2_cut_off !
    # ! and the maximum overlapping distance, rin

    pot_module.r_cut_off = 0.0

    mG_dim = int((atoms.iatom_types * (atoms.iatom_types + 1) * n_set_ann * 5) / 2)
    max_tri_index = int((atoms.iatom_types * (atoms.iatom_types + 1)) / 2)  # ! Upper triang. !

    if int(mG_dim) != Nodes_of_layer[1]:
        sim_box.error_check(1, 'ERROR dim. of Gis not equal to Nodes_of_layer(1)...')

    d4_ann = pow(d_ann, 4)
    pot_module.r_cut_off = Rc_ann
    pot_module.r_max_pot = pot_module.r_cut_off + 1.0  # ! Add additional 1 Ang to pot array for consistency with tabulated potentials!
    pot_module.rindr = 0.5  # ! [Ang] !

    return
    # ! End of init_param_ANN !

# ! ---------------------------------------------------------------------
# ! Calculates Analytical derivatives and force calculation.
# ! ---------------------------------------------------------------------

def Frc_ANN_OMP():
    global Max_net_layers, n_set_ann, net_atom_types, iflag_ann, net_layers, net_in, net_out, mG_dim, max_tri_index
    global net_layers, net_in, net_out, mG_dim, max_tri_index, Rc_ann, d_ann, d4_ann, Gauss_ann, range_min_ann
    global ActFunc_shift, Nodes_of_layer, r0_value, r0G_value, Gi_atom, dG_i, Gi_list, Gi_new
    global U0, U1, U2, W1_ann, W3_ann, W2_ann, B1_ann, B3_ann, B2_ann, dBOP_param_dxij, buf_ann
    global r0Rc, r0pRc, U1f1, U2f1, U1f2, U2f2, U1f3, U2f3, Gi_dev, xr_ij0, xr_ij1, xr_ij2, xr_ij3, xr_ij_dev, fsij_dev, dfuN_dev
    global Gi_3D_dev1, Gi_3D_dev2, Gi_3D_dev3, dfs_rij_3D1, dfs_rij_3D2, dfs_rij_3D3, dfs_rij_3D, Gi_3D_dev, dBOP_param_dxij_
    global ireport

    #use atoms

    time_initial = time.time()
    cpu_start = 0.0
    time_Gi = 0.0
    time_Gi_total = 0.0

    dgij1, dgij2, dgij3, U_x1, U_x2, U_x3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    Gi_sum_01, Gi_sum_11, Gi_sum_21, Gi_sum_31, Gi_sum_41 = 0.0, 0.0, 0.0, 0.0, 0.0
    Gi_sum_02, Gi_sum_12, Gi_sum_22, Gi_sum_32, Gi_sum_42 = 0.0, 0.0, 0.0, 0.0, 0.0
    Gi_sum_03, Gi_sum_13, Gi_sum_23, Gi_sum_33, Gi_sum_43 = 0.0, 0.0, 0.0, 0.0, 0.0
    fij = [0.0] * (3 + 1)
    fr = [0.0] * (9 + 1)
    dcos_ijk = [0.0] * (3 + 1)

    if sim_box.ihalt != 0:
        return

    if ireport > 0:
        print('Frc_ANN_OMP(mG_dim=', mG_dim, ' n_set_ann=', n_set_ann, ' max_nbrs=', atoms.max_nbrs, ')')

    ecohe = 0.0

    Rc = Rc_ann
    Rc2 = pow(Rc_ann, 2)
    d4 = d4_ann
    Sigma2 = 2.0 * pow(Gauss_ann, 2)
    for i in range(len(r0Rc)):
        r0Rc[i] = Rc_ann * r0_value[i]
    for i in range(len(r0pRc)):
        r0pRc[i] = Rc_ann + r0_value[i]

    sim_box.h11 = sim_box.h[1][1]
    sim_box.h12 = sim_box.h[1][2]
    sim_box.h13 = sim_box.h[1][3]
    sim_box.h22 = sim_box.h[2][2]
    sim_box.h23 = sim_box.h[2][3]
    sim_box.h33 = sim_box.h[3][3]

    # ! --- Start I loop over ni atoms --- !

    if ireport > 0:
        cpu_start = time.time()

    for ni in range(1, sim_box.natoms + 1):  # ! loop over atoms !
        sxn = atoms.sx[ni]
        syn = atoms.sy[ni]
        szn = atoms.sz[ni]

        for nb1 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !

            nj = atoms.nbr_list[nb1][ni]
            ij_delta = 1 - min(abs(nj - ni), 1)  # ! 1 if ni=nj; 0 if ni=/=nj !

            sx0 = atoms.sx[nj] - sxn
            sx0 = sx0 - round(sx0)  # ! make periodic along X !
            sy0 = atoms.sy[nj] - syn
            sy0 = sy0 - round(sy0)  # ! make periodic along Y !
            sz0 = atoms.sz[nj] - szn
            sz0 = sz0 - round(sz0)  # ! make periodic along Z !

            xij = sim_box.h11 * sx0 + sim_box.h12 * sy0 + sim_box.h13 * sz0
            yij = sim_box.h22 * sy0 + sim_box.h23 * sz0
            zij = sim_box.h33 * sz0
            r2ij = pow(xij, 2) + pow(yij, 2) + pow(zij, 2) + ij_delta * Rc2  # ! rij+Rc when i=j
            rij = math.sqrt(r2ij)
            r1ij = 1.0 / rij

            xr_ij0[nb1][ni] = rij
            xr_ij1[nb1][ni] = xij * r1ij
            xr_ij2[nb1][ni] = yij * r1ij
            xr_ij3[nb1][ni] = zij * r1ij

            RcRij = max(Rc - rij, 0)
            RcRij3 = pow(RcRij, 3)
            RcRij4 = RcRij3 * RcRij
            RcRij5 = RcRij4 * RcRij
            fc_rij = 0.25 * RcRij4 / (d4 + RcRij4)
            denominator = pow(Gauss_ann * (d4 + RcRij4), 2)

            for n_set in range(1, n_set_ann + 1):
                r0 = r0_value[n_set]  # ! do not use r0G_value() here !
                RijRo = rij - r0
                expRijRo = math.exp(-pow(RijRo / Gauss_ann, 2))
                fsij_dev[nb1][n_set][ni] = fc_rij * expRijRo
                dfsij = (r0Rc[n_set] + rij * (rij - r0pRc[n_set]) - Sigma2) * d4 - RijRo * RcRij5
                dfsij = 0.5 * dfsij * RcRij3 * expRijRo / denominator  # ! *2/4 !
                dfs_rij_3D1[n_set][nb1][ni] = dfsij * xr_ij1[nb1][ni]  # ! dfs_ij*xij/rij !
                dfs_rij_3D2[n_set][nb1][ni] = dfsij * xr_ij2[nb1][ni]  # ! dfs_ij*xij/rij !
                dfs_rij_3D3[n_set][nb1][ni] = dfsij * xr_ij3[nb1][ni]  # ! dfs_ij*xij/rij !

    # ! --- Start II loop over ni atoms --- !

    for ni in range(1, sim_box.natoms + 1):  # ! loop over atoms !
        for n_set in range(1, n_set_ann + 1):

            Gi_sum0, Gi_sum1, Gi_sum2, Gi_sum3, Gi_sum4 = 0.0, 0.0, 0.0, 0.0, 0.0

            for nb1 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !
                for nb2 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !

                    fsij_fsik = fsij_dev[nb1][n_set][ni] * fsij_dev[nb2][n_set][ni]

                    cos_ijk = xr_ij1[nb1][ni] * xr_ij1[nb2][ni] + xr_ij2[nb1][ni] * xr_ij2[nb2][ni] + \
                              xr_ij3[nb1][ni] * xr_ij3[nb2][ni]
                    cos2_ijk = pow(cos_ijk, 2)
                    Pl2 = 1.5 * cos2_ijk - 0.5
                    Pl4 = (4.375 * cos2_ijk - 3.750) * cos2_ijk + 0.375
                    Pl6 = ((14.4375 * cos2_ijk - 19.6875) * cos2_ijk + 6.5625) * cos2_ijk - 0.3125

                    Gi_sum0 = Gi_sum0 + fsij_fsik
                    Gi_sum1 = Gi_sum1 + cos_ijk * fsij_fsik
                    Gi_sum2 = Gi_sum2 + Pl2 * fsij_fsik
                    Gi_sum3 = Gi_sum3 + Pl4 * fsij_fsik
                    Gi_sum4 = Gi_sum4 + Pl6 * fsij_fsik
                # ! do nb2 = 1, max_nbrs !
            # ! do nb1 = 1, max_nbrs !

            # ! Calc. global G-vector !
            ind_set = n_set * 5 - 4
            Gi_dev[ind_set][ni] = Gi_sum0
            Gi_dev[ind_set + 1][ni] = Gi_sum1
            Gi_dev[ind_set + 2][ni] = Gi_sum2
            Gi_dev[ind_set + 3][ni] = Gi_sum3
            Gi_dev[ind_set + 4][ni] = Gi_sum4

    # ! --- Start III loop over ni atoms --- !

    for ni in range(1, sim_box.natoms + 1):  # ! loop over atoms !
        for nb1 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !
            rij1 = 1.0 / xr_ij0[nb1][ni]
            for n_set in range(1, n_set_ann + 1):
                Gi_sum_01, Gi_sum_02, Gi_sum_03 = 0.0, 0.0, 0.0
                Gi_sum_11, Gi_sum_12, Gi_sum_13 = 0.0, 0.0, 0.0
                Gi_sum_21, Gi_sum_22, Gi_sum_23 = 0.0, 0.0, 0.0
                Gi_sum_31, Gi_sum_32, Gi_sum_33 = 0.0, 0.0, 0.0
                Gi_sum_41, Gi_sum_42, Gi_sum_43 = 0.0, 0.0, 0.0

                fsij = fsij_dev[nb1][n_set][ni]

                for nb2 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !

                    fsik = 2.0 * fsij_dev[nb2][n_set][ni]

                    cos_ijk = xr_ij1[nb1][ni] * xr_ij1[nb2][ni] + xr_ij2[nb1][ni] * xr_ij2[nb2][ni] + \
                              xr_ij3[nb1][ni] * xr_ij3[nb2][ni]
                    cos2_ijk = pow(cos_ijk, 2)

                    dcos_ijk[1] = (xr_ij1[nb2][ni] - xr_ij1[nb1][ni] * cos_ijk) * rij1
                    dcos_ijk[2] = (xr_ij2[nb2][ni] - xr_ij2[nb1][ni] * cos_ijk) * rij1
                    dcos_ijk[3] = (xr_ij3[nb2][ni] - xr_ij3[nb1][ni] * cos_ijk) * rij1

                    Gi_sum_01 = Gi_sum_01 + fsik * dfs_rij_3D1[n_set][nb1][ni]
                    Gi_sum_02 = Gi_sum_02 + fsik * dfs_rij_3D2[n_set][nb1][ni]
                    Gi_sum_03 = Gi_sum_03 + fsik * dfs_rij_3D3[n_set][nb1][ni]

                    dgij1 = cos_ijk * dfs_rij_3D1[n_set][nb1][ni] + fsij * dcos_ijk[1]
                    dgij2 = cos_ijk * dfs_rij_3D2[n_set][nb1][ni] + fsij * dcos_ijk[2]
                    dgij3 = cos_ijk * dfs_rij_3D3[n_set][nb1][ni] + fsij * dcos_ijk[3]
                    Gi_sum_11 = Gi_sum_11 + fsik * dgij1
                    Gi_sum_12 = Gi_sum_12 + fsik * dgij2
                    Gi_sum_13 = Gi_sum_13 + fsik * dgij3

                    Pl2 = (1.5 * cos2_ijk - 0.5)
                    dPl2 = 3.0 * cos_ijk
                    dgij1 = Pl2 * dfs_rij_3D1[n_set][nb1][ni] + fsij * dPl2 * dcos_ijk[1]
                    dgij2 = Pl2 * dfs_rij_3D2[n_set][nb1][ni] + fsij * dPl2 * dcos_ijk[2]
                    dgij3 = Pl2 * dfs_rij_3D3[n_set][nb1][ni] + fsij * dPl2 * dcos_ijk[3]
                    Gi_sum_21 = Gi_sum_21 + fsik * dgij1
                    Gi_sum_22 = Gi_sum_22 + fsik * dgij2
                    Gi_sum_23 = Gi_sum_23 + fsik * dgij3

                    Pl4 = (4.375 * cos2_ijk - 3.75) * cos2_ijk + 0.375
                    dPl4 = (17.5 * cos2_ijk - 7.5) * cos_ijk
                    dgij1 = Pl4 * dfs_rij_3D1[n_set][nb1][ni] + fsij * dPl4 * dcos_ijk[1]
                    dgij2 = Pl4 * dfs_rij_3D2[n_set][nb1][ni] + fsij * dPl4 * dcos_ijk[2]
                    dgij3 = Pl4 * dfs_rij_3D3[n_set][nb1][ni] + fsij * dPl4 * dcos_ijk[3]
                    Gi_sum_31 = Gi_sum_31 + fsik * dgij1
                    Gi_sum_32 = Gi_sum_32 + fsik * dgij2
                    Gi_sum_33 = Gi_sum_33 + fsik * dgij3

                    Pl6 = ((14.4375 * cos2_ijk - 19.6875) * cos2_ijk + 6.5625) * cos2_ijk - 0.3125
                    dPl6 = (86.625 * pow(cos2_ijk, 2) - 78.75 * cos2_ijk + 13.125) * cos_ijk
                    dgij1 = Pl6 * dfs_rij_3D1[n_set][nb1][ni] + fsij * dPl6 * dcos_ijk[1]
                    dgij2 = Pl6 * dfs_rij_3D2[n_set][nb1][ni] + fsij * dPl6 * dcos_ijk[2]
                    dgij3 = Pl6 * dfs_rij_3D3[n_set][nb1][ni] + fsij * dPl6 * dcos_ijk[3]
                    Gi_sum_41 = Gi_sum_41 + fsik * dgij1
                    Gi_sum_42 = Gi_sum_42 + fsik * dgij2
                    Gi_sum_43 = Gi_sum_43 + fsik * dgij3

                icol = n_set * 5 - 4

                # !if(iPOT_file_ver.gt.0) then
                # ! Gi_fact0 = 1.d0/(Gi_dev(icol,ni) + 0.5d0)
                # ! Gi_fact1 = 1.d0/(Gi_dev(icol+1,ni) + 0.5d0)
                # ! Gi_fact2 = 1.d0/(Gi_dev(icol+2,ni) + 0.5d0)
                # ! Gi_fact3 = 1.d0/(Gi_dev(icol+3,ni) + 0.5d0)
                # ! Gi_fact4 = 1.d0/(Gi_dev(icol+4,ni) + 0.5d0)
                # !else
                # ! Gi_fact0 = 1.d0; Gi_fact1 = 1.d0; Gi_fact2 = 1.d0;
                # ! Gi_fact3 = 1.d0; Gi_fact4 = 1.d0;
                # !endif

                Gi_3D_dev1[icol][nb1][ni] = Gi_sum_01
                Gi_3D_dev1[icol + 1][nb1][ni] = Gi_sum_11
                Gi_3D_dev1[icol + 2][nb1][ni] = Gi_sum_21
                Gi_3D_dev1[icol + 3][nb1][ni] = Gi_sum_31
                Gi_3D_dev1[icol + 4][nb1][ni] = Gi_sum_41

                Gi_3D_dev2[icol][nb1][ni] = Gi_sum_02
                Gi_3D_dev2[icol + 1][nb1][ni] = Gi_sum_12
                Gi_3D_dev2[icol + 2][nb1][ni] = Gi_sum_22
                Gi_3D_dev2[icol + 3][nb1][ni] = Gi_sum_32
                Gi_3D_dev2[icol + 4][nb1][ni] = Gi_sum_42

                Gi_3D_dev3[icol][nb1][ni] = Gi_sum_03
                Gi_3D_dev3[icol + 1][nb1][ni] = Gi_sum_13
                Gi_3D_dev3[icol + 2][nb1][ni] = Gi_sum_23
                Gi_3D_dev3[icol + 3][nb1][ni] = Gi_sum_33
                Gi_3D_dev3[icol + 4][nb1][ni] = Gi_sum_43

    if ireport > 0:
        time_Gi = time.time()
        time_Gi_total = time_Gi_total + (time_Gi - cpu_start)

    for ni in range(1, sim_box.natoms + 1):
        if pot_module.iPOT_file_ver > 0:
            for icol in range(1, mG_dim + 1):
                U1[icol] = math.log(Gi_dev[icol][ni] + 0.5)
        else:
            for icol in range(1, mG_dim + 1):
                U1[icol] = Gi_dev[icol][ni]

        # ! --- Gis are done here for atom ni --- !
        # ! --- Start NN on atom ni --- !

        # ! -- Input Layer --- !

        for iraw in range(1, Nodes_of_layer[2] + 1):  # ! 1.. 20 !
            U_vect = B1_ann[iraw]
            for icol in range(1, Nodes_of_layer[1] + 1):  # ! 1.. 60 !
                U_vect = U_vect + U1[icol] * W1_ann[icol][iraw]
            U2[iraw] = 1.0 / (1.0 + math.exp(-U_vect)) + ActFunc_shift
            expU = math.exp(-U_vect)
            dfuN_dev[iraw][1][ni] = expU / pow(1.0 + expU, 2)

        # ! -- Hidden Layers --- !

        for layer in range(2, net_layers - 2 + 1):
            NLcurr = Nodes_of_layer[layer]
            NLnext = Nodes_of_layer[layer + 1]
            for i in range(1, NLcurr + 1):
                U1[i] = U2[i]
            for iraw in range(1, NLnext + 1):  # ! 1.. 20 !
                U_vect = B2_ann[iraw][layer]
                for icol in range(NLcurr + 1):  # ! 1.. 20 !
                    U_vect = U_vect + U1[icol] * W2_ann[icol][iraw][layer]
                U2[iraw] = 1.0 / (1.0 + math.exp(-U_vect)) + ActFunc_shift
                expU = math.exp(-U_vect)
                dfuN_dev[iraw][layer][ni] = expU / pow(1.0 + expU, 2)

        # ! -- Output Layer --- !

        U3_vect = None

        for iraw in range(1, Nodes_of_layer[net_layers] + 1):  # ! 1.. 8 !
            U3_vect = B3_ann[iraw]
            for icol in range(1, Nodes_of_layer[net_layers - 1] + 1):  # ! 1.. 20 !
                U3_vect = U3_vect + U2[icol] * W3_ann[icol][iraw]

        atoms.Ep_of[ni] = 2.0 * U3_vect

        # ! Twice the individual atomic energy  !
        # ! Devided by 2 later in MSR for	   !
        # ! compatibility with other potentials !

    for ni in range(1, sim_box.natoms + 1):
        for nb1 in range(1, atoms.max_nbrs + 1):

            # ! --- DO ANN for each (i-j) pair using Gis as input vectors --- !

            # ! --- Input Layer --- !

            for iraw in range(1, Nodes_of_layer[2] + 1):  # ! 1.. 20 !
                U_x1, U_x2, U_x3 = 0.0, 0.0, 0.0

                for icol in range(1, Nodes_of_layer[1] + 1):  # ! 1.. 60 !
                    U_x1 = U_x1 + Gi_3D_dev1[icol][nb1][ni] * W1_ann[icol][iraw]
                    U_x2 = U_x2 + Gi_3D_dev2[icol][nb1][ni] * W1_ann[icol][iraw]
                    U_x3 = U_x3 + Gi_3D_dev3[icol][nb1][ni] * W1_ann[icol][iraw]
                U2f1[iraw] = U_x1 * dfuN_dev[iraw][1][ni]
                U2f2[iraw] = U_x2 * dfuN_dev[iraw][1][ni]
                U2f3[iraw] = U_x3 * dfuN_dev[iraw][1][ni]

            # ! --- Hidden Layers --- !

            for layer in range(2, net_layers - 2 + 1):
                NLcurr = Nodes_of_layer[layer]
                NLnext = Nodes_of_layer[layer + 1]

                for i in range(1, NLcurr + 1):
                    U1f1[i] = U2f1[i]
                    U1f2[i] = U2f2[i]
                    U1f3[i] = U2f3[i]

                for iraw in range(1, NLnext + 1):  # ! 1.. 20 !
                    U_x1, U_x2, U_x3 = 0.0, 0.0, 0.0

                    for icol in range(1, NLcurr + 1):  # ! 1.. 20 !
                        U_x1 = U_x1 + U1f1[icol] * W2_ann[icol][iraw][layer]
                        U_x2 = U_x2 + U1f2[icol] * W2_ann[icol][iraw][layer]
                        U_x3 = U_x3 + U1f3[icol] * W2_ann[icol][iraw][layer]
                    U2f1[iraw] = U_x1 * dfuN_dev[iraw][layer][ni]
                    U2f2[iraw] = U_x2 * dfuN_dev[iraw][layer][ni]
                    U2f3[iraw] = U_x3 * dfuN_dev[iraw][layer][ni]

            # ! --- Output Layer --- !

            for iraw in range(1, Nodes_of_layer[net_layers] + 1):  # ! 1.. 1 !
                U_x1, U_x2, U_x3 = 0.0, 0.0, 0.0

                for icol in range(1, Nodes_of_layer[net_layers - 1] + 1):  # ! 1.. 20 !
                    U_x1 = U_x1 + U2f1[icol] * W3_ann[icol][iraw]
                    U_x2 = U_x2 + U2f2[icol] * W3_ann[icol][iraw]
                    U_x3 = U_x3 + U2f3[icol] * W3_ann[icol][iraw]
                dBOP_param_dxij_[1][iraw][nb1][ni] = U_x1
                dBOP_param_dxij_[2][iraw][nb1][ni] = U_x2
                dBOP_param_dxij_[3][iraw][nb1][ni] = U_x3

    # ! --- End of ANN for each (i-j) pair using gij as input vectors --- !

    # ! --- Calc Actual Force Vectors --- !

    ecohe = 0.0

    for ni in range(1, sim_box.natoms + 1):  # ! loop over atoms !
        fr = [0.0] * (3 + 1)

        for nb1 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !
            nj = atoms.nbr_list[nb1][ni]

            # !call mm_prefetch(dBOP_param_dxij_(1,1,nb1,nj),1)
            # !call mm_prefetch(dBOP_param_dxij_(1,1,nb2,nj),1)

            ij_delta = min(abs(nj - ni), 1)  # ! 0 if ni=nj; 1 if ni=/=nj !
            nbrs_of_j = ij_delta * atoms.max_nbrs  # ! rij < Rc !

            nbi = nb1

            for nb2 in range(1, nbrs_of_j + 1):  # ! search for i as a neighbor of j !
                nj2 = atoms.nbr_list[nb2][nj]
                if nj2 == ni:
                    nbi = nb2  # ! i is the nbi-th nbr of j !

            for i in range(1, 3 + 1):
                fij[i] = dBOP_param_dxij_[i][1][nb1][ni] - dBOP_param_dxij_[i][1][nbi][nj]
            for i in range(1, 3 + 1):
                fr[i] = fr[i] + fij[i]

        for i in range(1, 3 + 1):
            atoms.frr[i][ni] = fr[i]

        ecohe = ecohe + 0.50 * atoms.Ep_of[ni]

    time_final = time.time()
    print('Delta time =', time_final - time_initial, ' s (of each Frc_ANN_OMP call')

    return ecohe
    # ! End of Frc_ANN_OMP !

# ! ---------------------------------------------------------------------
# ! Calculates atomic forces using OpenACC
# ! ---------------------------------------------------------------------

def Frc_ANN_ACC():
    # u use sys_ACC
    # u use atoms
    global Max_net_layers, n_set_ann, net_atom_types, iflag_ann, net_layers, net_in, net_out, mG_dim, max_tri_index
    global net_layers, net_in, net_out, mG_dim, max_tri_index, Rc_ann, d_ann, d4_ann, Gauss_ann, range_min_ann
    global ActFunc_shift, Nodes_of_layer, r0_value, r0G_value, Gi_atom, dG_i, Gi_list, Gi_new
    global U0, U1, U2, W1_ann, W3_ann, W2_ann, B1_ann, B3_ann, B2_ann, dBOP_param_dxij, buf_ann
    global r0Rc, r0pRc, U1f1, U2f1, U1f2, U2f2, U1f3, U2f3, Gi_dev, xr_ij0, xr_ij1, xr_ij2, xr_ij3, xr_ij_dev, fsij_dev, dfuN_dev
    global Gi_3D_dev1, Gi_3D_dev2, Gi_3D_dev3, dfs_rij_3D1, dfs_rij_3D2, dfs_rij_3D3, dfs_rij_3D, Gi_3D_dev, dBOP_param_dxij_
    global ireport


    iMb = 1024 ** 2  # etait integer kind 8 sera number (long)
    memory_request = -1  # integer kind 8
    memory1 = -1  # integer

    dgij = [0.0] * 4  # vecteur taille 3, double
    xr_ij = [0.0] * 4  # vecteur taille 3, double
    xr_ik = [0.0] * 4  # vecteur taille 3, double
    dcos_ijk = [0.0] * 4  # vecteur taille 3, double

    r0Rc = []  # vecteur taille n_set_ann, double
    r0pRc = []  # vecteur taille n_set_ann, double

    U1f = []  # 2 dim , double
    U2f = []  # 2 dim , double
    Gi_dev = []  # 2 dim , double

    xr_ij_dev = []  # 3 dim , double
    fsij_dev = []  # 3 dim , double
    dfuN_dev = []  # 3 dim , double

    dfs_rij_3D = []  # 4 dim , double
    Gi_3D_dev = []  # 4 dim , double
    dBOP_param_dxij_ = []  # 4 dim , double

    ialloc = []
    for i in range(10 + 1): ialloc.append(0)

    if sim_box.ihalt != 0:
        return

    """
    write(6,*)'Frc_ANN_ACC: natoms=',natoms,' max_nbrs=',max_nbrs
    """

    max_ls = 1
    for i in range(2, net_layers - 1 + 1):
        if Nodes_of_layer[i] > max_ls:
            max_ls = Nodes_of_layer[i]

    nBOP_params = Nodes_of_layer[net_layers]
    """
    ! Those are replacements of ACC_* equivalents	!
    ! redefined in pgmc_sys_ACC.f and pgmc_sys_OMP.f !
    """

    # gpu My_GPU_free_mem = GET_GPU_FREE_MEM(My_GPU) / iMb  #! in Mbs !
    """
    !	 write(6,*)'1: My_GPU_free_mem=',My_GPU_free_mem
    """
    memory1 = 4 * (2 * atoms.max_nbrs + n_set_ann * atoms.max_nbrs) + mG_dim + 3 * (
                mG_dim + nBOP_params) * atoms.max_nbrs + max_ls * net_layers
    memory_request = memory1 * sim_box.natoms * 8
    memory_request = memory_request / iMb  # ! in Mbs !
    memory_request = memory_request * 1.3  # ! safe margin !

    if nACC_devices > 0:
        if memory_request > My_GPU_free_mem:
            print("Erreur ligne 914 _ANN: memory_request>My_GPU_free_mem")
            print(memory_request , " " , My_GPU_free_mem , " " , My_GPU , " " , sim_box.mynod)
            sim_box.ihalt = 1
            return

    # f 13   format(/,'ERROR: Requested memory in Frc_ANN_ACC is:',i6,
    # f 1 ' Mb,',/,'which exceeds the available GPU memory of',i6,
    # f 2 ' Mb on GPU:',i2,' on node:',i5,/,'Increase number of nodes.')

    for i in range(4):  # ATTENTION LA POS 0 EST UTILE!!!
        xr_ij_dev.append([])
        for j in range(atoms.max_nbrs + 1):
            xr_ij_dev[i].append([])
            for k in range(sim_box.natoms + 1):
                xr_ij_dev[i][j].append(0.0)

    for i in range(atoms.max_nbrs + 1):
        fsij_dev.append([])
        for j in range(n_set_ann + 1):
            fsij_dev[i].append([])
            for k in range(sim_box.natoms + 1):
                fsij_dev[i][j].append(0.0)

    for i in range(3 + 1):
        dfs_rij_3D.append([])
        for j in range(atoms.max_nbrs + 1):
            dfs_rij_3D[i].append([])
            for k in range(sim_box.natoms + 1):
                dfs_rij_3D[i][j].append([])
                for l in range(n_set_ann + 1):
                    dfs_rij_3D[i][j][k].append(0.0)

    for i in range(mG_dim + 1):
        Gi_dev.append([])
        for j in range(sim_box.natoms + 1):
            Gi_dev[i].append(0.0)

    print("Message debug 824 len(Gi_dev) ", Gi_dev)

    for i in range(3 + 1):
        Gi_3D_dev.append([])
        for j in range(mG_dim + 1):
            Gi_3D_dev[i].append([])
            for k in range(atoms.max_nbrs + 1):
                Gi_3D_dev[i][j].append([])
                for l in range(sim_box.natoms + 1):
                    Gi_3D_dev[i][j][k].append(0.0)

    for i in range(max_ls + 1):
        dfuN_dev.append([])
        for j in range(net_layers - 2 + 1):
            dfuN_dev[i].append([])
            for k in range(sim_box.natoms + 1):
                dfuN_dev[i][j].append(0.0)

    for i in range(3 + 1):
        U1f.append([])
        for j in range(max_ls + 1):
            U1f[i].append(0.0)

    for i in range(3 + 1):
        U2f.append([])
        for j in range(max_ls + 1):
            U2f[i].append(0.0)

    for i in range(3 + 1):
        dBOP_param_dxij_.append([])
        for j in range(nBOP_params + 1):
            dBOP_param_dxij_[i].append([])
            for k in range(atoms.max_nbrs + 1):
                dBOP_param_dxij_[i][j].append([])
                for l in range(sim_box.natoms + 1):
                    dBOP_param_dxij_[i][j][k].append(0.0)

    ierr = 0
    for i in range(1, 10 + 1):
        ierr = ierr + ialloc[i]
    if ierr != 0:
        print('ERROR allocating x in Frc_ANN_ACC')
        sim_box.ihalt = 1
        return

    ecohe = 0.0
    Rc = Rc_ann
    Rc2 = Rc_ann ** 2
    d4 = d4_ann
    Sigma2 = 2.0 * Gauss_ann ** 2

    r0Rc = []
    r0pRc = []
    for i in range(len(r0_value)):  # len = n_set_ann+1 normalement
        # r0Rc(:) = Rc_ann*r0_value(:)
        # r0pRc(:) = Rc_ann + r0_value(:)
        r0Rc.append(Rc_ann * r0_value[i])
        r0pRc.append(Rc_ann * r0_value[i])

    sim_box.h11 = sim_box.h[1][1]
    sim_box.h12 = sim_box.h[1][2]
    sim_box.h13 = sim_box.h[1][3]
    sim_box.h22 = sim_box.h[2][2]
    sim_box.h23 = sim_box.h[2][3]
    sim_box.h33 = sim_box.h[3][3]

    """
    !
    ! --- Start I loop over ni atoms ---
    !
    """

    for ni in range(1, sim_box.natoms + 1):  # loop over atoms
        for nb1 in range(1, atoms.max_nbrs + 1):  # Loop I: i - j bond

            nj = nbr_list[nb1, ni]
            ij_delta = 1 - min([abs(nj - ni), 1])  # ! 1 if ni=nj; 0 if ni=/=nj !

            sxn = atoms.sx[ni]
            syn = atoms.sy[ni]
            szn = atoms.sz[ni]

            sx0 = atoms.sx[nj] - sxn
            sx0 = sx0 - dnint[sx0]  # ! make periodic along X !
            sy0 = atoms.sy[nj] - syn
            sy0 = sy0 - dnint[sy0]  # ! make periodic along Y !
            sz0 = atoms.sz[nj] - szn
            sz0 = sz0 - dnint[sz0]  # ! make periodic along Z !

            xij = sim_box.h11 * sx0 + sim_box.h12 * sy0 + sim_box.h13 * sz0
            yij = sim_box.h22 * sy0 + sim_box.h23 * sz0
            zij = sim_box.h33 * sz0
            r2ij = xij ** 2 + yij ** 2 + zij ** 2 + ij_delta * Rc2  # ! rij+Rc when i=j
            rij = sqrt[r2ij]
            r1ij = 1.0 / rij

            xr_ij[1] = xij * r1ij
            xr_ij[2] = yij * r1ij
            xr_ij[3] = zij * r1ij
            xr_ij_dev[0][nb1][ni] = rij
            xr_ij_dev[1][nb1][ni] = xr_ij[1]
            xr_ij_dev[2][nb1][ni] = xr_ij[2]
            xr_ij_dev[3][nb1][ni] = xr_ij[3]

            RcRij = max([Rc - rij, 0.0])
            RcRij3 = RcRij ** 3
            RcRij4 = RcRij3 * RcRij
            RcRij5 = RcRij4 * RcRij
            fc_rij = 0.25 * RcRij4 / (d4 + RcRij4)
            denominator = (Gauss_ann * (d4 + RcRij4)) ** 2

            for n_set in range(1, n_set_ann + 1):
                r0 = r0_value[n_set]  # ! do not use r0G_value() here !
                RijRo = rij - r0
                expRijRo = math.exp(-(RijRo / Gauss_ann) ** 2)
                fsij_dev[nb1][n_set][ni] = fc_rij * expRijRo
                dfsij = (r0Rc[n_set] + rij * (rij - r0pRc[n_set]) - Sigma2) * d4 - -RijRo * RcRij5
                dfsij = 0.5 * dfsij * RcRij3 * expRijRo / denominator  # ! *2/4 !
                dfs_rij_3D[1][nb1][ni][n_set] = dfsij * xr_ij[1]  # ! dfs_ij*xij/rij !
                dfs_rij_3D[2][nb1][ni][n_set] = dfsij * xr_ij[2]  # ! dfs_ij*xij/rij !
                dfs_rij_3D[3][nb1][ni][n_set] = dfsij * xr_ij[3]  # ! dfs_ij*xij/rij !

    """
    !
    ! --- Start II loop over ni atoms ---
    !
    """
    for ni in range(1, sim_box.natoms + 1):  # ! loop over atoms !
        for n_set in range(1, n_set_ann + 1):
            Gi_sum0 = 0.0
            Gi_sum1 = 0.0
            Gi_sum2 = 0.0
            Gi_sum3 = 0.0
            Gi_sum4 = 0.0
            for nb1 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !
                for nb2 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !
                    fsij_fsik = fsij_dev[nb1][n_set][ni] * fsij_dev[nb2][n_set][ni]

                    xr_ij[1] = xr_ij_dev[1][nb1][ni]
                    xr_ij[2] = xr_ij_dev[2][nb1][ni]
                    xr_ij[3] = xr_ij_dev[3][nb1][ni]

                    xr_ik[1] = xr_ij_dev[1][nb2][ni]
                    xr_ik[2] = xr_ij_dev[2][nb2][ni]
                    xr_ik[3] = xr_ij_dev[3][nb2][ni]

                    cos_ijk = xr_ij[1] * xr_ik[1] + xr_ij[2] * xr_ik[2] + xr_ij[3] * xr_ik[3]
                    cos2_ijk = cos_ijk ** 2
                    Pl2 = 1.5 * cos2_ijk - 0.5
                    Pl4 = (4.375 * cos2_ijk - 3.75) * cos2_ijk + 0.375
                    pl6 = ((14.4375 * cos2_ijk - 19.6875) * cos2_ijk + 6.5625) * cos2_ijk - 0.3125

                    Gi_sum0 = Gi_sum0 + fsij_fsik
                    Gi_sum1 = Gi_sum1 + cos_ijk * fsij_fsik
                    Gi_sum2 = Gi_sum2 + Pl2 * fsij_fsik
                    Gi_sum3 = Gi_sum3 + Pl4 * fsij_fsik
                    Gi_sum4 = Gi_sum4 + Pl6 * fsij_fsik
            """
            ! Calc. global G-vector !
            """
            ind_set = n_set * 5 - 4
            Gi_dev[ind_set][ni] = Gi_sum0
            Gi_dev[ind_set + 1][ni] = Gi_sum1
            Gi_dev[ind_set + 2][ni] = Gi_sum2
            Gi_dev[ind_set + 3][ni] = Gi_sum3
            Gi_dev[ind_set + 4][ni] = Gi_sum4
    """
    ! --- Start III loop over ni atoms ---
    """
    for ni in range(1, sim_box.natoms + 1):  # loop over atoms
        for nb1 in range(1, atoms.max_nbrs + 1):  # Loop I: i - j bond
            for n_set in range(1, n_set_ann + 1):

                Gi_sum0_x = 0.0
                Gi_sum0_y = 0.0
                Gi_sum0_z = 0.0
                Gi_sum1_x = 0.0
                Gi_sum1_y = 0.0
                Gi_sum1_z = 0.0
                Gi_sum2_x = 0.0
                Gi_sum2_y = 0.0
                Gi_sum2_z = 0.0
                Gi_sum3_x = 0.0
                Gi_sum3_y = 0.0
                Gi_sum3_z = 0.0
                Gi_sum4_x = 0.0
                Gi_sum4_y = 0.0
                Gi_sum4_z = 0.0

                rij = xr_ij_dev[0][nb1][ni]
                xr_ij[1] = xr_ij_dev[1][nb1][ni]
                xr_ij[2] = xr_ij_dev[2][nb1][ni]
                xr_ij[3] = xr_ij_dev[3][nb1][ni]
                rij1 = 1.0 / rij

                fsij = fsij_dev[nb1][n_set][ni]

                for nb2 in range(1, atoms.max_nbrs + 1):
                    fsik = 2.0 * fsij_dev[nb2][n_set][ni]

                    xr_ik[1] = xr_ij_dev[1][nb2][ni]
                    xr_ik[2] = xr_ij_dev[2][nb2][ni]
                    xr_ik[3] = xr_ij_dev[3][nb2][ni]

                    cos_ijk = xr_ij[1] * xr_ik[1] + xr_ij[2] * xr_ik[2] + xr_ij[3] * xr_ik[3]
                    cos2_ijk = cos_ijk ** 2
                    dcos_ijk[1] = [xr_ik[1] - xr_ij[1] * cos_ijk] * rij1
                    dcos_ijk[2] = [xr_ik[2] - xr_ij[2] * cos_ijk] * rij1
                    dcos_ijk[3] = [xr_ik[3] - xr_ij[3] * cos_ijk] * rij1

                    """		   
                    ! (xik/rik - xij/rij*cos_ijk) / rij  !
                    """

                    Gi_sum0_x = Gi_sum0_x + fsik * dfs_rij_3D[1][nb1][ni][n_set]
                    Gi_sum0_y = Gi_sum0_y + fsik * dfs_rij_3D[2][nb1][ni][n_set]
                    Gi_sum0_z = Gi_sum0_z + fsik * dfs_rij_3D[3][nb1][ni][n_set]

                    dgij[1] = cos_ijk * dfs_rij_3D[1][nb1][ni][n_set] + fsij * dcos_ijk[1]
                    dgij[2] = cos_ijk * dfs_rij_3D[2][nb1][ni][n_set] + fsij * dcos_ijk[2]
                    dgij[3] = cos_ijk * dfs_rij_3D[3][nb1][ni][n_set] + fsij * dcos_ijk[3]
                    Gi_sum1_x = Gi_sum1_x + fsik * dgij[1]
                    Gi_sum1_y = Gi_sum1_y + fsik * dgij[2]
                    Gi_sum1_z = Gi_sum1_z + fsik * dgij[3]

                    Pl2 = (1.50 * cos2_ijk - 0.50)
                    dPl2 = 3.0 * cos_ijk
                    dgij[1] = Pl2 * dfs_rij_3D[1][nb1][ni][n_set] + fsij * dPl2 * dcos_ijk[1]
                    dgij[2] = Pl2 * dfs_rij_3D[2][nb1][ni][n_set] + fsij * dPl2 * dcos_ijk[2]
                    dgij[3] = Pl2 * dfs_rij_3D[3][nb1][ni][n_set] + fsij * dPl2 * dcos_ijk[3]
                    Gi_sum2_x = Gi_sum2_x + fsik * dgij[1]
                    Gi_sum2_y = Gi_sum2_y + fsik * dgij[2]
                    Gi_sum2_z = Gi_sum2_z + fsik * dgij[3]

                    Pl4 = (4.3750 * cos2_ijk - 3.750) * cos2_ijk + 0.3750
                    dPl4 = (17.50 * cos2_ijk - 7.50) * cos_ijk
                    dgij[1] = Pl4 * dfs_rij_3D[1][nb1][ni][n_set] + fsij * dPl4 * dcos_ijk[1]
                    dgij[2] = Pl4 * dfs_rij_3D[2][nb1][ni][n_set] + fsij * dPl4 * dcos_ijk[2]
                    dgij[3] = Pl4 * dfs_rij_3D[3][nb1][ni][n_set] + fsij * dPl4 * dcos_ijk[3]
                    Gi_sum3_x = Gi_sum3_x + fsik * dgij[1]
                    Gi_sum3_y = Gi_sum3_y + fsik * dgij[2]
                    Gi_sum3_z = Gi_sum3_z + fsik * dgij[3]

                    Pl6 = ((14.43750 * cos2_ijk - 19.68750) * cos2_ijk + 6.5625) * cos2_ijk - 0.31250
                    dPl6 = (86.6250 * cos2_ijk ** 2 - 78.750 * cos2_ijk + 13.1250) * cos_ijk
                    dgij[1] = Pl6 * dfs_rij_3D[1][nb1][ni][n_set] + fsij * dPl6 * dcos_ijk[1]
                    dgij[2] = Pl6 * dfs_rij_3D[2][nb1][ni][n_set] + fsij * dPl6 * dcos_ijk[2]
                    dgij[3] = Pl6 * dfs_rij_3D[3][nb1][ni][n_set] + fsij * dPl6 * dcos_ijk[3]
                    Gi_sum4_x = Gi_sum4_x + fsik * dgij[1]
                    Gi_sum4_y = Gi_sum4_y + fsik * dgij[2]
                    Gi_sum4_z = Gi_sum4_z + fsik * dgij[3]

                icol = n_set * 5 - 4

                if pot_module.iPOT_file_ver > 0:
                    Gi_fact0 = 1.0 / (Gi_ev[icol, ni] + 0.50)
                    Gi_fact1 = 1.0 / (Gi_ev[icol + 1, ni] + 0.50)
                    Gi_fact2 = 1.0 / (Gi_ev[icol + 2, ni] + 0.50)
                    Gi_fact3 = 1.0 / (Gi_ev[icol + 3, ni] + 0.50)
                    Gi_fact4 = 1.0 / (Gi_ev[icol + 4, ni] + 0.50)
                else:
                    Gi_fact0 = 1.0
                    Gi_fact1 = 1.0
                    Gi_fact2 = 1.0
                    Gi_fact3 = 1.0
                    Gi_fact4 = 1.0

                Gi_3D_dev[1][icol][nb1][ni] = Gi_sum0_x * Gi_fact0
                Gi_3D_dev[2][icol][nb1][ni] = Gi_sum0_y * Gi_fact0
                Gi_3D_dev[3][icol][nb1][ni] = Gi_sum0_z * Gi_fact0

                Gi_3D_dev[1][icol + 1][nb1][ni] = Gi_sum1_x * Gi_fact1
                Gi_3D_dev[2][icol + 1][nb1][ni] = Gi_sum1_y * Gi_fact1
                Gi_3D_dev[3][icol + 1][nb1][ni] = Gi_sum1_z * Gi_fact1

                Gi_3D_dev[1][icol + 2][nb1][ni] = Gi_sum2_x * Gi_fact2
                Gi_3D_dev[2][icol + 2][nb1][ni] = Gi_sum2_y * Gi_fact2
                Gi_3D_dev[3][icol + 2][nb1][ni] = Gi_sum2_z * Gi_fact2

                Gi_3D_dev[1][icol + 3][nb1][ni] = Gi_sum3_x * Gi_fact3
                Gi_3D_dev[2][icol + 3][nb1][ni] = Gi_sum3_y * Gi_fact3
                Gi_3D_dev[3][icol + 3][nb1][ni] = Gi_sum3_z * Gi_fact3

                Gi_3D_dev[1][icol + 4][nb1][ni] = Gi_sum4_x * Gi_fact4
                Gi_3D_dev[2][icol + 4][nb1][ni] = Gi_sum4_y * Gi_fact4
                Gi_3D_dev[3][icol + 4][nb1][ni] = Gi_sum4_z * Gi_fact4

    for ni in range(1, sim_box.natoms + 1):

        if pot_module.iPOT_file_ver > 0:
            for icol in range(1, mG_dim + 1):
                U1[icol] = math.log(Gi_dev[icol][ni] + 0.5)
        else:
            for icol in range(1, mG_dim + 1):
                U1[icol] = Gi_dev[icol][ni]

        """
        ! --- Gis are done here for atom ni ---
        ! --- Start NN on atom ni ---

        ! -- Input Layer ---
        """

        for iraw in range(1, Nodes_of_layer[2] + 1):  # ! 1.. 20 !
            U_vect = B1_ann[iraw]
            for icol in range(1, Nodes_of_layer[1] + 1):  # ! 1.. 60 !
                U_vect = U_vect + U1[icol] * W1_ann[icol][iraw]
            U2[iraw] = 1.0 / (1.0 + math.exp(-U_vect)) + ActFunc_shift
            expU = math.exp(-U_vect)
            dfuN_dev[iraw][1][ni] = expU / (1.0 + expU) ** 2

        """
        !  -- Hidden Layers ---   
        """

        for layer in range(2, net_layers - 2 + 1):
            NLcurr = Nodes_of_layer[layer]
            NLnext = Nodes_of_layer[layer + 1]
            for iterator in range(1, NLcurr + 1):
                # U1(1:NLcurr)=U2(1:NLcurr)
                U1[iterator] = U2[iterator]

            for iraw in range(1, NLnext + 1):  # ! 1.. 20 !
                U_vect = B2_ann[iraw][layer]
                for icol in range(1, NLcurr + 1):  # ! 1.. 20 !
                    U_vect = U_vect + U1[icol] * W2_ann[icol][iraw][layer]
                U2[iraw] = 1.0 / (1.0 + math.exp(-U_vect)) + ActFunc_shift
                expU = math.exp[-U_vect]
                dfuN_dev[iraw][layer][ni] = expU / (1.0 + expU) ** 2

        """
        ! -- Output Layer ---	 
        """

        for iraw in range(1, Nodes_of_layer[net_layers] + 1):  # ! 1.. 8 !
            U3_vect = B3_ann[iraw]
            for icol in range(1, Nodes_of_layer[net_layers - 1] + 1):  # ! 1.. 20 !
                U3_vect = U3_vect + U2[icol] * W3_ann[icol][iraw]

        Ep_of[ni] = 2.0 * U3_vect
        """
        ! Twice the individual atomic energy  !
        ! Devided by 2 later in MSR for	   !
        ! compatibility with other potentials !
        """

    for ni in range(1, sim_box.natoms + 1):
        for nb1 in range(1, atoms.max_nbrs + 1):

            """
            ! --- DO ANN for each (i-j) pair using Gis as input vectors ---

            ! -- Input Layer ---  
            """

            for iraw in range(1, Nodes_of_layer[2] + 1):  # ! 1.. 20 !
                U_x = 0.0
                U_y = 0.0
                U_z = 0.0
                for icol in range(1, Nodes_of_layer[1] + 1):  # ! 1.. 60 !
                    U_x = U_x + Gi_3D_dev[1][icol][nb1][ni] * W1_ann[icol][iraw]
                    U_y = U_y + Gi_3D_dev[2][icol][nb1][ni] * W1_ann[icol][iraw]
                    U_z = U_z + Gi_3D_dev[3][icol][nb1][ni] * W1_ann[icol][iraw]
                U2f[1][iraw] = U_x * dfuN_dev[iraw][1][ni]
                U2f[2][iraw] = U_y * dfuN_dev[iraw][1][ni]
                U2f[3][iraw] = U_z * dfuN_dev[iraw][1][ni]

            """
            ! -- Hidden Layers --- 
            """
            for layer in range(2, net_layers - 2 + 1):
                NLcurr = Nodes_of_layer[layer]
                NLnext = Nodes_of_layer[layer + 1]

                for i in range(1, NLcurr + 1):
                    U1f[1][i] = U2f[1][i]
                    U1f[2][i] = U2f[2][i]
                    U1f[3][i] = U2f[3][i]

                for iraw in range(1, NLnext + 1):  # ! 1.. 20 !
                    U_x = 0.0
                    U_y = 0.0
                    U_z = 0.0

                    for icol in range(1, NLcurr + 1):  # ! 1.. 20 !
                        U_x = U_x + U1f[1][icol] * W2_ann[icol][iraw][layer]
                        U_y = U_y + U1f[2][icol] * W2_ann[icol][iraw][layer]
                        U_z = U_z + U1f[3][icol] * W2_ann[icol][iraw][layer]

                    U2f[1][iraw] = U_x * dfuN_dev[iraw][layer][ni]
                    U2f[2][iraw] = U_y * dfuN_dev[iraw][layer][ni]
                    U2f[3][iraw] = U_z * dfuN_dev[iraw][layer][ni]

            """
            ! -- Output Layer ---
            """

            for iraw in range(1, Nodes_of_layer[net_layers] + 1):  # ! 1.. 1 !

                U_x = 0.0
                U_y = 0.0
                U_z = 0.0

                for icol in range(1, Nodes_of_layer[net_layers - 1] + 1):  # ! 1.. 20 !
                    U_x = U_x + U2f[1][icol] * W3_ann[icol][iraw]
                    U_y = U_y + U2f[2][icol] * W3_ann[icol][iraw]
                    U_z = U_z + U2f[3][icol] * W3_ann[icol][iraw]

                dBOP_param_dxij_[1][iraw][nb1][ni] = U_x
                dBOP_param_dxij_[2][iraw][nb1][ni] = U_y
                dBOP_param_dxij_[3][iraw][nb1][ni] = U_z

    """
    ! --- End of ANN for each (i-j) pair using gij as input vectors ---

    ! --- Calc Actual Force Vectors ---
    """
    ecohe = 0.0

    for ni in range(1, sim_box.natoms + 1):  # ! loop over atoms !

        frr1 = 0.0
        frr2 = 0.0
        frr3 = 0.0

        for nb1 in range(1, atoms.max_nbrs + 1):  # ! Loop I: i - j bond !
            nj = nbr_list[nb1][ni]
            ij_delta = min([abs(nj - ni), 1])  # ! 0 if ni=nj; 1 if ni=/=nj !
            nbrs_of_j = ij_delta * atoms.max_nbrs  # ! rij < Rc !
            nbi = nb1

            for nb2 in range(1, nbrs_of_j + 1):  # ! search for i as a neighbor of j !
                nj2 = nbr_list[nb2][nj]
                if nj2 == ni:
                    nbi = nb2  # ! i is the nbi-th nbr of j !

            # ! Y3ij - Y3ji !
            fij1 = dBOP_param_dxij_[1][1][nb1][ni] - dBOP_param_dxij_[1][1][nbi][nj]
            fij2 = dBOP_param_dxij_[2][1][nb1][ni] - dBOP_param_dxij_[2][1][nbi][nj]
            fij3 = dBOP_param_dxij_[3][1][nb1][ni] - dBOP_param_dxij_[3][1][nbi][nj]
            frr1 = frr1 + fij1  # ! x,y,z forces !
            frr2 = frr2 + fij2  # ! x,y,z forces !
            frr3 = frr3 + fij3  # ! x,y],z forces !

        frr[1][ni] = frr1
        frr[2][ni] = frr2
        frr[3][ni] = frr3

        ecohe = ecohe + 0.5 * Ep_of[ni]

    xr_ij_dev = []
    fsij_dev = []
    dfs_rij_3D = []
    Gi_dev = []
    Gi_3D_dev = []
    dfuN_dev = []
    U1f = []
    U2f = []
    dBOP_param_dxij_ = []

    ierr = 0
    for i in range(1, 10 + 1):
        ierr = ierr + ialloc[i]
    if ierr != 0:
        print('ERROR deallocating x in Frc_ANN_ACC')
        sim_box.ihalt = 1

    return ecohe
    # ! End of Frc_ANN_ACC !

# !
# ! ---------------------------------------------------------------------
# !

def alloc_types_ANN():
    global Max_net_layers, n_set_ann, net_atom_types, iflag_ann, net_layers, net_in, net_out, mG_dim, max_tri_index
    global net_layers, net_in, net_out, mG_dim, max_tri_index, Rc_ann, d_ann, d4_ann, Gauss_ann, range_min_ann
    global ActFunc_shift, Nodes_of_layer, r0_value, r0G_value, Gi_atom, dG_i, Gi_list, Gi_new
    global U0, U1, U2, W1_ann, W3_ann, W2_ann, B1_ann, B3_ann, B2_ann, dBOP_param_dxij, buf_ann
    global r0Rc, r0pRc, U1f1, U2f1, U1f2, U2f2, U1f3, U2f3, Gi_dev, xr_ij0, xr_ij1, xr_ij2, xr_ij3, xr_ij_dev, fsij_dev, dfuN_dev
    global Gi_3D_dev1, Gi_3D_dev2, Gi_3D_dev3, dfs_rij_3D1, dfs_rij_3D2, dfs_rij_3D3, dfs_rij_3D, Gi_3D_dev, dBOP_param_dxij_
    global ireport


    ialloc = [0] * (8 + 1)

    max_cols = 1
    max_raws = 1
    for i in range(2, net_layers - 2 + 1):
        if Nodes_of_layer[i] > max_cols:
            max_cols = Nodes_of_layer[i]
        if Nodes_of_layer[i + 1] > max_raws:
            max_raws = Nodes_of_layer[i + 1]

    Ncols1 = Nodes_of_layer[1]
    Nraws1 = Nodes_of_layer[2]
    Ncols2 = max_cols
    Nraws2 = max_raws
    Ncols3 = Nodes_of_layer[net_layers - 1]
    Nraws3 = Nodes_of_layer[net_layers]

    nbuf_dim = Ncols1 * Nraws1 + (net_layers - 2) * max_cols * max_raws + Ncols3 * Nraws3

    W1_ann = [[0.0] * (Nraws1 + 1) for _ in range(Ncols1 + 1)]
    W2_ann = [[[0.0] * (net_layers - 2 + 1) for j in range(Nraws2 + 1)] for i in range(Ncols2 + 1)]
    W3_ann = [[0.0] * (Nraws3 + 1) for _ in range(Ncols3 + 1)]

    B1_ann = [0.0] * (Nraws1 + 1)
    B2_ann = [[0.0] * (net_layers - 2 + 1) for _ in range(Nraws2 + 1)]
    B3_ann = [0.0] * (Nraws3 + 1)
    buf_ann = [0.0] * (nbuf_dim + 1)

    ierror = 0
    for i in range(1, 7 + 1):
        ierror = ierror + ialloc[i]

    return ierror
    # ! End of alloc_types_ANN !

# !
# ! ---------------------------------------------------------------------
# !

def deall_types_ANN():

    global W1_ann, W2_ann, W3_ann, B1_ann, B2_ann, B3_ann, r0_value, r0G_value, buf_ann
    global ireport

    ialloc = [0] * (8 + 1)

    if W1_ann:
        W1_ann.clear()
    if W2_ann:
        W2_ann.clear()
    if W3_ann:
        W3_ann.clear()

    if B1_ann:
        B1_ann.clear()
    if B2_ann:
        B2_ann.clear()
    if B3_ann:
        B3_ann.clear()

    if r0_value:
        r0_value.clear()
    if r0G_value:
        r0G_value.clear()
    if buf_ann:
        buf_ann.clear()

    ierror = 0
    for i in range(1, 8 + 1):
        ierror = ierror + ialloc[i]

    return ierror
    # ! End of deall_types_ANN !


"""
!
! ---------------------------------------------------------------------
!					! alloc_atoms_ANN !
"""


def alloc_atoms_ANN():
    global Max_net_layers, n_set_ann, net_atom_types, iflag_ann, net_layers, net_in, net_out, mG_dim, max_tri_index
    global net_layers, net_in, net_out, mG_dim, max_tri_index, Rc_ann, d_ann, d4_ann, Gauss_ann, range_min_ann
    global ActFunc_shift, Nodes_of_layer, r0_value, r0G_value, Gi_atom, dG_i, Gi_list, Gi_new
    global U0, U1, U2, W1_ann, W3_ann, W2_ann, B1_ann, B3_ann, B2_ann, dBOP_param_dxij, buf_ann
    global r0Rc, r0pRc, U1f1, U2f1, U1f2, U2f2, U1f3, U2f3, Gi_dev, xr_ij0, xr_ij1, xr_ij2, xr_ij3, xr_ij_dev, fsij_dev, dfuN_dev
    global Gi_3D_dev1, Gi_3D_dev2, Gi_3D_dev3, dfs_rij_3D1, dfs_rij_3D2, dfs_rij_3D3, dfs_rij_3D, Gi_3D_dev, dBOP_param_dxij_
    global ireport

    ialloc = [0] * (17 + 1)

    max_ls = 1
    for i in range(1, net_layers + 1):
        if Nodes_of_layer[i] > max_ls:
            max_ls = Nodes_of_layer[i]

    lnodes = Nodes_of_layer[net_layers]

    if sim_box.I_have_GPU == 0:
        dBOP_param_dxij = [[[[0.0] * (sim_box.natoms_alloc + 1) for k in range(sim_box.nbrs_per_atom + 1)] for j in range(lnodes + 1)] for i in range(3 + 1)]

    Gi_atom = [[[0.0] * (max_tri_index + 1) for j in range(n_set_ann + 1)] for i in range(4 + 1)]
    dG_i = [[[0.0] * (max_tri_index + 1) for j in range(n_set_ann + 1)] for i in range(4 + 1)]
    Gi_list = [[0.0] * (sim_box.natoms_alloc + 1) for i in range(mG_dim + 1)]
    Gi_new = [[0.0] * (sim_box.nbrs_per_atom + 1) for i in range(mG_dim + 1)]

    U1 = [0.0] * (max_ls + 1)
    U2 = [0.0] * (max_ls + 1)

    xr_ij0 = [[0.0] * (sim_box.natoms_alloc + 1) for i in range(sim_box.nbrs_per_atom + 1)]
    xr_ij1 = [[0.0] * (sim_box.natoms_alloc + 1) for i in range(sim_box.nbrs_per_atom + 1)]
    xr_ij2 = [[0.0] * (sim_box.natoms_alloc + 1) for i in range(sim_box.nbrs_per_atom + 1)]
    xr_ij3 = [[0.0] * (sim_box.natoms_alloc + 1) for i in range(sim_box.nbrs_per_atom + 1)]
    fsij_dev = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(n_set_ann + 1)] for i in range(sim_box.nbrs_per_atom + 1)]
    dfs_rij_3D1 = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(sim_box.nbrs_per_atom + 1)] for i in range(n_set_ann + 1)]
    dfs_rij_3D2 = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(sim_box.nbrs_per_atom + 1)] for i in range(n_set_ann + 1)]
    dfs_rij_3D3 = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(sim_box.nbrs_per_atom + 1)] for i in range(n_set_ann + 1)]
    Gi_dev = [[0.0] * (sim_box.natoms_alloc + 1) for i in range(mG_dim + 1)]
    Gi_3D_dev1 = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(sim_box.nbrs_per_atom + 1)] for i in range(mG_dim + 1)]
    Gi_3D_dev2 = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(sim_box.nbrs_per_atom + 1)] for i in range(mG_dim + 1)]
    Gi_3D_dev3 = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(sim_box.nbrs_per_atom + 1)] for i in range(mG_dim + 1)]
    dfuN_dev = [[[0.0] * (sim_box.natoms_alloc + 1) for j in range(net_layers - 2 + 1)] for i in range(max_ls + 1)]
    U1f1 = [0.0] * (max_ls + 1)
    U1f2 = [0.0] * (max_ls + 1)
    U1f3 = [0.0] * (max_ls + 1)
    U2f1 = [0.0] * (max_ls + 1)
    U2f2 = [0.0] * (max_ls + 1)
    U2f3 = [0.0] * (max_ls + 1)
    dBOP_param_dxij_ = [[[[0.0] * (sim_box.natoms_alloc + 1) for k in range(sim_box.nbrs_per_atom + 1)] for j in range(lnodes + 1)] for i in range(3 + 1)]
    r0Rc = [0.0] * (n_set_ann + 1)
    r0pRc = [0.0] * (n_set_ann + 1)

    ierror = 0
    for i in range(1, 17 + 1):
        ierror = ierror + ialloc[i]
    if ierror != 0:
        print('ERROR allocating x in alloc_atoms_ANN')
        sim_box.ihalt = 1

    return ierror
    # ! End of ! alloc_atoms_ANN !

"""
!
! ---------------------------------------------------------------------
!						! deall_atoms_ANN !
"""


def deall_atoms_ANN():

    global dBOP_param_dxi, Gi_atom, dG_i, Gi_list, U1, U2, Gi_new, xr_ij0, xr_ij1, xr_ij2, xr_ij3, fsij_dev
    global dfs_rij_3D1, dfs_rij_3D2, dfs_rij_3D3, Gi_dev, Gi_3D_dev1, Gi_3D_dev2, Gi_3D_dev3, dfuN_dev
    global U1f1, U2f1, dBOP_param_dxij_, r0Rc
    global ireport


    ialloc = [0] * (17 + 1)

    if dBOP_param_dxij:
        dBOP_param_dxij.clear()

    if Gi_atom:
        Gi_atom.clear()
    if dG_i:
        dG_i.clear()
    if Gi_list:
        Gi_list.clear()

    if U1:
        U1.clear()
    if U2:
        U2.clear()

    if Gi_new:
        Gi_new.clear()

    if xr_ij0:
        xr_ij0.clear()
    if xr_ij1:
        xr_ij1.clear()
    if xr_ij2:
        xr_ij2.clear()
    if xr_ij3:
        xr_ij3.clear()
    if fsij_dev:
        fsij_dev.clear()
    if dfs_rij_3D1:
        dfs_rij_3D1.clear()
    if dfs_rij_3D2:
        dfs_rij_3D2.clear()
    if dfs_rij_3D3:
        dfs_rij_3D3.clear()
    if Gi_dev:
        Gi_dev.clear()
    if Gi_3D_dev1:
        Gi_3D_dev1.clear()
    if Gi_3D_dev2:
        Gi_3D_dev2.clear()
    if Gi_3D_dev3:
        Gi_3D_dev3.clear()
    if dfuN_dev:
        dfuN_dev.clear()
    if U1f1:
        U1f1.clear()
    if U2f1:
        U2f1.clear()
    if dBOP_param_dxij_:
        dBOP_param_dxij_.clear()
    if r0Rc:
        r0Rc.clear()

    ierror = 0
    for i in range(1, 17 + 1):
        ierror = ierror + ialloc[i]

    return ierror
    # ! End of deall_atoms_ANN !

"""
! 
! ------------------------------------------------------------------
!
"""

# u END MODULE  ! BOP !