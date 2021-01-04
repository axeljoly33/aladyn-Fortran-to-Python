#
# ------------------------------------------------------------------
# 12-10-2020
#
# aladyn code:
#
# MiniApp for a simple molecular dynamics simulation
# of an atomistic structure using Al Neural Network potential.
# Converted to Python.
#
# Source Files:
#    aladyn_sys.py   - system module
#    py  - contains general purpose moduli
#    aladyn_IO.py    - I/O operations
#    aladyn_ANN.py   - Artificial Neural Network calculations
#    py    - Molecular Dynamics integrator and subroutines
#    aladyn.py       - Main program
#
# Compilation:
# Execution:
#
# For further information contact:
#
# Yann Abou Jaoude - Axel Joly
# Ecole Supérieure d'Ingénieurs Léonard-de-Vinci
# 12 Avenue Léonard de Vinci,
# Courbevoie, 92400, FRANCE
# phone: (+33) 01 41 16 70 00
# fax:
# e-mail: yann.abou_jaoude@edu.devinci.fr - axel.joly@edu.devinci.fr
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 04-26-2019
#
# aladyn code:
#
# MiniApp for a simple molecular dynamics simulation
# of an atomistic structure using Al Neural Network potential.
#
# Source Files: 
#    aladyn_sys.f   - system module
#    f  - contains general purpose moduli
#    aladyn_IO.f    - I/O operations
#    aladyn_ANN.f   - Artificial Neural Network calculations
#    f    - Molecular Dynamics integrator and subroutines
#    aladyn.f       - Main program
#
# Compilation: use makefile  (type make)
# Execution:   ./aladyn
#
# For further information contact:
#
# Vesselin Yamakov
# National Institute of Aerospace
# 100 Exploration Way,
# Hampton, VA 23666
# phone: (757)-864-2850
# fax:   (757)-864-8911
# e-mail: yamakov@nianet.org
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Notices:
# Copyright 2018 United States Government as represented by the 
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

import aladyn_sys
import sys_ACC

import atoms
import sim_box
import constants
import pot_module
import node_conf
import group_conf

import MD

import aladyn_IO
import aladyn_ANN


# !
# !-------------------------------------------------------------------
# !

def PROGRAM_END(ierr):
    #use sys_OMP
    #use sys_ACC
    #use sim_box
    #use pot_module
    #use atoms
    #use ANN

    ierror = 0

    ierror = atoms.deall_atoms_sys(ierror)
    ierror = atoms.deall_atoms_MD(ierror)

    aladyn_ANN.deall_types_ANN(ierror)
    aladyn_ANN.deall_atoms_ANN(ierror)

    ierror = sim_box.deall_buffers(ierror)
    ierror = pot_module.deall_pot_types(ierror)
    ierror = sim_box.deall_cells(ierror)

    if ierr != 0:
        sys.exit("")    # ! PROGRAM_END was called due to an error !
                        # ! STOP the execution, otherwise continue !
    # ! End of PROGRAM_END !

# !
# ! **********************************************************************
# ! *** report instantaneous properties                                ***
# ! **********************************************************************
# !

def report(jstep):
    # use pot_module
    # use MD

    # !
    # ! *** temperature and energy
    # !

    epot = pot_module.PotEnrg_glb / sim_box.natoms
    MD.get_T()
    etott = epot + sim_box.Ek_sys

    print(jstep, ' t=', sim_box.real_time, ' ps, Ep=', epot, ' + Ek=', sim_box.Ek_sys,
          ' = Etot=', etott, ' eV/atom,   Tsys=', sim_box.T_sys, ' K')

    return
    # ! End of report !

# !
# !-------------------------------------------------------------------
# !

def link_cell_setup():

    # use sim_box

    # !
    # ! subroutine to set up a cell structure in which to assign atoms
    # !

    ncell_per_node_old = sim_box.ncell_per_node

    # ! Parallel values               ! 1D, Serial values        !
    mynodZ = 0
    mynodY = 0

    ncZ_per_node = sim_box.nnz
    lZstart = 0
    lZend = sim_box.nnz - 1
    iZ_shift = sim_box.nnz
    ncY_per_node = sim_box.nny
    lYstart = 0
    lYend = sim_box.nny - 1
    iY_shift = sim_box.nny
    nXYlayer = sim_box.nnx * sim_box.nny
    ncell_per_node = sim_box.nnx * sim_box.nny * sim_box.nnz

    # !     write(6, 10) nodes_on_Y, ncell_per_node
    # ! 10 format('link_cell_setup: nodes_on_Y=', i2, ' ncell_per_node=', i5)

    cellix = float(sim_box.nnx)
    celliy = float(sim_box.nny)
    celliz = float(sim_box.nnz)

    ncell_per_node = (ncell_per_node / 8 + 1) * 8

    # ! *** Neighbor nodes index ***

    # !  kp1YZ | kp1Z | kp1ZY  !
    # ! ---------------------- !
    # !   km1Y | mynod | kp1Y  !
    # ! ---------------------- !
    # !  km1YZ | km1Z | km1ZY  !

    kp1Z = 0  # ! Serial mode !
    km1Z = 0
    kp1Y = 0
    km1Y = 0

    kp1YZ = 0
    kp1ZY = 0
    km1YZ = 0
    km1ZY = 0

    if ncell_per_node.gt.ncell_per_node_old:
        if sim_box.id_of_cell is None:
            sim_box.alloc_cells(ierror)  # ! alloc_ates cells in aladyn_mods !
            sim_box.error_check(ierror, 'ERROR in alloc_cells...')

    # !
    # ! Collect cell ids and indices for use in get_neighbors
    # !

    k = 0

    for iz in range(lZstart, lZend + 1):
        izz = iz + iZ_shift
        iz_nr = izz % sim_box.nnz

        if sim_box.nodes_on_Y == 1:  # ! 1D node topology !
            for iy in range(0, sim_box.nny - 1 + 1):
                iyy = iy + sim_box.nny
                iy_nr = iyy % sim_box.nny
                for ix in range(0, sim_box.nnx - 1 + 1):
                    k = k + 1
                    ixx = ix + sim_box.nnx
                    sim_box.id_of_cell[k] = iz_nr * nXYlayer + iy_nr * sim_box.nnx + \
                                                        (ixx % sim_box.nnx)
        else:  # ! 2D node topology !
            for iy in range(lYstart, lYend + 1):
                iyy = iy + iY_shift
                iy_nr = iyy % sim_box.nny
                for ix in range(0, sim_box.nnx - 1 + 1):
                    k = k + 1
                    ixx = ix + sim_box.nnx
                    sim_box.id_of_cell[k] = iz_nr * nXYlayer + iy_nr * sim_box.nnx + \
                                                        (ixx % sim_box.nnx)
        # ! if (nodes_on_Y.eq.1)... !

    ncells_all = k

    return
    # ! End of link_cell_setup !

# !
# !--------------------------------------------------------------------
# !

def nodeRight_of(node):
    # use sim_box

    node_Z = node / sim_box.nodes_on_Y
    node_Y = node % sim_box.nodes_on_Y
    nodeRight_of =  node_Z * sim_box.nodes_on_Y + ((node_Y + 1) % sim_box.nodes_on_Y)

    return
    # ! End of nodeRight_of(node) !

# !
# !--------------------------------------------------------------------
# !

def nodeLeft_of(node):
    # use sim_box

    node_Z = node / sim_box.nodes_on_Y
    node_Y = node % sim_box.nodes_on_Y
    nodeLeft_of = node_Z * sim_box.nodes_on_Y + \
                  ((node_Y - 1 + sim_box.nodes_on_Y) % sim_box.nodes_on_Y)

    return
    # ! End of nodeLeft_of(node) !

# !
# !--------------------------------------------------------------------
# !

def nodeDown_of(node):
    # use sim_box

    node_Z = node / sim_box.nodes_on_Y
    node_Y = node % sim_box.nodes_on_Y
    nodeDown_of = ((node_Z - 1 + sim_box.nodes_on_Z) % sim_box.nodes_on_Z) * \
                  sim_box.nodes_on_Y + node_Y

    return
    # ! End of nodeDown_of(node) !

# !
# !--------------------------------------------------------------------
# !

def nodeUp_of(node):
    # use sim_box

    node_Z = node / sim_box.nodes_on_Y
    node_Y = node % sim_box.nodes_on_Y
    nodeUp_of = ((node_Z + 1) % sim_box.nodes_on_Z) * sim_box.nodes_on_Y + node_Y

    return
    # ! End of nodeUp_of(node) !

# !
# !---------------------------------------------------------------------
# ! Performs loop over atoms in ALL of the neighboring cells
# ! Uses OMP, but prepares the system for ACC(GPU) implementation.
# !---------------------------------------------------------------------
# !

def get_neighbors():

    # use sim_box
    # use atoms
    # use pot_module

    ll_nbr = [0] * sim_box.natoms_per_cell3

    # !
    # ! do loop over all cells
    # !

    h11 = sim_box.h[1][1]
    h12 = sim_box.h[1][2]
    h13 = sim_box.h[1][3]
    h22 = sim_box.h[2][2]
    h23 = sim_box.h[2][3]
    h33 = sim_box.h[3][3]

    for i in range(1, sim_box.natoms + 1):
        for j in range(0, sim_box.nbrs_per_atom + 1):
            atoms.nbr_list[j][i] = i  # ! Initial state: all nbrs are self - nbrs !

    max_nbrs = 0
    sz0_cut = pot_module.r_cut_off / h33

    for ic in range(1, sim_box.ncells_all + 1):  # ! Each ic is independent !
        icell = sim_box.id_of_cell[ic]
        iz_nr = icell / sim_box.nXYlayer
        iyx = icell % sim_box.nXYlayer
        iy_nr = iyx / sim_box.nnx
        ixx = (iyx % sim_box.nnx) + sim_box.nnx

        nr_in_cell = sim_box.natoms_in_cell[icell]  # ! number atoms in icell !
        for n in range(1, nr_in_cell + 1):  # ! VECTORIZED: speedup: 4.760 !
            ll_nbr[n] = sim_box.n_in_cell[n][icell]

        l_in_cell = nr_in_cell

        # !
        # ! Loop over atoms in all 27 neighboring cells
        # !
        # ! 0 <= izl <= ncZ_per_node - 1: range = ncZ_per_node
        # ! 0 <= iyl <= ncY_per_node - 1: range = ncY_per_node
        # !

        for izl in range(-1, 1 + 1):
            kzn = ((iz_nr + sim_box.nnz + izl) % sim_box.nnz) * sim_box.nXYlayer
            for iyl in range(-1, 1 + 1):
                jyn = kzn + ((iy_nr + sim_box.nny + iyl) % sim_box.nny) * \
                      sim_box.nnx
                for i in range(-1, 1 + 1):
                    jcell = jyn + ((i + ixx) % sim_box.nnx)
                    if jcell != icell:
                        ns = sim_box.natoms_in_cell[jcell]  # ! number atoms in jcell !
                        for n in range(1, ns + 1):  # ! VECTORIZED: speedup: 4.760 !
                            ll_nbr[l_in_cell + n] = sim_box.n_in_cell[n][jcell]  # ! atom n in jcell !
                        l_in_cell = l_in_cell + ns
                    # ! if (icell.ne.jcell)... !
                # ! do i = -1, 1 !
            # ! do iyl = !
        # ! do izl = !

        # !
        # ! Start: Find neighbors of atoms in icell.
        # !

        for n in range(1, nr_in_cell + 1):

            nr = ll_nbr[n]

            sxn = atoms.sx[nr]
            syn = atoms.sy[nr]
            szn = atoms.sz[nr]

            k_all = 0  # ! no neighbors !

            # !!!!!DIR$ NOVECTOR
            for k in range(1, l_in_cell + 1):
                l = ll_nbr[k]
                sz0 = atoms.sz[l] - szn
                if sz0 >= 0.5:          # ! make periodic along Z !
                    sz0 = sz0 - 1.0     # ! 2x faster than sz0 = sz0 - dnint(sz0) !
                elif sz0 < - 0.5:
                    sz0 = sz0 + 1.0
                if abs(sz0) < sz0_cut:
                    rz0 = h33 * sz0
                    sy0 = atoms.sy[l] - syn
                    if sy0 >= 0.5:          # ! make periodic along Y !
                        sy0 = sy0 - 1.0
                    elif sy0 < - 0.5:
                        sy0 = sy0 + 1.0
                    ry0 = h22 * sy0 + h23 * sz0
                    sx0 = atoms.sx[l] - sxn
                    if sx0 >= 0.5:  # ! make periodic along X !
                        sx0 = sx0 - 1.0
                    elif sx0 < - 0.5:
                        sx0 = sx0 + 1.0
                    rx0 = h11 * sx0 + h12 * sy0 + h13 * sz0
                    r2 = rx0 ** 2 + ry0 ** 2 + rz0 ** 2

                    if (r2 < pot_module.r2_cut_off) and (l != nr):
                        k_all = k_all + 1
                        atoms.nbr_list[k_all][nr] = l
                    # ! if (r2.lt.r2_cut_off)...
                # ! if (abs(sz0).lt.r_cut_off)... !
            # ! do do k = 1, l_in_cell !

            max_nbrs = max(k_all, max_nbrs)

        # ! do n = 1, nr_in_cell

    # ! do ic = 1, ncells_all !

    # ! ensure max_nbrs is a multiple of 8 to avoid remainder loops after vectorization
    # ! max_nbrs = 56

    if (max_nbrs % 8) != 0:
        max_nbrs = ((max_nbrs / 8) + 1) * 8

    return
    # ! End of get_neighbors !

# !
# ! -------------------------------------------------------------------
# !

def force_global(ilong):
    # !
    # ! ** *subroutine for doing force calculation with linked cells
    # !        ilong = (1) do full force calculation
    # !                (0) skip node_config and finder when
    # !                    sx() haven't changed, like in box rescaling

    # use sim_box
    # use pot_module
    # use IO

    if ilong != 0:
        node_config()  # ! Update ncell, ncell_per_node, natoms_alloc
        aladyn_IO.finder()  # ! Redistribute atoms to nodes !
        # ! if (ilong.ne.0)... !

    get_neighbors()

    ienergy = 0  # Created because it did not exist

    force(ienergy)

    # ! --- Sum Pot.Energy from all nodes ---

    PotEnrg_glb = pot_module.ecoh
    PotEnrg_atm = PotEnrg_glb / sim_box.natoms

    # !     call PROGRAM_END(1)  ! VVVV !

    return
    # ! End of force_global !

# !
# ! -------------------------------------------------------------------
# !   Looks for optimal node architecture configuration
# ! -------------------------------------------------------------------
# !

def node_config():
    # !
    # !   ***  updates hij matrix according to the farthest atoms in the
    # !        system in Y and Z directions, and
    # !   ***  looks for optimal node architecture configuration
    # !

    #use sim_box
    #use pot_module
    #use IO
    #use atoms

    natoms_alloc_new = 0 #  ! local !
    nnx_min, nny_min, nnz_min = 0, 0, 0

    nflag = 0
    ierror = 0
    nodes_on_Yo, nodes_on_Zo = sim_box.nodes_on_Y, sim_box.nodes_on_Z
    cell_size_X, cell_size_Y, cell_size_Z = 0.0, 0.0, 0.0

    i1, i2, i3 = 1, 2, 3

    nodes_on_Y, nodes_on_Z = 1, 1
    res = get_config(i1, i2, i3, 1, 1, 1)
    nnx_min, nny_min, nnz_min = res[0], res[1], res[2]
    nnx_cell, nny_cell, nnz_cell = res[3], res[4], res[5]
    ierror = res[6]
    nnx_try, nny_try, nnz_try = 1, 1, 1

    if ierror > 0:

        print(' ')
        print('ERROR: Unable to construct a suitable link-cell grid!')
        print(' ')
        print('System Box size:', sim_box.h[i1][i1], sim_box.h[i2][i2],
              sim_box.h[i3][i3], ';   min. cell size=', sim_box.size)  # 2(f12.6,' x '),f12.6
        if (ierror & 1) > 0:
            print(' Cells per node on X =', sim_box.nnx, '  must be  >   2: NO')
        else:
            print(' Cells per node on X =', sim_box.nnx, '  must be  >   2: YES')

        if (ierror & 2) > 0:
            print(' Cells per node on Y =', sim_box.nny, '  must be  > ', nny_min - 1, ': NO')
        else:
            print(' Cells per node on Y =', sim_box.nny, '  must be  > ', nny_min - 1, ': YES')

        if (ierror & 4) > 0:
            print(' Cells per node on Z =', sim_box.nnz, '  must be  > ', nnz_min - 1, ': NO')
        else:
            print(' Cells per node on Z =', sim_box.nnz, '  must be  > ', nnz_min - 1, ': YES')
        print('Decrease number of nodes or increase system size...')

        PROGRAM_END(1)

    else:  # ! if(ierror.gt.0)... !
        cell_size_X = sim_box.h[i1][i1] / sim_box.nnx
        cell_size_Y = sim_box.h[i2][i2] / sim_box.nny
        cell_size_Z = sim_box.h[i3][i3] / sim_box.nnz
    # endif ! if(ierror.gt.0)... !

    ncell = sim_box.nnx * sim_box.nny * sim_box.nnz

    print("Debug yann cette valeur n'est pas transmise a Reading structure:", atoms.sys_vol)

    sim_box.matinv(sim_box.h, sim_box.hi)
    atom_vol1 = atoms.sys_vol / sim_box.natoms

    # ! Get the maximum possible number of atoms per link cell !

    rZ_min = 10.0  # ! Start max value in Ang. !

    print('nelem_in_com=', pot_module.nelem_in_com)

    # ! read el. types as listed in aladyn.com !
    for i in range(1, pot_module.nelem_in_com + 1):
        iZ = pot_module.iZ_elem_in_com[i]
        rad_of_Z = pot_module.elem_radius[iZ]  # ! [Ang] !
        if rad_of_Z < rZ_min:
            rZ_min = rad_of_Z

    atom_vol2 = 4.0 / 3.0 * 3.141592 * math.pow(rZ_min, 3)

    # !     write(1000+mynod,*)'sys_vol=',sys_vol,' r_max=',r_max, ' rZ_min=',rZ_min
    # !     write(1000+mynod,*)'atom_vol1=',atom_vol1,' atom_vol2',atom_vol2

    if atom_vol1 < atom_vol2:
        atom_vol = atom_vol1
    else:
        atom_vol = atom_vol2

    # !     write(6,*)'atom_vol=',atom_vol,' cell_volume=', cell_volume,' rZ_min=',rZ_min

    cell_volume = (cell_size_X + 2.0* rZ_min) * (cell_size_Y + 2.0 * rZ_min) * (cell_size_Z + 2.0 * rZ_min)
    cell3_volume = (3.0 * cell_size_X + 2.0 * rZ_min) * (3.0 * cell_size_Y + 2.0 * rZ_min) * \
                   (3.0 * cell_size_Z + 2.0 * rZ_min)
    natoms_per_cell = int(cell_volume / atom_vol) + 1
    natoms_per_cell = (natoms_per_cell / 8 + 1) * 8
    natoms_per_cell3 = int(cell3_volume / atom_vol) + 1
    natoms_per_cell3 = (natoms_per_cell3 / 8 + 1) * 8

    nflag = abs(sim_box.nnx - sim_box.nxold) + \
            abs(sim_box.nny - sim_box.nyold) + \
            abs(sim_box.nnz - sim_box.nzold) + \
            abs(nodes_on_Y - nodes_on_Yo) + abs(nodes_on_Z - nodes_on_Zo)

    # !  reset cell grid if necessary
    if nflag > 0:
        link_cell_setup()

        print('\n', 'Link cell configuration:', '\n', ' axis nodes cells/n thickness; total cell:', ncell)

        print('On X: ', 1, ' x ', sim_box.nnx, ' x ', cell_size_X)
        print('On Y: ', nodes_on_Y, ' x ', nny_cell, ' x ', cell_size_Y)
        print('On Z: ', nodes_on_Z, ' x ', nnz_cell, ' x ', cell_size_Z)

        print(' ')
    # ! if (nflag.gt.0)... !

    natoms_alloc_new = sim_box.natoms + 100

    if natoms_alloc_new > sim_box.natoms_alloc:  # ! update natoms_alloc !
        natoms_alloc = (natoms_alloc_new / 64 + 1) * 64
    cut_off_vol = 4.0 / 3.0 * 3.141592 * math.pow(pot_module.r_cut_off, 3)
    nbrs_per_atom = int(round(cut_off_vol / atom_vol))  # ! Correct one !
    print('nbrs_per_atom=', nbrs_per_atom)
    nbrs_alloc = nbrs_per_atom * sim_box.natoms_alloc

    nxold = sim_box.nnx
    nyold = sim_box.nny
    nzold = sim_box.nnz

    if not atoms.ident:
        aladyn_IO.alloc_atoms()  # ! alloc_ates natoms_alloc atoms in aladyn_IO!

    return nflag
    # ! End of node_config !

# !
# ! -------------------------------------------------------------------
# !  Looks for optimal node architecture configuration at a given
# !  number of nodes on X:nodes_X, on Y:nodes_Y, and on Z:nodes_Z
# ! -------------------------------------------------------------------
# !

def get_config(i1, i2, i3, nodes_X, nodes_Y, nodes_Z):

    #use sim_box

    # !      write(50,10) i1,i2,i3, nodes_X, nodes_Y, nodes_Z
    # !  10  format('get_config(',6i3,')')
    nnx_min, nny_min, nnz_min = 0, 0, 0

    ierror = 0
    res_nnd_fit = nnd_fit(nodes_X, i1)
    nnx, nnx_min, MC_rank_X = res_nnd_fit[0], res_nnd_fit[1], res_nnd_fit[2]
    if nnx == 0:
        ierror = operator.ior(ierror, 1)
    res_nnd_fit = nnd_fit(nodes_Y, i2)
    nny, nny_mi, MC_rank_Y = res_nnd_fit[0], res_nnd_fit[1], res_nnd_fit[2]
    if nny == 0:
        ierror = operator.ior(ierror, 2)
    res_nnd_fit = nnd_fit(nodes_Z, i3)
    nnz, nnz_min, MC_rank_Z = res_nnd_fit[0], res_nnd_fit[1], res_nnd_fit[2]
    if nnz == 0:
        ierror = operator.ior(ierror, 4)

    # !      write(50,*)'get_conf: nnx,y,z=',nnx,nny,nnz,ierror
    # !      write(50,*)'get_conf: MC_rank_X,Y,Z=',
    # !    1 MC_rank_X,MC_rank_Y,MC_rank_Z

    nnx_cell = nnx / nodes_X
    nny_cell = nny / nodes_Y
    nnz_cell = nnz / nodes_Z

    # ! Check if cells per node commensurate with MC_ranks !

    nn_mod = nnx_cell % MC_rank_X
    if nn_mod != 0:
        ierror = operator.ior(ierror, 1)
    nn_mod = nny_cell % MC_rank_Y
    if nn_mod != 0:
        ierror = operator.ior(ierror, 2)
    nn_mod = nnz_cell % MC_rank_Z
    if nn_mod != 0:
        ierror = operator.ior(ierror, 4)

    # !      write(50,*) 'get_conf: nnx, y, z_cell=', nnx_cell, nny_cell, nnz_cell, ierror
    # !      write(50,*) ' '

    res = [nnx_min, nny_min, nnz_min, nnx_cell, nny_cell, nnz_cell, ierror]

    return res
    # ! End of get_config !

# !
# ! -------------------------------------------------------------------
# ! Finds the best number of cells in a given direction (nnx,nny,nnz),
# ! which commensurate with MC_rank and nodes_on_X,Y,Z
# ! -------------------------------------------------------------------
# !

def nnd_fit(nodes_on_D, iD):

    # use sim_box

    nnd_min = 3
    nnd_max = int(sim_box.h[iD][iD] / sim_box.size)

    nnd_tmp = 0
    MC_rank_D = sim_box.MC_rank

    for nnd in range(nnd_max, nnd_min - 1, -1):

        nnd_nodes = nnd % nodes_on_D  # ! check nnd vs nodes_on_D !

        if nnd_nodes == 0:  # ! nnd fits on nodes_on_D !
            nnd_rank = nnd % MC_rank_D  # ! check nnd with MC_rank_D !
            if nnd_rank == 0:
                nnd_tmp = nnd  # ! good nnd found !
            while nnd_rank > 0:  # ! if doesn't do...         !
                if(MC_rank_D < nnd_min) and (MC_rank_D < sim_box.MC_rank_max):
                    MC_rank_D = MC_rank_D + 1
                    nnd_rank = nnd % MC_rank_D  # ! check with new MC_rank_D !
                    if nnd_rank == 0:
                        nnd_tmp = nnd  # ! good nnd found !
                else:
                    nnd_rank = 0  # ! stop do while() !
            # ! do while(irepeat.gt.0) !

            if nnd_tmp > 0:
                break  # ! exit do nnd loop !
        # ! if(nnd_nodes.eq.0)... !

    # ! do nnd = nnd_max, nnd_min, -1 !

    nnd_fit = nnd_tmp

    # !     write(50,*)'d:',iD,' nnd_fit=',nnd_fit,' nnd_min=',nnd_min
    # !     write(50,*)'      MC_rank_D=',MC_rank_D

    res = [nnd_fit, nnd_min, MC_rank_D]

    return res
    # ! End of nnd_fit(nnn) !

# !
# ! -------------------------------------------------------------------
# !

def SIM_run():

    # use constants
    # use sim_box
    # use pot_module
    # use atoms
    # use MD
    # use IO
    # use ANN

    MD.init_vel(sim_box.T_set)
    force_global(1)  # ! err.check node_config finder !
    MD.initaccel()  # ! sets accelerations using forces !

    print(' ')
    print('PotEnrg_atm=', pot_module.PotEnrg_atm)
    print('Sys. Pot.En=', pot_module.PotEnrg_atm * sim_box.natoms)

    report(0)  # ! Initial structure measurement !

    BkT = 1.0 / (constants.Boltz_Kb * sim_box.T_sys)

    # !
    # !  ******************************************************************
    # !  *** begin do loop for MD steps                                 ***
    # !  ******************************************************************
    # !

    istep = 0

    for kstep in range(1, sim_box.nstep + 1):  # ! MD loop !
        istep = istep + 1

        E1 = pot_module.PotEnrg_glb / sim_box.natoms

        # ! --- MD step start ---
        real_time = sim_box.real_time + 1000.0 * atoms.dt_step  # ! [fs] MD run !

        MD.predict_atoms(ndof_flag)

        force_global(0)  # ! no node_config !

        MD.correct_atoms(ndof_flag)  # ! calc.sumPxyz() !
        MD.T_broadcast()  # ! Send A_fr, sumPxyz() and calc.pTemp(ntp) !

        # ! --- MD step end ---

        if ((kstep % sim_box.measure_step) == 0) and (kstep < sim_box.nstep):
            force_global(0)
            report(kstep)

    # ! do kstep = 1, nstep ! end of MD loop !

    pot_module.get_chem()

    # ! Calc.Final Energy w stress !
    force_global(0)  # ! calc atm stress !
    report(kstep - 1)  # ???? if kstep then go in loop, if istep then change the arg
    aladyn_IO.write_structure_plt()

    # !
    # !  ******************************************************************
    # !  *** end of do loop for time steps                              ***
    # !  ******************************************************************
    # !

    return
    # ! End of SIM_run !

# !
# !---------------------------------------------------------------------
# !

def init_param():

    # use constants
    # use sim_box
    # use atoms
    # use pot_module
    # use MD
    # use IO

    seed_string = ""

    istep = 0
    acc_V_rate = 0.0
    PotEnrg_glb = 0.0
    ecoh = 0.0

    # !
    # ! ** *define the inverse, hi(1..3, 1..3), of h(1..3, 1..3)
    # !
    # ! now add the periodic images of the input geometry
    # !

    ibox_error = 0
    if abs(2.0 * sim_box.h[1][2]) > sim_box.h[1][1]:
        ibox_error = 1
    if abs(2.0 * sim_box.h[1][3]) > sim_box.h[1][1]:
        ibox_error = 2
    if abs(2.0 * sim_box.h[1][3]) > sim_box.h[2][2]:
        ibox_error = 3
    if abs(2.0 * sim_box.h[2][3]) > sim_box.h[1][1]:
        ibox_error = 4
    if abs(2.0 * sim_box.h[2][3]) > sim_box.h[2][2]:
        ibox_error = 5
    if ibox_error != 0:
        print('ERROR! Unacceptable h-matrix!')
        if ibox_error == 1:
            print('h(1,2) or xy must be less than (xhi-xlo)/2')
        elif ibox_error == 2:
            print('h(1,3) or xz must be less than (xhi-xlo)/2')
        elif ibox_error == 3:
            print('h(1,3) or xz must be less than (yhi-ylo)/2')
        elif ibox_error == 4:
            print('h(2,3) or xz must be less than (xhi-xlo)/2')
        elif ibox_error == 5:
            print('h(2,3) or xz must be less than (yhi-ylo)/2')
        PROGRAM_END(1)

    sim_box.matinv(sim_box.h, sim_box.hi)  # ! h * hi = I

    for ntp in range(1, atoms.iatom_types + 1):
        pot_module.Am_of_type[ntp] = 0.0
        pot_module.pTemp[ntp] = sim_box.T_sys
        pot_module.E_kin[ntp] = 1.5 * constants.Boltz_Kb * sim_box.T_sys

    for n in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[n]
        pot_module.Am_of_type[ntp] = pot_module.Am_of_type[ntp] + \
                                                 pot_module.amass[ntp]

    sum_mass = pot_module.Am_of_type

    avr_mass = 0.0
    for iatom in range(1, atoms.iatom_types + 1):  # ! possible but inefficient vect. !
        avr_mass = avr_mass + sum_mass(iatom)
    avr_mass = avr_mass / sim_box.natoms

    # ! ** *set up the random number generator

    iseed = 6751
    # ! Convert the string 'str_filename' to a numeric value !
    n = len(sim_box.str_filename)
    for i in range(1, n + 1):
        iseed = iseed + ord(sim_box.str_filename[i])
    # ! randomize with str_filename !

    iseed0 = iseed + 17
    sim_box.rmarin(iseed0)  # ! node depndnt init.random generator

    MD.init_MD()

    return
    # ! End of init_param !

# !
# !---------------------------------------------------------------------
# ! At start up:
# !
# ! call read_pot
# !      call read_pot_dat ! sets iatom_types, ipair_types,
# !                   ! iPOT_file_type, iPOT_func_type
# !
# !      ! for ANN potential types !
# !           call input_pot_ANN ! read ANN pot.file !
# !           call alloc_types_ANN
# !
# !      call calc_pot_param
# !           call init_param_ANN
# !
# !---------------------------------------------------------------------
# !

def read_pot():

    # use pot_module
    # use IO
    # use ANN

    pot_module.init_elements()

    ierror = 0
    aladyn_IO.read_pot_dat()  # ! get iatom_types, ifile_numbs, call alloc_types !

    # ! alloc_ates ANN types, alloc_ates ANN_BOP types !
    aladyn_ANN.input_pot_ANN(pot_module.iPOT_func_type, ierror)

    sim_box.error_check(ierror, 'ERROR in read_pot...')

    aladyn_ANN.init_param_ANN()  # ! rescales or initializes pot.param. !
    # ! for ANN_BOP, alloc_ates BOP types too !
    sim_box.error_check(sim_box.ihalt, 'ERROR in read_pot.')

    return
    # ! End of read_pot !

# !
# !---------------------------------------------------------------------
# !  ! DO NOT CALL ALONE, only through force_global !
# !---------------------------------------------------------------------
# !

def force(ienergy):

    # use sys_OMP
    # use sys_ACC
    # use sim_box
    # use pot_module
    # use IO
    # use ANN

    if sim_box.I_have_GPU > 0:
        aladyn_ANN.Frc_ANN_ACC(pot_module.ecoh)  # ! Analytical derivatives !
    else:
        aladyn_ANN.Frc_ANN_OMP(pot_module.ecoh)

    sim_box.error_check(sim_box.ihalt, 'ERROR in force().')

    return
    # ! End of force !

# !
# !---------------------------------------------------------------------
# ! Pre - calculates the many - body part of the interatomic potential
# !---------------------------------------------------------------------
# !

def node_management():

    # use sys_OMP
    # use sys_ACC
    # use sim_box
    # use IO

    node_info = node_conf.node_conf()

    my_node_name, node_name, name_of_node, name_in_group = "", "", "", ""
    ngroup_of_node = [0]

    get_node_config(node_info)
    report_node_config(node_info)
    ierror=0
    sim_box.alloc_nodes(1, ierror)
    sim_box.error_check(ierror, 'ERROR in alloc_nodes...')

    nxold = 0
    nyold = 0
    nzold = 0

    nodes_on_Y = 0
    nodes_on_Z = 0

    return
    # ! End of node_management !

# !
# !---------------------------------------------------------------------
# ! Report node resources
# !---------------------------------------------------------------------
# !

def report_node_config(node_info):

    # use sys_OMP
    # use sys_ACC  # ! defines devicetype !
    # use sim_box

    MP_procs = node_info.MP_procs
    MP_threads = node_info.MP_threads
    I_have_GPU = node_info.I_have_GPU
    My_GPU_id = node_info.My_GPU_id
    devicetype = node_info.devicetype
    nACC_devices = node_info.nACC_devices

    print(' ')
    print('-----------------------------------------------')
    print('| ')
    print('| Node Resources:')

    # ! Those are replacements of ACC_ * equivalents !
    # ! redefined in pgmc_sys_ACC.f and pgmc_sys_OMP.f !

    if I_have_GPU > 0:
        aladyn_sys.sys_ACC.set_device_num(My_GPU_id, devicetype)
        aladyn_sys.sys_ACC.GPU_Init(aladyn_sys.sys_ACC.acc_device_current)
        My_GPU_mem = aladyn_sys.sys_ACC.get_gpu_mem(My_GPU_id)
        My_GPU_free_mem = aladyn_sys.sys_ACC.get_gpu_free_mem(My_GPU_id)
        print(' | GPUs detected', I_have_GPU, '\n | My_GPU_id=', My_GPU_id, ' with memory of:', My_GPU_mem,
              ' bytes, free:', My_GPU_free_mem)
    else:
        print(' | No GPU detected.')

    if My_GPU_id == -1:
        print(' | CPUs:', MP_procs, ' using ', MP_threads, ' threads and no GPU devices')
    elif My_GPU_id == 0:
        print(' | CPUs:', MP_procs, ' using ', MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', devicetype)
    elif My_GPU_id == 1:
        print(' | CPUs:', MP_procs, ' using ', MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', devicetype)
    elif My_GPU_id == 2:
        print(' | CPUs:', MP_procs, ' using ', MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', devicetype)
    else:
        print(' | CPUs:', MP_procs, ' using ', MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', devicetype)

    print('|')
    print('-----------------------------------------------')
    print(' ')

    return node_info
    # ! End of report_node_config !

# !
# !---------------------------------------------------------------------
# ! Check for host node resources
# !---------------------------------------------------------------------
# !

def get_node_config(node_info):

    # use sys_OMP
    # use sys_ACC  # ! defines devicetype !
    # use sim_box

    check_resources(node_info)
    node_info.node_name = 'my_node'

    return node_info
    # ! End of get_node_config !

# !
# !---------------------------------------------------------------------
# ! Check node resources
# !---------------------------------------------------------------------
# !

def check_resources(node_info):

    # use sys_OMP
    # use sys_ACC
    # use sim_box

    # ! Those are replacements of ACC_ * equivalents    !
    # ! redefined in pgmc_sys_ACC.f and pgmc_sys_OMP.f !

    ###devicetype = aladyn_sys.sys_ACC.get_device_type()
    devicetype = 0


    ###nACC_devices = aladyn_sys.sys_ACC.get_num_devices(devicetype)

    nACC_devices = 0


    I_have_GPU = nACC_devices

    # ! devicetype = 1; I_have_GPU = 1  ! Test GPU code without GPU VVVV !

    My_GPU_id = -1
    if nACC_devices > 0:
        My_GPU_id = 0

    # ! Those are replacements of OMP_ * equivalents !
    # ! redefined in pgmc_sys_OMP.f and pgmc_sys_ACC.f !

    ###MP_procs = aladyn_sys.GET_NUM_PROCS()

    MP_procs = 1
    ### MP_max_threads = aladyn_sys.GET_MAX_THREADS()
    MP_max_threads = 1
    # ! MP_threads = aladyn_sys.GET_NUM_THREADS()
    MP_threads = MP_max_threads

    node_info.MP_procs = MP_procs
    node_info.MP_threads = MP_threads
    node_info.I_have_GPU = I_have_GPU
    node_info.My_GPU_id = My_GPU_id
    node_info.devicetype = devicetype
    node_info.nACC_devices = nACC_devices

    return node_info
    # ! End of check_resources !

# !
# !---------------------------------------------------------------------
# ! Main program
# !---------------------------------------------------------------------
# !

def ParaGrandMC():

    # use sys_OMP
    # use sys_ACC
    # use constants
    # use sim_box
    # use IO

    mynod = 0
    mpi_nodes = 1

    aladyn_IO.read_Args()

    node_management()  # ! Manages node architecture and devices !

    read_pot()  # ! POTENTIAL from GMU pot set of files !

    aladyn_IO.read_com()  # ! READ SIMULATION OPTIONS !

    aladyn_IO.read_structure()  # ! READ ATOMIC STRUCTURE after read_pot !
    # ! and keeps s-coord. from here on !
    # ! first call alloc_atoms !
    init_param()  # ! Calls init_vel

    SIM_run()

    print('NORMAL termination of the program.')
    PROGRAM_END(0)

    return
    # ! END OF MAIN PROGRAM !

ParaGrandMC()