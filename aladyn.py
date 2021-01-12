#
# ------------------------------------------------------------------
# 01-12-2021
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

import time

import PROGRAM_END
import aladyn_sys
import sys_ACC
import atoms
import sim_box
import constants
import pot_module
import node_conf
import MD
import aladyn_IO
import aladyn_ANN


# !
# !-------------------------------------------------------------------
# !

# !
# ! **********************************************************************
# ! *** report instantaneous properties                                ***
# ! **********************************************************************
# !

def report(jstep):

    # !
    # ! *** temperature and energy
    # !

    epot = pot_module.PotEnrg_glb / sim_box.natoms
    MD.get_T()
    etott = epot + sim_box.Ek_sys

    print(jstep, ', t=', sim_box.real_time, 'ps, Ep=', epot, '+ Ek=', sim_box.Ek_sys,
          '= Etot=', etott, 'eV/atom, Tsys=', sim_box.T_sys, 'K')

    return
    # ! End of report !

# !
# !-------------------------------------------------------------------
# !

def nodeRight_of(node):

    node_Z = node / sim_box.nodes_on_Y
    node_Y = node % sim_box.nodes_on_Y
    nodeRight_of =  node_Z * sim_box.nodes_on_Y + ((node_Y + 1) % sim_box.nodes_on_Y)

    return
    # ! End of nodeRight_of(node) !

# !
# !--------------------------------------------------------------------
# !

def nodeLeft_of(node):

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

    ll_nbr = [0] * (sim_box.natoms_per_cell3 + 1)

    # !
    # ! do loop over all cells
    # !

    sim_box.h11 = sim_box.h[1][1]
    sim_box.h12 = sim_box.h[1][2]
    sim_box.h13 = sim_box.h[1][3]
    sim_box.h22 = sim_box.h[2][2]
    sim_box.h23 = sim_box.h[2][3]
    sim_box.h33 = sim_box.h[3][3]

    for i in range(1, sim_box.natoms + 1):
        for j in range(0, sim_box.nbrs_per_atom + 1):
            atoms.nbr_list[j][i] = i  # ! Initial state: all nbrs are self - nbrs !

    atoms.max_nbrs = 0
    sz0_cut = pot_module.r_cut_off / sim_box.h33

    for ic in range(1, sim_box.ncells_all + 1):  # ! Each ic is independent !
        icell = sim_box.id_of_cell[ic]
        iz_nr = int(icell / sim_box.nXYlayer)
        iyx = icell % sim_box.nXYlayer
        iy_nr = int(iyx / sim_box.nnx)
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
                jyn = kzn + ((iy_nr + sim_box.nny + iyl) % sim_box.nny) * sim_box.nnx
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
                    rz0 = sim_box.h33 * sz0
                    sy0 = atoms.sy[l] - syn
                    if sy0 >= 0.5:          # ! make periodic along Y !
                        sy0 = sy0 - 1.0
                    elif sy0 < - 0.5:
                        sy0 = sy0 + 1.0
                    ry0 = sim_box.h22 * sy0 + sim_box.h23 * sz0
                    sx0 = atoms.sx[l] - sxn
                    if sx0 >= 0.5:  # ! make periodic along X !
                        sx0 = sx0 - 1.0
                    elif sx0 < - 0.5:
                        sx0 = sx0 + 1.0
                    rx0 = sim_box.h11 * sx0 + sim_box.h12 * sy0 + sim_box.h13 * sz0
                    r2 = pow(rx0, 2) + pow(ry0, 2) + pow(rz0, 2)

                    if (r2 < pot_module.r2_cut_off) and (l != nr):
                        k_all = k_all + 1
                        atoms.nbr_list[k_all][nr] = l
                    # ! if (r2.lt.r2_cut_off)...
                # ! if (abs(sz0).lt.r_cut_off)... !
            # ! do do k = 1, l_in_cell !

            atoms.max_nbrs = max(k_all, atoms.max_nbrs)

        # ! do n = 1, nr_in_cell

    # ! do ic = 1, ncells_all !

    # ! ensure max_nbrs is a multiple of 8 to avoid remainder loops after vectorization
    # ! max_nbrs = 56

    if (atoms.max_nbrs % 8) != 0:
        atoms.max_nbrs = (int(atoms.max_nbrs / 8) + 1) * 8

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

    if ilong != 0:
        nflag = aladyn_IO.node_config()  # ! Update ncell, ncell_per_node, natoms_alloc
        aladyn_IO.finder()  # ! Redistribute atoms to nodes !
    # ! if (ilong.ne.0)... !

    get_neighbors()

    ienergy = 0  # Created because it did not exist

    force(ienergy)

    # ! --- Sum Pot.Energy from all nodes ---

    pot_module.PotEnrg_glb = pot_module.ecoh
    pot_module.PotEnrg_atm = pot_module.PotEnrg_glb / sim_box.natoms

    # !     call PROGRAM_END(1)  ! VVVV !

    return
    # ! End of force_global !

# !
# ! -------------------------------------------------------------------
# !

def SIM_run():

    MD.init_vel(sim_box.T_set)
    force_global(1)  # ! err.check node_config finder !
    MD.initaccel()  # ! sets accelerations using forces !

    print(' ')
    print('PotEnrg_atm=', pot_module.PotEnrg_atm)
    print('Sys. Pot.En=', pot_module.PotEnrg_atm * sim_box.natoms)

    report(0)  # ! Initial structure measurement !

    sim_box.BkT = 1.0 / (constants.Boltz_Kb * sim_box.T_sys)

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
        sim_box.real_time = sim_box.real_time + 1000.0 * atoms.dt_step  # ! [fs] MD run !

        MD.predict_atoms(aladyn_IO.ndof_flag)

        force_global(0)  # ! no node_config !

        MD.correct_atoms(aladyn_IO.ndof_flag)  # ! calc.sumPxyz() !
        MD.T_broadcast()  # ! Send A_fr, sumPxyz() and calc.pTemp(ntp) !

        # ! --- MD step end ---

        if ((kstep % sim_box.measure_step) == 0) and (kstep < sim_box.nstep):
            force_global(0)
            report(kstep)

    # ! do kstep = 1, nstep ! end of MD loop !

    pot_module.get_chem()

    # ! Calc.Final Energy w stress !
    force_global(0)  # ! calc atm stress !
    report(kstep + 1 - 1)  # ! kstep = 11 in Fortran, kstep = 10 in Python !
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

    seed_string = ""

    sim_box.istep = 0
    acc_V_rate = 0.0
    pot_module.PotEnrg_glb = 0.0
    pot_module.ecoh = 0.0

    # !
    # ! *** define the inverse, hi(1..3, 1..3), of h(1..3, 1..3)
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
        PROGRAM_END.PROGRAM_END(1)

    sim_box.matinv(sim_box.h, sim_box.hi)  # ! h * hi = I

    for ntp in range(1, atoms.iatom_types + 1):
        pot_module.Am_of_type[ntp] = 0.0
        pot_module.pTemp[ntp] = sim_box.T_sys
        pot_module.E_kin[ntp] = 1.5 * constants.Boltz_Kb * sim_box.T_sys

    for n in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[n]
        pot_module.Am_of_type[ntp] = pot_module.Am_of_type[ntp] + pot_module.amass[ntp]

    for i in range(1, atoms.iatom_types + 1):
        pot_module.sum_mass[i] = pot_module.Am_of_type[i]

    pot_module.avr_mass = 0.0
    for iatom in range(1, atoms.iatom_types + 1):  # ! possible but inefficient vect. !
        pot_module.avr_mass = pot_module.avr_mass + pot_module.sum_mass[iatom]
    pot_module.avr_mass = pot_module.avr_mass / sim_box.natoms

    # ! *** set up the random number generator

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

    pot_module.init_elements()

    ierror = 0
    aladyn_IO.read_pot_dat()  # ! get iatom_types, ifile_numbs, call alloc_types !

    ierror = aladyn_ANN.input_pot_ANN(pot_module.iPOT_func_type)
    # ! alloc_ates ANN types, alloc_ates ANN_BOP types !

    sim_box.error_check(ierror, 'ERROR in read_pot...')

    aladyn_ANN.init_param_ANN()     # ! rescales or initializes pot.param. !
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

    print('sim_box.I_have_GPU =', sim_box.I_have_GPU)
    print('pot_module.ecoh =', pot_module.ecoh)

    if sim_box.I_have_GPU > 0:
        pot_module.ecoh = aladyn_ANN.Frc_ANN_ACC()  # ! Analytical derivatives !
    else:
        pot_module.ecoh = aladyn_ANN.Frc_ANN_OMP()

    sim_box.error_check(sim_box.ihalt, 'ERROR in force().')

    return
    # ! End of force !

# !
# !---------------------------------------------------------------------
# ! Pre - calculates the many - body part of the interatomic potential
# !---------------------------------------------------------------------
# !

def node_management():

    node_info = node_conf.node_conf()

    my_node_name, node_name, name_of_node, name_in_group = "", "", "", ""
    ngroup_of_node = [0] * (1 + 1)

    get_node_config(node_info)
    report_node_config(node_info)

    ierror = 0
    ierror = sim_box.alloc_nodes(1)
    sim_box.error_check(ierror, 'ERROR in alloc_nodes...')

    sim_box.nxold = 0
    sim_box.nyold = 0
    sim_box.nzold = 0

    sim_box.nodes_on_Y = 0
    sim_box.nodes_on_Z = 0

    return
    # ! End of node_management !

# !
# !---------------------------------------------------------------------
# ! Report node resources
# !---------------------------------------------------------------------
# !

def report_node_config(node_info):

    sim_box.MP_procs = node_info.MP_procs
    sim_box.MP_threads = node_info.MP_threads
    sim_box.I_have_GPU = node_info.I_have_GPU
    My_GPU_id = node_info.My_GPU_id
    sys_ACC.devicetype = node_info.devicetype
    sim_box.nACC_devices = node_info.nACC_devices

    print(' ')
    print('-----------------------------------------------')
    print('| ')
    print('| Node Resources:')

    # ! Those are replacements of ACC_ * equivalents !
    # ! redefined in pgmc_sys_ACC.f and pgmc_sys_OMP.f !

    if sim_box.I_have_GPU > 0:
        sys_ACC.set_device_num(My_GPU_id, sys_ACC.devicetype)
        sys_ACC.GPU_Init(sys_ACC.acc_device_current)
        My_GPU_mem = sys_ACC.get_gpu_mem(My_GPU_id)
        My_GPU_free_mem = sys_ACC.get_gpu_free_mem(My_GPU_id)
        print('| GPUs detected', sim_box.I_have_GPU, '\n | My_GPU_id=', My_GPU_id, ' with memory of:',
              My_GPU_mem, ' bytes, free:', My_GPU_free_mem)
    else:
        print('| No GPU detected.')

    if My_GPU_id == -1:
        print('| CPUs:', sim_box.MP_procs, ' using ', sim_box.MP_threads, ' threads and no GPU devices')
    elif My_GPU_id == 0:
        print('| CPUs:', sim_box.MP_procs, ' using ', sim_box.MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', sys_ACC.devicetype)
    elif My_GPU_id == 1:
        print('| CPUs:', sim_box.MP_procs, ' using ', sim_box.MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', sys_ACC.devicetype)
    elif My_GPU_id == 2:
        print('| CPUs:', sim_box.MP_procs, ' using ', sim_box.MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', sys_ACC.devicetype)
    else:
        print('| CPUs:', sim_box.MP_procs, ' using ', sim_box.MP_threads, ' threads\n | and the ', My_GPU_id + 1,
              '-st GPU of devicetype=', sys_ACC.devicetype)

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

    # ! Those are replacements of ACC_ * equivalents    !
    # ! redefined in pgmc_sys_ACC.f and pgmc_sys_OMP.f !

    sys_ACC.devicetype = sys_ACC.get_device_type()

    sim_box.nACC_devices = sys_ACC.get_num_devices(sys_ACC.devicetype)

    sim_box.I_have_GPU = sim_box.nACC_devices

    # ! devicetype = 1 ; I_have_GPU = 1 !
    # ! Test GPU code without GPU VVVV !

    My_GPU_id = -1
    if sim_box.nACC_devices > 0:
        My_GPU_id = 0

    # ! Those are replacements of OMP_ * equivalents !
    # ! redefined in pgmc_sys_OMP.f and pgmc_sys_ACC.f !

    sim_box.MP_procs = aladyn_sys.GET_NUM_PROCS()

    sim_box.MP_max_threads = aladyn_sys.GET_MAX_THREADS()

    # ! MP_threads = aladyn_sys.GET_NUM_THREADS() !
    sim_box.MP_threads = sim_box.MP_max_threads

    node_info.MP_procs = sim_box.MP_procs
    node_info.MP_threads = sim_box.MP_threads
    node_info.I_have_GPU = sim_box.I_have_GPU
    node_info.My_GPU_id = My_GPU_id
    node_info.devicetype = sys_ACC.devicetype
    node_info.nACC_devices = sim_box.nACC_devices

    return node_info
    # ! End of check_resources !

# !
# !---------------------------------------------------------------------
# ! Main program
# !---------------------------------------------------------------------
# !

def ParaGrandMC():

    time_start_01 = time.time()

    sim_box.mynod = 0
    sim_box.mpi_nodes = 1

    aladyn_IO.read_Args()

    node_management()  # ! Manages node architecture and devices !

    read_pot()  # ! POTENTIAL from GMU pot set of files !

    aladyn_IO.read_com()  # ! READ SIMULATION OPTIONS !

    aladyn_IO.read_structure()  # ! READ ATOMIC STRUCTURE after read_pot !
    # ! and keeps s-coord. from here on !
    # ! first call alloc_atoms !

    init_param()  # ! Calls init_vel

    SIM_run()

    time_end_01 = time.time()
    delta_time_01 = time_end_01 - time_start_01
    print('Total runtime:', delta_time_01)

    print('NORMAL termination of the program.')
    PROGRAM_END.PROGRAM_END(0)

    return
    # ! END OF MAIN PROGRAM !

ParaGrandMC()

# ---------------------------------------------------------------------
#
#      END FILE  ! aladyn !
#
# =====================================================================
