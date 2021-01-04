#
# ------------------------------------------------------------------
# 12-10-2020
#
# Input - Output module subroutines for aladyn.f code.
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
# Input - Output module subroutines for aladyn.f code
#
# Vesselin Yamakov
# National Institute of Aerospace
# 100 Exploration Way,
# Hampton, VA 23666
# phone: (757)-864-2850
# fax:   (757)-864-8911
# e-mail: yamakov@nianet.org
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
#import aladyn_sys
import atoms
import sim_box
import constants
import pot_module
import node_conf
import group_conf
#import aladyn_MD
import aladyn_ANN#ann
#import aladyn


#
# ------------------------------------------------------------------
#

# ! IO beginning !

Lines_INI = 0
Lines_Read = 0

pltFile= 'structure.plt'

##private :: subroutine fast_GET
##private :: subroutine fast_PUT
##private :: subroutine shell_P2P

#
# --------------------------------------------------------------------
#

def finder():

    natoms_in_cell = [0] * (sim_box.ncell_per_node+1)  # ! speedup: 1.450 !

    cellix = float(sim_box.nnx)
    celliy = float(sim_box.nny)
    celliz = float(sim_box.nnz)

    for n in range(1, sim_box.natoms + 1):  # ! 104 vs 51: speedup: 2.010 !
        sxn = atoms.sx[n]
        syn = atoms.sy[n]
        szn = atoms.sz[n]

        if sxn >= 0.5:  # ! make periodic along X !
            sxn = sxn - 1.0  # ! 2xfaster than sxn=sxn-dnint(sxn) !
        elif sxn < -0.5:
            sxn = sxn + 1.0

        if syn >= 0.5:  # ! make periodic along Y !
            syn = syn - 1.0
        elif syn < -0.5:
            syn = syn + 1.0

        if szn >= 0.5:  # ! make periodic along Z !
            szn = szn - 1.0
        elif szn < -0.5:
            szn = szn + 1.0

        atoms.sx[n] = sxn
        atoms.sy[n] = syn
        atoms.sz[n] = szn
    # ! do n=1,natoms !
    #
    # *** now update the link-cell arrays
    #
    ierror = 0
    MC_rank_XY = sim_box.MC_rank_X * sim_box.MC_rank_Y

    for i in range(1, sim_box.natoms + 1):

        ix = int((atoms.sx[i] + 0.5) * cellix)
        iy = int((atoms.sy[i] + 0.5) * celliy)
        iz = int((atoms.sz[i] + 0.5) * celliz)

        # put atom "i" in cell "(ix,iy,iz)"

        icell = operator.mod(iz + sim_box.iZ_shift, sim_box.nnz) * \
                sim_box.nXYlayer + \
                operator.mod(iy + sim_box.iY_shift, sim_box.nny) * \
                sim_box.nnx + \
                operator.mod( ix + sim_box.nnx, sim_box.nnx)

        sim_box.ncell_of_atom[i] = icell

        if natoms_in_cell[icell] < sim_box.natoms_per_cell:
            natoms_in_cell[icell] += 1
            sim_box.n_in_cell[natoms_in_cell[icell]][icell] = i
        else:
            ierror = 1

    # ! do i=1,natoms !

    sim_box.error_check(ierror, 'ERROR: FINDER: Too many atoms per cell...')
    # ! End of finder !

#
# ---------------------------------------------------------------------
#

def alloc_atoms():
    # use sys_OMP
    # use sys_ACC
    # use ANN

    print('ALLOCATE_ATOMS: ', sim_box.natoms_alloc)

    ierr_acc = 0
    ierror = 0

    atoms.alloc_atoms_sys(ierror)
    sim_box.error_check(ierror, 'Memory alloc_atoms_sys error in alloc_atoms...')

    if ierror != 0:
        aladyn_ANN.alloc_atoms_ANN(ierror)
        sim_box.error_check(ierror, 'Memory alloc_atoms_ANN error in alloc_atoms...')
        return

    atoms.alloc_atoms_MD(ierror)  # ! alloc x1,.. x5 !

    nbufsize = 16 * sim_box.natoms_alloc  # ! max isize in finder_MD !
    if ierror == 0:
        sim_box.alloc_buffers(ierror)

    sim_box.error_check(ierror, 'MEMORY alloc_buffers error in alloc_atoms...')

    # ! End of alloc_atoms !

#
# ---------------------------------------------------------------------
#  Writes structure output in plt format
# ---------------------------------------------------------------------
#

def write_structure_plt():

    fname = ""
    ffname = ""
    h_out = 0.0

    isize = 7
    itime = int(sim_box.real_time)

    dt1 = 1.0 / atoms.dt_step  # ! 1/[ps] !

    fname = 'structure.' + str(itime) + '.plt'

    n = len(fname)
    k = 0
    for i in range(1, n + 1):
        if fname[i] == ' ':
            k += 1
        else:
            fname[i - k] = fname[i]

    for i in range(n - k + 1, n + 1):
        fname[i] = ' '

    ffname = sim_box.file_path + fname
    f = open(ffname)

    xtmx = sim_box.h[1][1] * 0.5  # ! half the size in X in [Ang.] !
    xtmn = -xtmx
    ytmx = sim_box.h[2][2] * 0.5  # ! half the size in Y in [Ang.] !
    ytmn = -ytmx
    ztmx = sim_box.h[3][3] * 0.5  # ! half the size in Z in [Ang.] !
    ztmn = -ztmx

    print('#', xtmn, ytmn, ztmn)
    print('#', xtmx, ytmx, ztmx)
    print('#', xtmn, ytmn, ztmn)
    print('#', xtmx, ytmx, ztmx)
    print('#', sim_box.nbas, sim_box.natoms, sim_box.natoms_buf,
          sim_box.natoms_free)
    print('#', sim_box.r_plt, sim_box.mdx, sim_box.mdy, sim_box.mdz)
    print('#', sim_box.ibcx, sim_box.ibcy, sim_box.ibcz)
    print('#', sim_box.ipo, sim_box.ipl)
    print('#', pot_module.PotEnrg_atm, sim_box.T_sys)
    print('#', pot_module.PotEnrg_atm, sim_box.T_sys)

    for kk in range(1, sim_box.natoms + 1):
        Cx = sim_box.h[1][1] * atoms.sx[kk] + sim_box.h[1][2] * \
             atoms.sy[kk] + sim_box.h[1][3] * atoms.sz[kk]
        Cy = sim_box.h[2][2] * atoms.sy[kk] + sim_box.h[2][3] * \
             atoms.sz[kk]
        Cz = sim_box.h[3][3] * atoms.sz[kk]

        ntp = atoms.ntype[kk]  # ! write type as in pot.dat file !
        print(atoms.ident[kk], Cx, Cy, Cz, ntp, 0)

    # ! do kk = 1, natoms !

    f.close()

    print(0)  # ! 0 - no velocities; 1 - vel. !
    # ! End of write_structure_plt !

#
# --------------------------------------------------------------------
# This subroutine reads pot.dat file
# --------------------------------------------------------------------
#

def read_pot_dat():
    # use constants

    pot_module.filename = ""

    ierror = 0

    r_ADP_cut = 0.0
    iatom_types = 1  # ! Number of chemical elements !

    # ==== Reading chemical species and filenames of the potential ======

    ifile_numbs = iatom_types * (iatom_types + 1) / 2
    ipair_types = pow(iatom_types, 2)

    pot_module.alloc_pot_types(ierror)  # ! in pot_module !
    # ! allocates arrays that are common to ALL pot file types and formats !

    sim_box.error_check(ierror, 'alloc_pot_types error in read_pot_dat...')

    pot_module.ielement[0] = 0
    print("debug yann pot_module.elem_symb ", len(pot_module.elem_symb))
    pot_module.elem_symb[1] = 'Al'
    pot_module.gram_mol[1] = 26.982
    ielem = pot_module.numb_elem_Z(pot_module.elem_symb[1])
    pot_module.ielement[1] = ielem  # ! Z - number !
    pot_module.amass[1] = pot_module.gram_mol[1] * constants.atu
    # ! convert to atomic units !
    # ! atu = 100.0/(cavog * evjoul) = 0.010364188 eV.ps^2/nm^2 !

    iPOT_func_type = 1
    pot_module.filename = './ANN.dat'

    print(' CHEMICAL ELEMENTS:', '   TYPE   ELEMENT', '    Z    Atomic Mass  POT_func_type')
    for i in range(1, iatom_types + 1):
        print("debug yann iatom_types ", iatom_types)
        print(i, pot_module.elem_symb[i], pot_module.ielement[i],
              pot_module.gram_mol[i], iPOT_func_type)

    nelem_in_com = iatom_types

    iPOT_file_type = 10  # ! ANN file with a trained NN !

    print('Potential file format: ANN')
    n_pot_files = 1

    MC_rank = 3
    # ! End of read_pot_dat !

#
# ---------------------------------------------------------------------
# Establish the initial chemical composition from structure file
# Called by read_structure (only once)
# ---------------------------------------------------------------------
#

def structure_chem():
    ierror = 0

    # *** Collect atom types ...
    natoms_of_type = [0]

    for n in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[n]  # ! corresponds to ann.dat type order !
        if (ntp > atoms.iatom_types) or (ntp == 0):
            ierror = 1
            print('ERROR: atom n=', n, ' of id=', atoms.ident[n], ' is of unknown type:', ntp)
            break
        else:
            natoms_of_type[ntp] = natoms_of_type[ntp] + 1

    if ierror != 0:
        print("Le programme s'est arreter sur une erreur dans la fonction structure_chem de aladyn.io ligne 330 a 342.")
        sys.exit("")


    print('ncell_per_node=', sim_box.ncell_per_node)
    print('ncell=', sim_box.ncell)
    print('cell_volume=', sim_box.cell_volume, ' [Ang^3];')
    print('    Atoms allocated per node:', sim_box.natoms_alloc)
    print('    Atoms allocated per cell:', sim_box.natoms_per_cell)
    print('    Atoms allocated per 3x3x3 cells:', sim_box.natoms_per_cell3)
    print('Neighbors allocated per atom:', sim_box.nbrs_per_atom)

    # 15   print('    Close neighbors per atom:',i8)

    # 83   print(i2,': (',A2,'):  Atom %:',f9.4)
    # ! End of structure_chem !

#
# ---------------------------------------------------------------------
#  ***  read the input atomic structure for MD simulation          ***
#       using plt format from "sold"
# ---------------------------------------------------------------------
#

def read_structure_plt():

    global pltFile

    file_in = ""

    str82 = ""
    LINE = ""
    LINE2 = ""
    DUM2 = ""
    print("debug yann point de passage read_structure_plt")

    rsmax0 = 1.0

    f = open(pltFile, "r")

    itime = int(sim_box.start_time)
    print(' Start time:', itime, ' [MCS]')

    print('Reading ', pltFile ,' ...')

    fields = f.readline().strip().split()
    DUM2 = fields[0], xmn = fields[1], ymn = fields[2], zmn = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], xmx = fields[1], ymx = fields[2], zmx = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], xtmn = fields[1], ytmn = fields[2], ztmn = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], xtmx = fields[1], ytmx = fields[2], ztmx = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], nbas = fields[1], natoms_tot = fields[2], natoms_buf = fields[3], natoms_free = fields[4]
    # ! total, buffer, free atoms !
    fields = f.readline().strip().split()
    DUM2 = fields[0], r_plt = fields[1], mdx = fields[2], mdy = fields[3], mdz = fields[4]
    fields = f.readline().strip().split()
    DUM2 = fields[0], ibcx = fields[1], ibcy = fields[2], ibcz = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], ipo = fields[1], ipl = fields[2]
    fields = f.readline().strip().split()
    DUM2 = fields[0], PotEnrg_atm = fields[1]

    # Check if the plt file has a dof column
    fields = f.readline().strip().split()
    LINE = fields[0] + ' ' + fields[1] + ' ' + fields[2] + ' ' + fields[3] + ' ' + fields[4] + ' ' + fields[5]
    LINE2 = LINE + ' -1'  # ! add "-1" to line end !
    print('LINE2:', LINE2)
    fields = LINE2.strip().split()
    id = fields[0], xk = fields[1], yk = fields[2], zk = fields[3], ktype = fields[4], kdof = fields[5]
    print(id, xk, yk, zk, ktype, kdof)

    if kdof == -1:
        kdof_exist = 0  # ! old plt file (no constrains) !
    else:
        kdof_exist = 1  # ! new plt file with constrains !

    f.close()
    f = open('pltFile', "r")

    fields = f.readline().strip().split()
    DUM2 = fields[0], xmn = fields[1], ymn = fields[2], zmn = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], xmx = fields[1], ymx = fields[2], zmx = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], xtmn = fields[1], ytmn = fields[2], ztmn = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], xtmx = fields[1], ytmx = fields[2], ztmx = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], nbas = fields[1], natoms_tot = fields[2], natoms_buf = fields[3], natoms_free = fields[4]
    # ! total, buffer, free atoms !
    fields = f.readline().strip().split()
    DUM2 = fields[0], r_plt = fields[1], mdx = fields[2], mdy = fields[3], mdz = fields[4]
    fields = f.readline().strip().split()
    DUM2 = fields[0], ibcx = fields[1], ibcy = fields[2], ibcz = fields[3]
    fields = f.readline().strip().split()
    DUM2 = fields[0], ipo = fields[1], ipl = fields[2]
    fields = f.readline().strip().split()
    DUM2 = fields[0], PotEnrg_atm = fields[1]

    print('kdof_exist=', kdof_exist)

    # First spread the atoms uniformly to all nodes

    natoms = natoms_tot

    sim_box.h[1][1] = (xtmx - xtmn)  # ! system size from second 2 lines !
    sim_box.h[2][2] = (ytmx - ytmn)
    sim_box.h[3][3] = (ztmx - ztmn)

    sim_box.h[1][2] = 0.0  # ! ORT structure !
    sim_box.h[1][3] = 0.0
    sim_box.h[2][3] = 0.0
    sim_box.h[2][1] = 0.0
    sim_box.h[3][1] = 0.0
    sim_box.h[3][2] = 0.0

    sim_box.matinv(sim_box.h, sim_box.hi)

    print('Input file Pot. Enrg=', PotEnrg_atm)
    print(' ')

    nxold = 0  # ! first node_config call !
    nyold = 0
    nzold = 0
    nflag = aladyn.node_config()
    # ! gets ncell, ncell_per_node  !
    # ! natoms_alloc, call alloc_atoms !

    ndof_flag = 0

    for i in range(1, natoms + 1):
        idof = 0
        if kdof_exist != 0:
            fields = f.readline().strip().split()
            id = fields[0], xk = fields[1], yk = fields[2], zk = fields[3], ktype = fields[4], idof = fields[5]
            ndof_flag = operator.ior(ndof_flag, idof)  # ! collect DOF constraints !
        else:
            fields = f.readline().strip().split()
            id = fields[0], xk = fields[1], yk = fields[2], zk = fields[3], ktype = fields[4]

        atoms.ident[i] = int(id)
        atoms.rx[i] = float(xk)
        atoms.ry[i] = float(yk)
        atoms.rz[i] = float(zk)
        atoms.ntype[i] = int(ktype)

    f.close()
    # ! End of read_structure_plt !

#
# ---------------------------------------------------------------------
# Read input structure files in plt or lammps format
# Note: read_structure must be called AFTER read_pot and read_com
# ---------------------------------------------------------------------
#

def read_structure():

    print("debug yann INP_STR_TYPE",pot_module.INP_STR_TYPE )

    if pot_module.INP_STR_TYPE:
        read_structure_plt()  # ! plt type !

    print(' ')
    print('h(i,j) matrix:')
    print(sim_box.h[1][1], sim_box.h[1][2], sim_box.h[1][3])
    print(sim_box.h[2][1], sim_box.h[2][2], sim_box.h[2][3])
    print(sim_box.h[3][1], sim_box.h[3][2], sim_box.h[3][3])
    print('Crystal structure has ', sim_box.natoms, ' atoms')

    structure_chem()

    sim_box.matinv(sim_box.h, sim_box.hi)  # ! h * hi = I !
    hi11 = sim_box.hi[1][1], hi12 = sim_box.hi[1][2], hi13 = sim_box.hi[1][3]
    hi22 = sim_box.hi[2][2], hi23 = sim_box.hi[2][3], hi33 = sim_box.hi[3][3]

    for n in range(1, sim_box.natoms + 1):
        atoms.sx[n] = hi11 * atoms.rx[n] + hi12 * atoms.ry[n] + \
                                  hi13 * atoms.rz[n]
        atoms.sy[n] = hi22 * atoms.ry[n] + hi23 * atoms.rz[n]
        atoms.sz[n] = hi33 * atoms.rz[n]
    # ! End of read_structure !

#
# ---------------------------------------------------------------------
#   ***  read the input parameters for md calculation  ***************
#        Called only once in Main after read_pot to read aladyn.com
#        first initialization lines.
# ---------------------------------------------------------------------
#

def read_com():
    ##use constants

    LINE0 = ""
    LINE = ""
    LeadChar = ""
    chem = ""
    str_format_inp = ""
    str_format_out = ""
    Astring = ""

    ##10  format(A200)
    ##12  format(A2)
    ##16  format('LINE: ',A200)
    ##50  format(A)

    no_line = 0
    new_mu0 = 0
    new_alpha0 = 0
    iensemble = 0
    itriclinic = 0  # ! assume ORTHORHOMBIC system !
    at_vol_inp = -1.0  # ! no preset atomic volume !
    iHEAD_WRITE = 1  # ! write a header line in the *.dat file !
    iCrack = 0  # ! No crack !

    ierror = 0
    start_time = 0.0

    for i in range(0, atoms.iatom_types + 1):
        pot_module.iZ_elem_in_com[i] = 13

    # --- Some Initializations ---

    real_time = start_time  # ! [fms] !
    T_sys = sim_box.T_set

    r_cut_short = pot_module.r_cut_off  # ! pot. cut off !

    size = pot_module.r_cut_off

    pot_module.r2_cut_off = pow(pot_module.r_cut_off, 2)  # ! [Ang^2] !
    r2_cut_short = pow(r_cut_short, 2)
    # ! End of read_com !

#
# ---------------------------------------------------------------------
#  Command Options:
#  Version type:      aladyn -v 1  (default 0)
#  Number MD steps:   aladyn -n 10 (default 10)
#  Measure step:      aladyn -m 5  (default 1)
# ---------------------------------------------------------------------
#

def read_Args():

    N_args = 0

    str_opt = ""
    str_num = ""
    chem = ""

    # ! Default values: !
    iver = 0
    nstep = 10
    measure_step = 1
    dt_step = 0.001  # ! 1 fs = 0.001 ps !
    start_time = 0.0
    T_set = 100.0  # ! K !

    k = 0
    N_args = len(sys.argv)
    GETARG = str(sys.argv)

    while k < N_args:
        str_opt = GETARG[k]
        k += 1

        if (str_opt == '-v') or (str_opt == '-V'):
            str_num = GETARG[k]
            k += 1
            iver = str_num
        elif (str_opt == '-n') or (str_opt == '-N'):
            str_num = GETARG[k]
            k += 1
            nstep = str_num
        elif (str_opt == '-m') or (str_opt == '-M'):
            str_num = GETARG[k]
            k += 1
            measure_step = str_num
        # ! if((str_opt(1:2)... !

    # ! do while(k.le.N_args) !

    nodes_on_Y = 1
    nodes_on_Z = 1

    print(' Version:', iver)
    print(' Executing:    ', nstep, ' MD steps')
    print(' Measure at each ', measure_step, ' MD steps')
    # ! End of read_Args !

#
# ---------------------------------------------------------------------
#
#      END FILE  ! IO !
#
# =====================================================================
