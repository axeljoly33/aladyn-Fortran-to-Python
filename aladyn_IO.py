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
import aladyn_sys
import aladyn_mods
import aladyn_MD
import aladyn_ANN
import aladyn


#
# ------------------------------------------------------------------
#

# ! IO beginning !

Lines_INI = 0
Lines_Read = 0

##private :: subroutine fast_GET
##private :: subroutine fast_PUT
##private :: subroutine shell_P2P

#
# --------------------------------------------------------------------
#

def finder():

    natoms_in_cell = [0] * aladyn_mods.sim_box.ncell_per_node  # ! speedup: 1.450 !

    cellix = float(aladyn_mods.sim_box.nnx)
    celliy = float(aladyn_mods.sim_box.nny)
    celliz = float(aladyn_mods.sim_box.nnz)

    for n in range(1, aladyn_mods.sim_box.natoms + 1):  # ! 104 vs 51: speedup: 2.010 !
        sxn = aladyn_mods.atoms.sx[n]
        syn = aladyn_mods.atoms.sy[n]
        szn = aladyn_mods.atoms.sz[n]

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

        aladyn_mods.atoms.sx[n] = sxn
        aladyn_mods.atoms.sy[n] = syn
        aladyn_mods.atoms.sz[n] = szn
    # ! do n=1,natoms !
    #
    # *** now update the link-cell arrays
    #
    ierror = 0
    MC_rank_XY = aladyn_mods.sim_box.MC_rank_X * aladyn_mods.sim_box.MC_rank_Y

    for i in range(1, aladyn_mods.sim_box.natoms + 1):

        ix = int((aladyn_mods.atoms.sx[i] + 0.5) * cellix)
        iy = int((aladyn_mods.atoms.sy[i] + 0.5) * celliy)
        iz = int((aladyn_mods.atoms.sz[i] + 0.5) * celliz)

        # put atom "i" in cell "(ix,iy,iz)"

        icell = operator.mod(iz + aladyn_mods.sim_box.iZ_shift, aladyn_mods.sim_box.nnz) * \
                aladyn_mods.sim_box.nXYlayer + \
                operator.mod(iy + aladyn_mods.sim_box.iY_shift, aladyn_mods.sim_box.nny) * \
                aladyn_mods.sim_box.nnx + \
                operator.mod( ix + aladyn_mods.sim_box.nnx, aladyn_mods.sim_box.nnx)

        aladyn_mods.sim_box.ncell_of_atom[i] = icell

        if natoms_in_cell[icell] < aladyn_mods.sim_box.natoms_per_cell:
            natoms_in_cell[icell] += 1
            aladyn_mods.sim_box.n_in_cell[natoms_in_cell[icell]][icell] = i
        else:
            ierror = 1

    # ! do i=1,natoms !

    aladyn_mods.sim_box.error_check(ierror, 'ERROR: FINDER: Too many atoms per cell...')
    # ! End of finder !

#
# ---------------------------------------------------------------------
#

def alloc_atoms():
    # use sys_OMP
    # use sys_ACC
    # use ANN

    print('ALLOCATE_ATOMS: ', aladyn_mods.sim_box.natoms_alloc)

    ierr_acc = 0
    ierror = 0

    aladyn_mods.atoms.alloc_atoms_sys(ierror)
    aladyn_mods.sim_box.error_check(ierror, 'Memory alloc_atoms_sys error in alloc_aladyn_mods.atoms...')

    if ierror != 0:
        aladyn_ANN.alloc_atoms_ANN(ierror)
        aladyn_mods.sim_box.error_check(ierror, 'Memory alloc_atoms_ANN error in alloc_aladyn_mods.atoms...')
        return

    aladyn_mods.atoms.alloc_atoms_MD(ierror)  # ! alloc x1,.. x5 !

    nbufsize = 16 * aladyn_mods.sim_box.natoms_alloc  # ! max isize in finder_MD !
    if ierror == 0:
        aladyn_mods.sim_box.alloc_buffers(ierror)

    aladyn_mods.sim_box.error_check(ierror, 'MEMORY alloc_buffers error in alloc_aladyn_mods.atoms...')

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
    itime = int(aladyn_mods.sim_box.real_time)

    dt1 = 1.0 / aladyn_mods.atoms.dt_step  # ! 1/[ps] !

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

    ffname = aladyn_mods.sim_box.file_path + fname
    f = open(ffname)

    xtmx = aladyn_mods.sim_box.h[1][1] * 0.5  # ! half the size in X in [Ang.] !
    xtmn = -xtmx
    ytmx = aladyn_mods.sim_box.h[2][2] * 0.5  # ! half the size in Y in [Ang.] !
    ytmn = -ytmx
    ztmx = aladyn_mods.sim_box.h[3][3] * 0.5  # ! half the size in Z in [Ang.] !
    ztmn = -ztmx

    print('#', xtmn, ytmn, ztmn)
    print('#', xtmx, ytmx, ztmx)
    print('#', xtmn, ytmn, ztmn)
    print('#', xtmx, ytmx, ztmx)
    print('#', aladyn_mods.sim_box.nbas, aladyn_mods.sim_box.natoms, aladyn_mods.sim_box.natoms_buf,
          aladyn_mods.sim_box.natoms_free)
    print('#', aladyn_mods.sim_box.r_plt, aladyn_mods.sim_box.mdx, aladyn_mods.sim_box.mdy, aladyn_mods.sim_box.mdz)
    print('#', aladyn_mods.sim_box.ibcx, aladyn_mods.sim_box.ibcy, aladyn_mods.sim_box.ibcz)
    print('#', aladyn_mods.sim_box.ipo, aladyn_mods.sim_box.ipl)
    print('#', aladyn_mods.pot_module.PotEnrg_atm, aladyn_mods.sim_box.T_sys)
    print('#', aladyn_mods.pot_module.PotEnrg_atm, aladyn_mods.sim_box.T_sys)

    for kk in range(1, aladyn_mods.sim_box.natoms + 1):
        Cx = aladyn_mods.sim_box.h[1][1] * aladyn_mods.atoms.sx[kk] + aladyn_mods.sim_box.h[1][2] * \
             aladyn_mods.atoms.sy[kk] + aladyn_mods.sim_box.h[1][3] * aladyn_mods.atoms.sz[kk]
        Cy = aladyn_mods.sim_box.h[2][2] * aladyn_mods.atoms.sy[kk] + aladyn_mods.sim_box.h[2][3] * \
             aladyn_mods.atoms.sz[kk]
        Cz = aladyn_mods.sim_box.h[3][3] * aladyn_mods.atoms.sz[kk]

        ntp = aladyn_mods.atoms.ntype[kk]  # ! write type as in pot.dat file !
        print(aladyn_mods.atoms.ident[kk], Cx, Cy, Cz, ntp, 0)

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

    filename0 = ""

    ierror = 0

    r_ADP_cut = 0.0
    iatom_types = 1  # ! Number of chemical elements !

    # ==== Reading chemical species and filenames of the potential ======

    ifile_numbs = iatom_types * (iatom_types + 1) / 2
    ipair_types = pow(iatom_types, 2)

    aladyn_mods.pot_module.alloc_pot_types(ierror)  # ! in pot_module !
    # ! allocates arrays that are common to ALL pot file types and formats !

    aladyn_mods.sim_box.error_check(ierror, 'alloc_pot_types error in read_pot_dat...')

    aladyn_mods.pot_module.ielement[0] = 0
    aladyn_mods.pot_module.elem_symb[0] = 'Al'
    aladyn_mods.pot_module.gram_mol[0] = 26.982
    ielem = aladyn_mods.pot_module.numb_elem_Z(aladyn_mods.pot_module.elem_symb[0])
    aladyn_mods.pot_module.ielement[0] = ielem  # ! Z - number !
    aladyn_mods.pot_module.amass[0] = aladyn_mods.pot_module.gram_mol[0] * aladyn_mods.constants.atu
    # ! convert to atomic units !
    # ! atu = 100.0/(cavog * evjoul) = 0.010364188 eV.ps^2/nm^2 !

    iPOT_func_type = 1
    aladyn_mods.pot_module.filename[0] = './ANN.dat'

    print(' CHEMICAL ELEMENTS:', '   TYPE   ELEMENT', '    Z    Atomic Mass  POT_func_type')
    for i in range(1, iatom_types + 1):
        print(i, aladyn_mods.pot_module.elem_symb[i], aladyn_mods.pot_module.ielement[i],
              aladyn_mods.pot_module.gram_mol[i], iPOT_func_type)

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

    for n in range(1, aladyn_mods.sim_box.natoms + 1):
        ntp = aladyn_mods.atoms.ntype[n]  # ! corresponds to ann.dat type order !
        if (ntp > aladyn_mods.atoms.iatom_types) or (ntp == 0):
            ierror = 1
            print('ERROR: atom n=', n, ' of id=', aladyn_mods.atoms.ident[n], ' is of unknown type:', ntp)
            break
        else:
            natoms_of_type[ntp] = natoms_of_type[ntp] + 1

    if ierror != 0:
        aladyn.PROGRAM_END(1)

    print('ncell_per_node=', aladyn_mods.sim_box.ncell_per_node)
    print('ncell=', aladyn_mods.sim_box.ncell)
    print('cell_volume=', aladyn_mods.sim_box.cell_volume, ' [Ang^3];')
    print('    Atoms allocated per node:', aladyn_mods.sim_box.natoms_alloc)
    print('    Atoms allocated per cell:', aladyn_mods.sim_box.natoms_per_cell)
    print('    Atoms allocated per 3x3x3 cells:', aladyn_mods.sim_box.natoms_per_cell3)
    print('Neighbors allocated per atom:', aladyn_mods.sim_box.nbrs_per_atom)

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

    file_in = ""

    str82 = ""
    LINE = ""
    LINE2 = ""
    DUM2 = ""

    rsmax0 = 1.0

    f = open('structure.plt', "r")

    itime = int(aladyn_mods.sim_box.start_time)
    print(' Start time:', itime, ' [MCS]')

    print('Reading structure.plt ...')

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
    f = open('structure.plt', "r")

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

    aladyn_mods.sim_box.h[1][1] = (xtmx - xtmn)  # ! system size from second 2 lines !
    aladyn_mods.sim_box.h[2][2] = (ytmx - ytmn)
    aladyn_mods.sim_box.h[3][3] = (ztmx - ztmn)

    aladyn_mods.sim_box.h[1][2] = 0.0  # ! ORT structure !
    aladyn_mods.sim_box.h[1][3] = 0.0
    aladyn_mods.sim_box.h[2][3] = 0.0
    aladyn_mods.sim_box.h[2][1] = 0.0
    aladyn_mods.sim_box.h[3][1] = 0.0
    aladyn_mods.sim_box.h[3][2] = 0.0

    aladyn_mods.sim_box.matinv(aladyn_mods.sim_box.h, aladyn_mods.sim_box.hi, dh)

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

        aladyn_mods.atoms.ident[i] = int(id)
        aladyn_mods.atoms.rx[i] = float(xk)
        aladyn_mods.atoms.ry[i] = float(yk)
        aladyn_mods.atoms.rz[i] = float(zk)
        aladyn_mods.atoms.ntype[i] = int(ktype)

    f.close()
    # ! End of read_structure_plt !

#
# ---------------------------------------------------------------------
# Read input structure files in plt or lammps format
# Note: read_structure must be called AFTER read_pot and read_com
# ---------------------------------------------------------------------
#

def read_structure():

    if aladyn_mods.pot_module.INP_STR_TYPE:
        read_structure_plt()  # ! plt type !

    print(' ')
    print('h(i,j) matrix:')
    print(aladyn_mods.sim_box.h[1][1], aladyn_mods.sim_box.h[1][2], aladyn_mods.sim_box.h[1][3])
    print(aladyn_mods.sim_box.h[2][1], aladyn_mods.sim_box.h[2][2], aladyn_mods.sim_box.h[2][3])
    print(aladyn_mods.sim_box.h[3][1], aladyn_mods.sim_box.h[3][2], aladyn_mods.sim_box.h[3][3])
    print('Crystal structure has ', aladyn_mods.sim_box.natoms, ' atoms')

    structure_chem()

    aladyn_mods.sim_box.matinv(aladyn_mods.sim_box.h, aladyn_mods.sim_box.hi, dh)  # ! h * hi = I !
    hi11 = aladyn_mods.sim_box.hi[1][1], hi12 = aladyn_mods.sim_box.hi[1][2], hi13 = aladyn_mods.sim_box.hi[1][3]
    hi22 = aladyn_mods.sim_box.hi[2][2], hi23 = aladyn_mods.sim_box.hi[2][3], hi33 = aladyn_mods.sim_box.hi[3][3]

    for n in range(1, aladyn_mods.sim_box.natoms + 1):
        aladyn_mods.atoms.sx[n] = hi11 * aladyn_mods.atoms.rx[n] + hi12 * aladyn_mods.atoms.ry[n] + \
                                  hi13 * aladyn_mods.atoms.rz[n]
        aladyn_mods.atoms.sy[n] = hi22 * aladyn_mods.atoms.ry[n] + hi23 * aladyn_mods.atoms.rz[n]
        aladyn_mods.atoms.sz[n] = hi33 * aladyn_mods.atoms.rz[n]
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

    for i in range(0, aladyn_mods.atoms.iatom_types + 1):
        aladyn_mods.pot_module.iZ_elem_in_com[i] = 13

    # --- Some Initializations ---

    real_time = start_time  # ! [fms] !
    T_sys = aladyn_mods.sim_box.T_set

    r_cut_short = aladyn_mods.pot_module.r_cut_off  # ! pot. cut off !

    size = aladyn_mods.pot_module.r_cut_off

    aladyn_mods.pot_module.r2_cut_off = pow(aladyn_mods.pot_module.r_cut_off, 2)  # ! [Ang^2] !
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
