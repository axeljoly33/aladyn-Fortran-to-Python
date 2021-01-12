#
# ------------------------------------------------------------------
# 01-12-2021
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
import os

import atoms
import sim_box
import constants
import pot_module
import PROGRAM_END
import aladyn_ANN


#
# ------------------------------------------------------------------
#

# ! IO beginning !

Lines_INI = 0
Lines_Read = 0
ndof_flag = 0

#
# --------------------------------------------------------------------
#

def finder():

    sim_box.natoms_in_cell = [0] * (sim_box.ncell_per_node + 1)  # ! speedup: 1.450 !

    sim_box.cellix = float(sim_box.nnx)
    sim_box.celliy = float(sim_box.nny)
    sim_box.celliz = float(sim_box.nnz)

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

        ix = int((atoms.sx[i] + 0.5) * sim_box.cellix)
        iy = int((atoms.sy[i] + 0.5) * sim_box.celliy)
        iz = int((atoms.sz[i] + 0.5) * sim_box.celliz)

        # put atom "i" in cell "(ix,iy,iz)"

        icell = operator.mod(iz + sim_box.iZ_shift, sim_box.nnz) * sim_box.nXYlayer + \
                operator.mod(iy + sim_box.iY_shift, sim_box.nny) * sim_box.nnx + \
                operator.mod(ix + sim_box.nnx, sim_box.nnx)

        sim_box.ncell_of_atom[i] = icell

        if sim_box.natoms_in_cell[icell] < sim_box.natoms_per_cell:
            sim_box.natoms_in_cell[icell] += 1
            sim_box.n_in_cell[sim_box.natoms_in_cell[icell]][icell] = i
        else:
            ierror = 1
    # ! do i=1,natoms !

    sim_box.error_check(ierror, 'ERROR: FINDER: Too many atoms per cell...')

    # ! End of finder !

#
# ---------------------------------------------------------------------
#

def alloc_atoms():

    print('ALLOCATE_ATOMS: ', sim_box.natoms_alloc)

    ierr_acc = 0
    ierror = 0

    ierror = atoms.alloc_atoms_sys()
    sim_box.error_check(ierror, 'Memory alloc_atoms_sys error in alloc_atoms...')

    if ierror != 0:

        return

    ierror = aladyn_ANN.alloc_atoms_ANN()
    sim_box.error_check(ierror, 'Memory alloc_atoms_ANN error in alloc_atoms...')

    ierror = atoms.alloc_atoms_MD()  # ! alloc x1,.. x5 !

    sim_box.nbufsize = 16 * sim_box.natoms_alloc  # ! max isize in finder_MD !
    if ierror == 0:
        ierror = sim_box.alloc_buffers()

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
    h_out = [[0.0] * (3 + 1) for i in range(3 + 1)]

    isize = 7
    itime = int(sim_box.real_time)

    dt1 = 1.0 / atoms.dt_step  # ! 1/[ps] !

    fname = 'structure.{0:8d}.plt'.format(itime)
    fname = fname.replace(" ", "0")

    ffname = sim_box.file_path + fname
    if os.path.exists(fname):
        os.remove(fname)
    else:
        print("The file does not exist")
    f = open(fname, "x")

    xtmx = sim_box.h[1][1] * 0.5  # ! half the size in X in [Ang.] !
    xtmn = -xtmx
    ytmx = sim_box.h[2][2] * 0.5  # ! half the size in Y in [Ang.] !
    ytmn = -ytmx
    ztmx = sim_box.h[3][3] * 0.5  # ! half the size in Z in [Ang.] !
    ztmn = -ztmx

    line = ""
    line = '#' + ' ' + str(xtmn) + ' ' + str(ytmn) + ' ' + str(ztmn)
    f.write(line)
    line = '#' + ' ' + str(xtmx) + ' ' + str(ytmx) + ' ' + str(ztmx)
    f.write(line)
    line = '#' + ' ' + str(xtmn) + ' ' + str(ytmn) + ' ' + str(ztmn)
    f.write(line)
    line = '#' + ' ' + str(xtmx) + ' ' + str(ytmx) + ' ' + str(ztmx)
    f.write(line)
    line = '#' + ' ' + str(sim_box.nbas) + ' ' + str(sim_box.natoms) + ' ' + str(sim_box.natoms_buf) + ' ' + \
           str(sim_box.natoms_free)
    f.write(line)
    line = '#' + ' ' + str(sim_box.r_plt) + ' ' + str(sim_box.mdx) + ' ' + str(sim_box.mdy) + ' ' + str(sim_box.mdz)
    f.write(line)
    line = '#' + ' ' + str(sim_box.ibcx) + ' ' + str(sim_box.ibcy) + ' ' + str(sim_box.ibcz)
    f.write(line)
    line = '#' + ' ' + str(sim_box.ipo) + ' ' + str(sim_box.ipl)
    f.write(line)
    line = '#' + ' ' + str(pot_module.PotEnrg_atm) + '  ' + str(sim_box.T_sys)
    f.write(line)

    for kk in range(1, sim_box.natoms + 1):
        Cx = sim_box.h[1][1] * atoms.sx[kk] + sim_box.h[1][2] * atoms.sy[kk] + sim_box.h[1][3] * atoms.sz[kk]
        Cy = sim_box.h[2][2] * atoms.sy[kk] + sim_box.h[2][3] * atoms.sz[kk]
        Cz = sim_box.h[3][3] * atoms.sz[kk]

        ntp = atoms.ntype[kk]  # ! write type as in pot.dat file !
        line = str(atoms.ident[kk]) + ' ' + str(Cx) + ' ' + str(Cy) + ' ' + str(Cz) + ' ' + str(ntp) + ' ' + str(0)
        f.write(line)

    # ! do kk = 1, natoms !

    line = str(0)  # ! 0 - no velocities; 1 - vel. !
    f.write(line)

    f.close()

    # ! End of write_structure_plt !

#
# --------------------------------------------------------------------
# This subroutine reads pot.dat file
# --------------------------------------------------------------------
#

def read_pot_dat():

    filename0 = ""

    ierror = 0

    r_ADP_cut = 0.0
    atoms.iatom_types = 1  # ! Number of chemical elements !

    # ==== Reading chemical species and filenames of the potential ======

    pot_module.ifile_numbs = atoms.iatom_types * (atoms.iatom_types + 1) / 2
    atoms.ipair_types = pow(atoms.iatom_types, 2)

    pot_module.alloc_pot_types()  # ! in pot_module !
    # ! allocates arrays that are common to ALL pot file types and formats !

    sim_box.error_check(ierror, 'alloc_pot_types error in read_pot_dat...')

    pot_module.ielement[0] = 0
    pot_module.elem_symb[1] = 'Al'
    pot_module.gram_mol[1] = 26.982
    ielem = pot_module.numb_elem_Z(pot_module.elem_symb[1])
    pot_module.ielement[1] = ielem  # ! Z - number !
    pot_module.amass[1] = pot_module.gram_mol[1] * constants.atu
    # ! convert to atomic units !
    # ! atu = 100.0/(cavog * evjoul) = 0.010364188 eV.ps^2/nm^2 !

    pot_module.iPOT_func_type = 1
    pot_module.filename = './ANN.dat'

    print('CHEMICAL ELEMENTS:', '   TYPE   ELEMENT', '    Z    Atomic Mass  POT_func_type')
    for i in range(1, atoms.iatom_types + 1):
        print('                       ', i, '    ', pot_module.elem_symb[i], '     ',
              pot_module.ielement[i], '   ', pot_module.gram_mol[i], '        ', pot_module.iPOT_func_type)

    pot_module.nelem_in_com = atoms.iatom_types

    pot_module.iPOT_file_type = 10  # ! ANN file with a trained NN !

    print('Potential file format: ANN')
    pot_module.n_pot_files = 1

    sim_box.MC_rank = 3

    return
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
    for i in range(len(pot_module.natoms_of_type)):
        pot_module.natoms_of_type[i] = 0

    for n in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[n]  # ! corresponds to ann.dat type order !
        if (ntp > atoms.iatom_types) or (ntp == 0):
            ierror = 1
            print('ERROR: atom n=', n, ' of id=', atoms.ident[n], ' is of unknown type:', ntp)
            break
        else:
            pot_module.natoms_of_type[ntp] = pot_module.natoms_of_type[ntp] + 1

    if ierror != 0:
        PROGRAM_END.PROGRAM_END(1)

    print('\nncell_per_node=', sim_box.ncell_per_node, 'ncell=', sim_box.ncell, 'cell_volume=', sim_box.cell_volume,
          ' [Ang^3];')
    print('    Atoms allocated per node:', sim_box.natoms_alloc)
    print('    Atoms allocated per cell:', sim_box.natoms_per_cell)
    print('    Atoms allocated per 3x3x3 cells:', sim_box.natoms_per_cell3)
    print('Neighbors allocated per atom:', sim_box.nbrs_per_atom)

    # ! End of structure_chem !

#
# ---------------------------------------------------------------------
#  ***  read the input atomic structure for MD simulation          ***
#       using plt format from "sold"
# ---------------------------------------------------------------------
#

def read_structure_plt():
    global ndof_flag

    file_in = ""

    str82 = ""
    LINE = ""
    LINE2 = ""
    DUM2 = ""

    rsmax0 = 1.0

    f = open('structure.plt', "r")

    itime = int(sim_box.start_time)
    # ! print('Start time:', itime, ' [MCS]') !

    print('Reading ', 'structure.plt' ,' ...')

    fields = f.readline().strip().split()  # ! 1st line !
    DUM2 = fields[0]
    xmn = float(fields[1])
    ymn = float(fields[2])
    zmn = float(fields[3])

    fields = f.readline().strip().split()  # ! 2nd line !
    DUM2 = fields[0]
    xmx = float(fields[1])
    ymx = float(fields[2])
    zmx = float(fields[3])

    fields = f.readline().strip().split()  # ! 3rd line !
    DUM2 = fields[0]
    xtmn = float(fields[1])
    ytmn = float(fields[2])
    ztmn = float(fields[3])

    fields = f.readline().strip().split()  # ! 4th line !
    DUM2 = fields[0]
    xtmx = float(fields[1])
    ytmx = float(fields[2])
    ztmx = float(fields[3])

    fields = f.readline().strip().split()  # ! 5th line !
    DUM2 = fields[0]
    sim_box.nbas = int(fields[1])
    sim_box.natoms_tot = int(fields[2])
    sim_box.natoms_buf = int(fields[3])
    sim_box.natoms_free = int(fields[4])

    # ! total, buffer, free atoms !

    fields = f.readline().strip().split()  # ! 6th line !
    DUM2 = fields[0]
    sim_box.r_plt = float(fields[1])
    sim_box.mdx = int(fields[2])
    sim_box.mdy = int(fields[3])
    sim_box.mdz = int(fields[4])

    fields = f.readline().strip().split()  # ! 7th line !
    DUM2 = fields[0]
    sim_box.ibcx = int(fields[1])
    sim_box.ibcy = int(fields[2])
    sim_box.ibcz = int(fields[3])

    fields = f.readline().strip().split()  # ! 8th line !
    DUM2 = fields[0]
    sim_box.ipo = int(fields[1])
    sim_box.ipl = int(fields[2])

    fields = f.readline().strip().split()  # ! 9th line !
    DUM2 = fields[0]
    pot_module.PotEnrg_atm = float(fields[1])

    # Check if the plt file has a dof column

    fields = f.readline().strip().split()  # ! 10th line !
    LINE = fields
    LINE.append('-1')  # ! add "-1" to line end !
    LINE2 = LINE
    # ! print('LINE2:', LINE2) !

    fields = LINE2
    id = int(fields[0])
    xk = float(fields[1])
    yk = float(fields[2])
    zk = float(fields[3])
    ktype = int(fields[4])
    kdof = int(fields[5])
    # ! print(id, xk, yk, zk, ktype, kdof) !

    if kdof == -1:
        kdof_exist = 0  # ! old plt file (no constrains) !
    else:
        kdof_exist = 1  # ! new plt file with constrains !

    f.close()
    f = open('structure.plt', "r")

    fields = f.readline().strip().split()  # ! 1st line !
    DUM2 = fields[0]
    xmn = float(fields[1])
    ymn = float(fields[2])
    zmn = float(fields[3])

    fields = f.readline().strip().split()  # ! 2nd line !
    DUM2 = fields[0]
    xmx = float(fields[1])
    ymx = float(fields[2])
    zmx = float(fields[3])

    fields = f.readline().strip().split()  # ! 3rd line !
    DUM2 = fields[0]
    xtmn = float(fields[1])
    ytmn = float(fields[2])
    ztmn = float(fields[3])

    fields = f.readline().strip().split()  # ! 4th line !
    DUM2 = fields[0]
    xtmx = float(fields[1])
    ytmx = float(fields[2])
    ztmx = float(fields[3])

    fields = f.readline().strip().split()  # ! 5th line !
    DUM2 = fields[0]
    sim_box.nbas = int(fields[1])
    sim_box.natoms_tot = int(fields[2])
    sim_box.natoms_buf = int(fields[3])
    sim_box.natoms_free = int(fields[4])

    # ! total, buffer, free atoms !

    fields = f.readline().strip().split()  # ! 6th line !
    DUM2 = fields[0]
    sim_box.r_plt = float(fields[1])
    sim_box.mdx = int(fields[2])
    sim_box.mdy = int(fields[3])
    sim_box.mdz = int(fields[4])

    fields = f.readline().strip().split()  # ! 7th line !
    DUM2 = fields[0]
    sim_box.ibcx = int(fields[1])
    sim_box.ibcy = int(fields[2])
    sim_box.ibcz = int(fields[3])

    fields = f.readline().strip().split()  # ! 8th line !
    DUM2 = fields[0]
    sim_box.ipo = int(fields[1])
    sim_box.ipl = int(fields[2])

    fields = f.readline().strip().split()  # ! 9th line !
    DUM2 = fields[0]
    pot_module.PotEnrg_atm = float(fields[1])

    # ! print('kdof_exist=', kdof_exist) !

    # ! First spread the atoms uniformly to all nodes !

    sim_box.natoms = sim_box.natoms_tot

    sim_box.h[1][1] = (xtmx - xtmn)  # ! system size from second 2 lines !
    sim_box.h[2][2] = (ytmx - ytmn)
    sim_box.h[3][3] = (ztmx - ztmn)

    sim_box.h[1][2] = 0.0  # ! ORT structure !
    sim_box.h[1][3] = 0.0
    sim_box.h[2][3] = 0.0
    sim_box.h[2][1] = 0.0
    sim_box.h[3][1] = 0.0
    sim_box.h[3][2] = 0.0

    dh = sim_box.matinv(sim_box.h, sim_box.hi)

    # ! print('Input file Pot. Enrg=', pot_module.PotEnrg_atm) !
    print(' ')

    sim_box.nxold = 0  # ! first node_config call !
    sim_box.nyold = 0
    sim_box.nzold = 0
    nflag = node_config()   # ! gets ncell, ncell_per_node  !
                            # ! natoms_alloc, call alloc_atoms !
    ndof_flag = 0

    for i in range(1, sim_box.natoms + 1):
        idof = 0
        if kdof_exist != 0:
            fields = f.readline().strip().split()
            id = int(fields[0])
            xk = float(fields[1])
            yk = float(fields[2])
            zk = float(fields[3])
            ktype = int(fields[4])
            idof = int(fields[5])
            ndof_flag = operator.ior(ndof_flag, idof)  # ! collect DOF constraints !
        else:
            fields = f.readline().strip().split()
            id = int(fields[0])
            xk = float(fields[1])
            yk = float(fields[2])
            zk = float(fields[3])
            ktype = int(fields[4])

        atoms.ident[i] = id
        atoms.rx[i] = xk
        atoms.ry[i] = yk
        atoms.rz[i] = zk
        atoms.ntype[i] = ktype

    f.close()

    return
    # ! End of read_structure_plt !

#
# ---------------------------------------------------------------------
# Read input structure files in plt or lammps format
# Note: read_structure must be called AFTER read_pot and read_com
# ---------------------------------------------------------------------
#

def read_structure():

    if pot_module.INP_STR_TYPE:
        read_structure_plt()  # ! plt type !

    print(' ')
    print('h(i,j) matrix:')
    print(sim_box.h[1][1], sim_box.h[1][2], '', sim_box.h[1][3])
    print(sim_box.h[2][1], '', sim_box.h[2][2], sim_box.h[2][3])
    print(sim_box.h[3][1], '', sim_box.h[3][2], '', sim_box.h[3][3])
    print('Crystal structure has', sim_box.natoms, ' atoms')

    structure_chem()

    sim_box.matinv(sim_box.h, sim_box.hi)  # ! h * hi = I !
    sim_box.hi11 = sim_box.hi[1][1]
    sim_box.hi12 = sim_box.hi[1][2]
    sim_box.hi13 = sim_box.hi[1][3]
    sim_box.hi22 = sim_box.hi[2][2]
    sim_box.hi23 = sim_box.hi[2][3]
    sim_box.hi33 = sim_box.hi[3][3]

    for n in range(1, sim_box.natoms + 1):
        atoms.sx[n] = sim_box.hi11 * atoms.rx[n] + sim_box.hi12 * atoms.ry[n] + sim_box.hi13 * atoms.rz[n]
        atoms.sy[n] = sim_box.hi22 * atoms.ry[n] + sim_box.hi23 * atoms.rz[n]
        atoms.sz[n] = sim_box.hi33 * atoms.rz[n]

    # ! End of read_structure !

#
# ---------------------------------------------------------------------
#   ***  read the input parameters for md calculation  ***************
#        Called only once in Main after read_pot to read aladyn.com
#        first initialization lines.
# ---------------------------------------------------------------------
#

def read_com():

    LINE0 = ""
    LINE = ""
    LeadChar = ""
    chem = ""
    str_format_inp = ""
    str_format_out = ""
    Astring = ""

    no_line = 0
    sim_box.new_mu0 = 0
    sim_box.new_alpha0 = 0
    sim_box.iensemble = 0
    sim_box.itriclinic = 0  # ! assume ORTHORHOMBIC system !
    atoms.at_vol_inp = -1.0  # ! no preset atomic volume !
    sim_box.iHEAD_WRITE = 1  # ! write a header line in the *.dat file !
    sim_box.iCrack = 0  # ! No crack !

    ierror = 0
    sim_box.start_time = 0.0

    for i in range(0, atoms.iatom_types + 1):
        pot_module.iZ_elem_in_com[i] = 13

    # --- Some Initializations ---

    sim_box.real_time = sim_box.start_time  # ! [fms] !
    sim_box.T_sys = sim_box.T_set

    pot_module.r_cut_short = pot_module.r_cut_off  # ! pot. cut off !

    sim_box.size = pot_module.r_cut_off

    pot_module.r2_cut_off = pow(pot_module.r_cut_off, 2)  # ! [Ang^2] !
    pot_module.r2_cut_short = pow(pot_module.r_cut_short, 2)
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
    sim_box.nstep = 10
    sim_box.measure_step = 1
    atoms.dt_step = 0.001  # ! 1 fs = 0.001 ps !
    sim_box.start_time = 0.0
    sim_box.T_set = 100.0  # ! K !

    k = 0
    N_args = len(sys.argv)
    GETARG = sys.argv

    while k < N_args:
        str_opt = GETARG[k]
        k += 1

        if (str_opt == '-v') or (str_opt == '-V'):
            str_num = GETARG[k]
            k += 1
            iver = str_num
            print('- Version ', iver)
        elif (str_opt == '-n') or (str_opt == '-N'):
            str_num = GETARG[k]
            k += 1
            sim_box.nstep = str_num
            print('- Executing ', sim_box.nstep, ' MD steps')
        elif (str_opt == '-m') or (str_opt == '-M'):
            str_num = GETARG[k]
            k += 1
            sim_box.measure_step = str_num
            print('- Measure at each ', sim_box.measure_step, ' MD steps')
        # ! if((str_opt(1:2)... !

    # ! do while(k.le.N_args) !

    sim_box.nodes_on_Y = 1
    sim_box.nodes_on_Z = 1

    # ! End of read_Args !

#
# ------------------------------------------------------------------
#

def link_cell_setup():

    # !
    # ! subroutine to set up a cell structure in which to assign atoms
    # !

    ncell_per_node_old = sim_box.ncell_per_node

    # ! Parallel values               ! 1D, Serial values        !
    sim_box.mynodZ = 0
    sim_box.mynodY = 0

    sim_box.ncZ_per_node = sim_box.nnz
    sim_box.lZstart = 0
    sim_box.lZend = sim_box.nnz - 1
    sim_box.iZ_shift = sim_box.nnz
    sim_box.ncY_per_node = sim_box.nny
    sim_box.lYstart = 0
    sim_box.lYend = sim_box.nny - 1
    sim_box.iY_shift = sim_box.nny
    sim_box.nXYlayer = sim_box.nnx * sim_box.nny
    sim_box.ncell_per_node = int(sim_box.nnx * sim_box.nny * sim_box.nnz)

    # !     write(6, 10) nodes_on_Y, ncell_per_node
    # ! 10 format('link_cell_setup: nodes_on_Y=', i2, ' ncell_per_node=', i5)

    sim_box.cellix = float(sim_box.nnx)
    sim_box.celliy = float(sim_box.nny)
    sim_box.celliz = float(sim_box.nnz)

    sim_box.ncell_per_node = int((sim_box.ncell_per_node / 8 + 1) * 8)

    # ! *** Neighbor nodes index ***

    # !  kp1YZ | kp1Z | kp1ZY  !
    # ! ---------------------- !
    # !   km1Y | mynod | kp1Y  !
    # ! ---------------------- !
    # !  km1YZ | km1Z | km1ZY  !

    sim_box.kp1Z = 0  # ! Serial mode !
    sim_box.km1Z = 0
    sim_box.kp1Y = 0
    sim_box.km1Y = 0

    sim_box.kp1YZ = 0
    sim_box.kp1ZY = 0
    sim_box.km1YZ = 0
    sim_box.km1ZY = 0

    if sim_box.ncell_per_node > ncell_per_node_old:
        if not sim_box.id_of_cell:
            ierror = 0
            ierror = sim_box.alloc_cells()  # ! alloc_ates cells in aladyn_mods !
            sim_box.error_check(ierror, 'ERROR in alloc_cells...')

    # !
    # ! Collect cell ids and indices for use in get_neighbors
    # !

    k = 0

    for iz in range(sim_box.lZstart, sim_box.lZend + 1):
        izz = iz + sim_box.iZ_shift
        iz_nr = izz % sim_box.nnz

        if sim_box.nodes_on_Y == 1:  # ! 1D node topology !
            for iy in range(0, sim_box.nny - 1 + 1):
                iyy = iy + sim_box.nny
                iy_nr = iyy % sim_box.nny
                for ix in range(0, sim_box.nnx - 1 + 1):
                    k = k + 1
                    ixx = ix + sim_box.nnx
                    sim_box.id_of_cell[k] = iz_nr * sim_box.nXYlayer + iy_nr * sim_box.nnx + \
                                                        (ixx % sim_box.nnx)
        else:  # ! 2D node topology !
            for iy in range(sim_box.lYstart, sim_box.lYend + 1):
                iyy = iy + sim_box.iY_shift
                iy_nr = iyy % sim_box.nny
                for ix in range(0, sim_box.nnx - 1 + 1):
                    k = k + 1
                    ixx = ix + sim_box.nnx
                    sim_box.id_of_cell[k] = iz_nr * sim_box.nXYlayer + iy_nr * sim_box.nnx + \
                                                        (ixx % sim_box.nnx)
        # ! if (nodes_on_Y.eq.1)... !

    sim_box.ncells_all = k

    return
    # ! End of link_cell_setup !

# !
# ! -------------------------------------------------------------------
# !
# !
# ! -------------------------------------------------------------------
# ! Finds the best number of cells in a given direction (nnx,nny,nnz),
# ! which commensurate with MC_rank and nodes_on_X,Y,Z
# ! -------------------------------------------------------------------
# !

def nnd_fit(nodes_on_D, iD):

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
# !  Looks for optimal node architecture configuration at a given
# !  number of nodes on X:nodes_X, on Y:nodes_Y, and on Z:nodes_Z
# ! -------------------------------------------------------------------
# !

def get_config(i1, i2, i3, nodes_X, nodes_Y, nodes_Z):

    # !      write(50,10) i1,i2,i3, nodes_X, nodes_Y, nodes_Z
    # !  10  format('get_config(',6i3,')')

    nnx_min, nny_min, nnz_min = 0, 0, 0

    ierror = 0
    res_nnd_fit = nnd_fit(nodes_X, i1)
    sim_box.nnx, nnx_min, sim_box.MC_rank_X = res_nnd_fit[0], res_nnd_fit[1], res_nnd_fit[2]
    if sim_box.nnx == 0:
        ierror = operator.ior(ierror, 1)
    res_nnd_fit = nnd_fit(nodes_Y, i2)
    sim_box.nny, nny_min, sim_box.MC_rank_Y = res_nnd_fit[0], res_nnd_fit[1], res_nnd_fit[2]
    if sim_box.nny == 0:
        ierror = operator.ior(ierror, 2)
    res_nnd_fit = nnd_fit(nodes_Z, i3)
    sim_box.nnz, nnz_min, sim_box.MC_rank_Z = res_nnd_fit[0], res_nnd_fit[1], res_nnd_fit[2]
    if sim_box.nnz == 0:
        ierror = operator.ior(ierror, 4)

    # !      write(50,*)'get_conf: nnx,y,z=',nnx,nny,nnz,ierror
    # !      write(50,*)'get_conf: MC_rank_X,Y,Z=',
    # !    1 MC_rank_X,MC_rank_Y,MC_rank_Z

    nnx_cell = sim_box.nnx / nodes_X
    nny_cell = sim_box.nny / nodes_Y
    nnz_cell = sim_box.nnz / nodes_Z

    # ! Check if cells per node commensurate with MC_ranks !

    nn_mod = nnx_cell % sim_box.MC_rank_X
    if nn_mod != 0:
        ierror = operator.ior(ierror, 1)
    nn_mod = nny_cell % sim_box.MC_rank_Y
    if nn_mod != 0:
        ierror = operator.ior(ierror, 2)
    nn_mod = nnz_cell % sim_box.MC_rank_Z
    if nn_mod != 0:
        ierror = operator.ior(ierror, 4)

    # !      write(50,*) 'get_conf: nnx, y, z_cell=', nnx_cell, nny_cell, nnz_cell, ierror
    # !      write(50,*) ' '

    res = [nnx_min, nny_min, nnz_min, nnx_cell, nny_cell, nnz_cell, ierror]

    return res
    # ! End of get_config !

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

    natoms_alloc_new = 0  # ! local !
    nnx_min, nny_min, nnz_min = 0, 0, 0

    nflag = 0
    ierror = 0
    nodes_on_Yo, nodes_on_Zo = sim_box.nodes_on_Y, sim_box.nodes_on_Z
    sim_box.cell_size_X, sim_box.cell_size_Y, sim_box.cell_size_Z = 0.0, 0.0, 0.0

    i1, i2, i3 = 1, 2, 3

    sim_box.nodes_on_Y, sim_box.nodes_on_Z = 1, 1
    res = get_config(i1, i2, i3, 1, 1, 1)
    nnx_min, nny_min, nnz_min = res[0], res[1], res[2]
    nnx_cell, nny_cell, nnz_cell = res[3], res[4], res[5]
    ierror = res[6]
    nny_try, nnz_try = 1, 1

    if ierror > 0:

        print(' ')
        print('ERROR: Unable to construct a suitable link-cell grid!')
        print(' ')
        print('System Box size:', sim_box.h[i1][i1], sim_box.h[i2][i2],
              sim_box.h[i3][i3], ';   min. cell size=', sim_box.size)

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

        PROGRAM_END.PROGRAM_END(1)

    else:  # ! if(ierror.gt.0)... !
        sim_box.cell_size_X = sim_box.h[i1][i1] / sim_box.nnx
        sim_box.cell_size_Y = sim_box.h[i2][i2] / sim_box.nny
        sim_box.cell_size_Z = sim_box.h[i3][i3] / sim_box.nnz
    # endif ! if(ierror.gt.0)... !

    sim_box.ncell = sim_box.nnx * sim_box.nny * sim_box.nnz

    atoms.sys_vol = sim_box.matinv(sim_box.h, sim_box.hi)
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

    atom_vol2 = 4.0 / 3.0 * 3.141592 * pow(rZ_min, 3)

    # !     write(1000+mynod,*)'sys_vol=',sys_vol,' r_max=',r_max, ' rZ_min=',rZ_min
    # !     write(1000+mynod,*)'atom_vol1=',atom_vol1,' atom_vol2',atom_vol2

    if atom_vol1 < atom_vol2:
        atom_vol = atom_vol1
    else:
        atom_vol = atom_vol2

    # !     write(6,*)'atom_vol=',atom_vol,' cell_volume=', cell_volume,' rZ_min=',rZ_min

    sim_box.cell_volume = (sim_box.cell_size_X + 2.0 * rZ_min) * (sim_box.cell_size_Y + 2.0 * rZ_min) * \
                          (sim_box.cell_size_Z + 2.0 * rZ_min)
    cell3_volume = (3.0 * sim_box.cell_size_X + 2.0 * rZ_min) * (3.0 * sim_box.cell_size_Y + 2.0 * rZ_min) * \
                   (3.0 * sim_box.cell_size_Z + 2.0 * rZ_min)
    sim_box.natoms_per_cell = int(sim_box.cell_volume / atom_vol) + 1
    sim_box.natoms_per_cell = int(sim_box.natoms_per_cell / 8 + 1) * 8
    sim_box.natoms_per_cell3 = int(cell3_volume / atom_vol) + 1
    sim_box.natoms_per_cell3 = int(sim_box.natoms_per_cell3 / 8 + 1) * 8

    nflag = abs(sim_box.nnx - sim_box.nxold) + \
            abs(sim_box.nny - sim_box.nyold) + \
            abs(sim_box.nnz - sim_box.nzold) + \
            abs(sim_box.nodes_on_Y - nodes_on_Yo) + abs(sim_box.nodes_on_Z - nodes_on_Zo)

    # !  reset cell grid if necessary
    if nflag > 0:
        link_cell_setup()

        print('\n', 'Link cell configuration:', '\n', ' axis nodes cells/n thickness; total cell:', sim_box.ncell)

        print('On X: ', 1, ' x  ', sim_box.nnx, '  x   ', sim_box.cell_size_X)
        print('On Y: ', sim_box.nodes_on_Y, ' x  ', int(nny_cell), '  x   ', sim_box.cell_size_Y)
        print('On Z: ', sim_box.nodes_on_Z, ' x  ', int(nnz_cell), ' x   ', sim_box.cell_size_Z)

        print(' ')
    # ! if (nflag.gt.0)... !

    natoms_alloc_new = sim_box.natoms + 100

    if natoms_alloc_new > sim_box.natoms_alloc:  # ! update natoms_alloc !
        sim_box.natoms_alloc = int((int(natoms_alloc_new / 64) + 1) * 64)

    cut_off_vol = 4.0 / 3.0 * 3.141592 * pow(pot_module.r_cut_off, 3)
    sim_box.nbrs_per_atom = int(round(cut_off_vol / atom_vol))  # ! Correct one !
    # ! print('nbrs_per_atom=', sim_box.nbrs_per_atom) !
    sim_box.nbrs_alloc = sim_box.nbrs_per_atom * sim_box.natoms_alloc

    sim_box.nxold = sim_box.nnx
    sim_box.nyold = sim_box.nny
    sim_box.nzold = sim_box.nnz

    if not atoms.ident:
        alloc_atoms()  # ! alloc_ates natoms_alloc atoms in aladyn_IO !

    return nflag
    # ! End of node_config !

# ---------------------------------------------------------------------
#
#      END FILE  ! aladyn_IO !
#
# =====================================================================
