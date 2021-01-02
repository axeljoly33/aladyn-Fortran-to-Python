#
# ------------------------------------------------------------------
# 12-10-2020
#
# General Module Unit for aladyn.f code.
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
# General Module Unit for aladyn.f code
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
import aladyn_MD
import aladyn_IO
import aladyn_ANN
import aladyn


#
# ------------------------------------------------------------------
#

class node_conf:
    node_name = ""
    devicetype = 0
    nACC_devices = 0
    I_have_GPU = 0
    My_GPU_id = 0
    MP_procs = 0
    MP_threads = 0

#
# ------------------------------------------------------------------
#

class group_conf:
    node_name = ""
    devicetype = 0
    nACC_devices = 0
    nodes = 0

#
# ------------------------------------------------------------------
#

class sim_box:
    # *** Set some working parameters here ***
    mynode_conf = node_conf

    file_path = "./"
    Max_Lines_INI = 1024
    CHUNK = 24

    str_filename = ""

    MP_threads = 0  # ! OMP: Number of OMP threads !
    MP_max_threads = 0  # ! OMP: MAx.Number of OMP threads !
    MP_procs = 0  # ! OMP: Number of OMP processors !

    nbufsize = 0

    MC_rank = 0
    MC_rank0 = 0
    MC_rank_X = 0
    MC_rank_Y = 0
    MC_rank_Z = 0
    MC_rank_max = 19  # ! large simple number !

    nn_count = 0

    natoms_alloc = 0
    nbrs_alloc = 0
    ncell_per_node = 0
    natoms = 0
    natoms_tot, natoms_buf, natoms_free = 0, 0, 0
    nbrs_per_atom = 0
    natoms_per_cell, natoms_per_cell3 = 0, 0
    nbas, ibcx, ibcy, ibcz, ipo, ipl, mdx, mdy, mdz = 0, 0, 0, 0, 0, 0, 0, 0, 0
    iend, new_mu, new_mu0, new_alpha, new_alpha0 = 0, 0, 0, 0, 0
    nstep, irigid, irigid0, iensemble, measure_step = 0, 0, 0, 0, 0
    ncpny, ncpnz, ido_finder, nnx, nny, nnz, ncell, nxold, nyold, nzold, ncells_all = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    mpi_nodes, mynod, ncZ_per_node, ncY_per_node, nXYlayer = 0, 0, 0, 0, 0
    nodes_on_Y, lYstart, lYend, mynodY, kp1Y, km1Y, iY_shift = 0, 0, 0, 0, 0, 0, 0
    nodes_on_Z, lZstart, lZend, mynodZ, kp1Z, km1Z, iZ_shift = 0, 0, 0, 0, 0, 0, 0
    kp1YZ, kp1ZY, km1YZ, km1ZY = 0, 0, 0, 0
    itriclinic, iHEAD_WRITE = 0, 0

    ihalt = 0
    nACC_devices = 0
    I_have_GPU, My_GPU = 0, 0

    istep = 0
    Line_Executed = [0] * Max_Lines_INI
    # ! = 0: line has not been executed !
    # ! = 1: line has been executed !
    n_of_all_moved = 0  # ! all MC moved atoms up to now !

    ncell_of_atom = []

    n_of_moved_atom = []

    BkT = 0.0  # ! = 1.0d0 / (Boltz_Kb * T_sys) !
    real_time, start_time, start_run_time = 0.0, 0.0, 0.0
    T_sys, T_set, Ek_sys = 0.0, 0.0, 0.0

    h = [[0.0] * 3 for i1 in range(3)]
    hi = [[0.0] * 3 for i2 in range(3)]

    # ! System shape matrices !
    h1 = [[0.0] * 3 for i3 in range(3)]
    h2 = [[0.0] * 3 for i4 in range(3)]
    h3 = [[0.0] * 3 for i5 in range(3)]
    h4 = [[0.0] * 3 for i6 in range(3)]
    h5 = [[0.0] * 3 for i7 in range(3)]

    h11, h12, h13, h22, h23, h33 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    hi11, hi12, hi13, hi22, hi23, hi33 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    density, a0_nm, r_plt = 0.0, 0.0, 0.0
    cellix, celliy, celliz, size = 0.0, 0.0, 0.0, 0.0
    cell_size_X, cell_size_Y, cell_size_Z, cell_volume = 0.0, 0.0, 0.0, 0.0

    # --- Crack parameters ---

    iCrack, iLine_crack, iPlane_crack = 0, 0, 0
    Crack_tip1, Crack_tip2, Crack_plane = 0.0, 0.0, 0.0
    Crack_rAB = 0.0

    # !
    # ! Forms a crack as:
    # !                     A o - ----------------------o B
    # ! where: A = CrackP_A(Crack_tip1, Crack_plane)
    # ! and B = CrackP_B(Crack_tip2, Crack_plane)
    # ! coord_A(x, y) =
    # ! coord(iLine_crack=(1, 2, 3 for x, y, z), iPlane_crack=(1, 2, 3 for x, y, z))
    # !

    # ! --- Buffers ---

    buf = [0.0]  # ! double precision !
    bufa = [0.0]  # ! double precision !
    bufb = [0.0]  # ! double precision !
    bufc = [0.0]  # ! double precision !

    nbuf = []
    nbufa = []

    kbuf = []
    kbufa = []
    kbufb = []
    kbufc = []

    natoms_in_cell = []

    n_in_cell = [[]]

    ibufY1 = []
    ibufY2 = []
    ibufZ1 = []
    ibufZ2 = []

    id_of_cell = []

    ipack1Y1, ipack2Y2, jpack1Y1, jpack2Y2 = 0, 0, 0, 0
    ipack1Z1, ipack2Z2, jpack1Z1, jpack2Z2 = 0, 0, 0, 0

    U = [0.0] * 97
    C, CD, CM = 0.0, 0.0, 0.0
    Us = [0.0] * 97
    Cs, CDs, CMs = 0.0, 0.0, 0.0
    I97, J97, I97s, J97s = 0, 0, 0, 0

    # !
    # ! ********************************************************
    # !                                                        $
    # !  Calculates the argument of a complex number x + iy    $
    # !                                                        $
    # ! ********************************************************
    # !

    def argument(self, x, y, r):

        # x, y, r = 0.0, 0.0, 0.0
        xr, yr, phi = 0.0, 0.0, 0.0
        PI = 3.141592654

        r = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
        xr = abs(x) / r
        yr = abs(y) / r

        if xr > 0.1:
            phi = math.atan(yr / xr)
        else:
            phi = (PI / 2.0) - math.atan(xr / yr)

        if (x < 0.0) and (y >= 0.0):
            phi = PI - phi
        if (x < 0.0) and (y < 0.0):
            phi = PI + phi
        if (x >= 0.0) and (y < 0.0):
            phi = 2.0 * PI - phi

        return phi

    # !
    # ! ********************************************************
    # !
    # !       RANDOM NUMBER GENERATOR
    # !
    # !       UNIVERSAL RANDOM NUMBER GENERATOR PROPOSED BY MARSAGLIA
    # !       AND ZAMAN IN REPORT FSU - SCRI - 87 - 50
    # !       GENERATES VECTOR 'RVEC' OF LENGTH 'LEN' OF PSEUDORANDOM
    # !       NUMBERS; THE COMMON BLOCK INCLUDES EVERYTHING NEEDED TO
    # !       COMPLETELY SPECIFY THE STATE OF THE GENERATOR.
    # !
    # !       Puts LEN random numbers in bufer buf(1..LEN)
    # !
    # ! ********************************************************
    # !

    def ranmar(self, LEN):

        UNI = 0.0

        for IVEC in range(1, LEN + 1):
            UNI = self.U[self.I97] - self.U[self.J97]
            if UNI < 0.0:
                UNI += 1.0
            self.U[self.I97] = UNI
            self.I97 -= 1
            if self.I97 == 0:
                self.I97 = 97
            self.J97 = self.J97 - 1
            if self.J97 == 0:
                self.J97 = 97
            self.C = self.C - self.CD
            if self.C < 0.0:
                self.C = self.C + self.CM
            UNI = UNI - self.C
            if UNI < 0.0:
                UNI += 1.0
            self.buf[IVEC] = UNI

    # ! ********************************************************
    # !
    # !       RANDOM NUMBER INITIALIZER
    # !
    # !       INITIALIZING ROUTINE FOR ranmar.THE INPUT VALUE SHOULD
    # !       BE IN THE RANGE: 0 <= IJKL <= 900 000 000
    # !       TO GET THE STANDARD VALUES IN THE MARSAGLIA - ZAMAN PAPER
    # !       (I=12, J=34, K=56, L=78) PUT IJKL = 54217137
    # !
    # ! ********************************************************

    def rmarin(self, IJKL):

        T = 0.0

        IJ = IJKL / 30082
        KL = IJKL - IJ * 30082
        I = ((IJ / 177) % 177) + 2
        J = (IJ % 177) + 2
        K = ((KL / 169) % 178) + 1
        L = (KL % 169)

        # ! WRITE(*, *) 'ranmar INITIALIZED: ', IJKL, I, J, K, L
        for II in range(1, 97 + 1):
            S = 0.0
            T = 0.5
            for JJ in range(1, 24 + 1):
                M = (((I * J) % 179) * K) % 179
                I = J
                J = K
                K = M
                L = (53 * L + 1) % 169
                if ((L * M) % 64) >= 32:
                    S = S + T
                T = 0.5 * T
            self.U[II] = S
        self.C = 362436.0 / 16777216.0
        self.CD = 7654321.0 / 16777216.0
        self.CM = 16777213.0 / 16777216.0
        self.I97 = 97
        self.J97 = 33

    # !
    # !----------------------------------------------------------------------
    # !

    def matinv(self, a, b, c):

        # a = [ [0.0]*3 for i in range(3)]
        # b = [ [0.0]*3 for i in range(3)]
        # c = 0.0

        d11 = a[2][2] * a[3][3] - a[2][3] * a[3][2]
        d22 = a[3][3] * a[1][1] - a[3][1] * a[1][3]
        d33 = a[1][1] * a[2][2] - a[1][2] * a[2][1]
        d12 = a[2][3] * a[3][1] - a[2][1] * a[3][3]
        d23 = a[3][1] * a[1][2] - a[3][2] * a[1][1]
        d31 = a[1][2] * a[2][3] - a[1][3] * a[2][2]
        d13 = a[2][1] * a[3][2] - a[3][1] * a[2][2]
        d21 = a[3][2] * a[1][3] - a[1][2] * a[3][3]
        d32 = a[1][3] * a[2][1] - a[2][3] * a[1][1]
        c = a[1][1] * d11 + a[1][2] * d12 + a[1][3] * d13
        b[1][1] = d11 / c
        b[2][2] = d22 / c
        b[3][3] = d33 / c
        b[1][2] = d21 / c
        b[2][3] = d32 / c
        b[3][1] = d13 / c
        b[2][1] = d12 / c
        b[3][2] = d23 / c
        b[1][3] = d31 / c

    # !
    # ! ---------------------------------------------------------------------
    # !

    def alloc_nodes(self, nodes, ierror):

        ialloc = [0] * 2

        # ! if (mynod.eq.0) write(6, * )'alloc_nodes(', nodes, ')'

        self.nbuf = [0] * nodes
        self.nbufa = [0] * nodes

        ierror = 0
        for i in range(1, 2 + 1):
            ierror = ierror + ialloc[i]

        return ierror
        # ! End of alloc_nodes !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def alloc_buffers(self, ierror):

        ialloc = [0] * 14

        # ! if (mynod.eq.0) write(6, * )'alloc_buffers: nbufsize=', nbufsize

        self.buf = [0] * self.nbufsize
        self.bufa = [0] * self.nbufsize
        self.bufb = [0] * self.nbufsize
        self.bufc = [0] * self.nbufsize

        self.ibufY1 = [0] * self.natoms_alloc
        self.ibufY2 = [0] * self.natoms_alloc
        self.ibufZ1 = [0] * self.natoms_alloc
        self.ibufZ2 = [0] * self.natoms_alloc

        self.kbuf = [0] * self.nbufsize
        self.kbufa = [0] * self.nbufsize
        self.kbufb = [0] * self.nbufsize
        self.kbufc = [0] * self.nbufsize
        self.ncell_of_atom = [0] * self.natoms_alloc

        ierror = 0
        for i in range(1, 14 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of alloc_buffers !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def deall_buffers(self, ierror):

        ialloc = [0] * 14

        if self.buf:
            self.buf.clear()
        if self.bufa:
            self.bufa.clear()
        if self.bufb:
            self.bufb.clear()
        if self.bufc:
            self.bufc.clear()

        if self.ibufY1:
            self.ibufY1.clear()
        if self.ibufY2:
            self.ibufY2.clear()
        if self.ibufZ1:
            self.ibufZ1.clear()
        if self.ibufZ2:
            self.ibufZ2.clear()

        if self.kbuf:
            self.kbuf.clear()
        if self.kbufa:
            self.kbufa.clear()
        if self.kbufb:
            self.kbufb.clear()
        if self.kbufc:
            self.kbufc.clear()
        if self.ncell_of_atom:
            self.ncell_of_atom.clear()

        ierror = 0
        for i in range(1, 14 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of deall_buffers !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def alloc_cells(self, ierror):

        ialloc = [0] * 4

        self.id_of_cell = [0] * self.ncell_per_node
        self.natoms_in_cell = [0] * self.ncell_per_node

        self.n_in_cell = [[0] * self.natoms_per_cell for i in range(self.ncell_per_node)]
        self.n_of_moved_atom = [0] * self.ncell_per_node

        ierror = 0
        for i in range(1, 4 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of alloc_cells !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def deall_cells(self, ierror):

        ialloc = [0] * 4

        if self.id_of_cell:
            self.id_of_cell.clear()
        if self.natoms_in_cell:
            self.natoms_in_cell.clear()
        if self.n_in_cell:
            self.n_in_cell.clear()
        if self.n_of_moved_atom:
            self.n_of_moved_atom.clear()

        ierror = 0
        for i in range(1, 4 + 1):
            ierror = ierror + ialloc[i]

        return ierror
        # ! End of deall_cells !

    # !
    # ! -------------------------------------------------------------------
    # !

    def error_check(self, ierror, message):

        if ierror != 0:
            if self.mynod == 0:
                print(message)
            aladyn.PROGRAM_END(1)
        # ! End of error_check !

    # ! End of sim_box !

#
# ------------------------------------------------------------------
#

class atoms:
    nbr_tot, max_nbrs = 0, 0

    iatom_types, ipair_types, icomp_high, icomp_low = 0, 0, 0, 0

    at_vol, at_vol_inp, sys_vol = 0.0, 0.0, 0.0

    sum_dcm = [0.0] * 3
    Sys_dcmR = [0.0] * 3
    Sys_dcmS = 0.0
    # ! System center of mass deviation !

    ident = []
    ntype = []

    # ! ident() -> id number of an atom             !
    # ! ntype() -> chem.type of an atom inside code !

    sx = [0.0]  # ! double precision !
    sy = [0.0]  # ! double precision !
    sz = [0.0]  # ! double precision !
    rx = [0.0]  # ! double precision !
    ry = [0.0]  # ! double precision !
    rz = [0.0]  # ! double precision !

    nbr_list = [[]]

    # !     MD atom arrays:

    dt_step = 0.001  # ! default MD step !

    A_fr, Q_heat = 0.0, 0.0
    Ek_wall = 0.0

    x1 = [0.0]  # ! double precision !
    y1 = [0.0]  # ! double precision !
    z1 = [0.0]  # ! double precision !
    x2 = [0.0]  # ! double precision !
    y2 = [0.0]  # ! double precision !
    z2 = [0.0]  # ! double precision !
    x3 = [0.0]  # ! double precision !
    y3 = [0.0]  # ! double precision !
    z3 = [0.0]  # ! double precision !
    x4 = [0.0]  # ! double precision !
    y4 = [0.0]  # ! double precision !
    z4 = [0.0]  # ! double precision !
    x5 = [0.0]  # ! double precision !
    y5 = [0.0]  # ! double precision !
    z5 = [0.0]  # ! double precision !
    sumPx = [0.0]  # ! double precision !
    sumPy = [0.0]  # ! double precision !
    sumPz = [0.0]  # ! double precision !
    Tscale = [0.0]  # ! double precision !

    frr = [[]]  # ! double precision !

    Ep_of = []  # ! double precision !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def alloc_atoms_sys(self, ierror):

        ialloc = [0] * 12

        self.ident = [0] * sim_box.natoms_alloc
        self.ntype = [0] * sim_box.natoms_alloc

        self.rx = [0] * sim_box.natoms_alloc
        self.ry = [0] * sim_box.natoms_alloc
        self.rz = [0] * sim_box.natoms_alloc

        self.sx = [0] * sim_box.natoms_alloc
        self.sy = [0] * sim_box.natoms_alloc
        self.sz = [0] * sim_box.natoms_alloc

        self.nbr_list = [[0] * sim_box.nbrs_per_atom for i in range(sim_box.natoms_alloc)]

        self.frr = [0] * sim_box.natoms_alloc  # from 3 to natoms_alloc
        self.Ep_of = [0] * sim_box.natoms_alloc

        ierror = 0
        for i in range(1, 12 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of alloc_atoms_sys !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def deall_atoms_sys(self, ierror):

        ialloc = [0] * 12

        if self.ident:
            self.ident.clear()
        if self.ntype:
            self.ntype.clear()

        if self.rx:
            self.rx.clear()
        if self.ry:
            self.ry.clear()
        if self.rz:
            self.rz.clear()

        if self.sx:
            self.sx.clear()
        if self.sy:
            self.sy.clear()
        if self.sz:
            self.sz.clear()

        if self.nbr_list:
            self.nbr_list.clear()

        if self.frr:
            self.frr.clear()
        if self.Ep_of:
            self.Ep_of.clear()

        ierror = 0
        for i in range(1, 12 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of deall_atoms_sys !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def alloc_atoms_MD(self, ierror):

        ialloc = [0] * 20

        self.x1 = [0.0] * sim_box.natoms_alloc
        self.y1 = [0.0] * sim_box.natoms_alloc
        self.z1 = [0.0] * sim_box.natoms_alloc

        self.x2 = [0] * sim_box.natoms_alloc
        self.y2 = [0] * sim_box.natoms_alloc
        self.z2 = [0] * sim_box.natoms_alloc

        self.x3 = [0] * sim_box.natoms_alloc
        self.y3 = [0] * sim_box.natoms_alloc
        self.z3 = [0] * sim_box.natoms_alloc

        self.x4 = [0] * sim_box.natoms_alloc
        self.y4 = [0] * sim_box.natoms_alloc
        self.z4 = [0] * sim_box.natoms_alloc

        self.x5 = [0] * sim_box.natoms_alloc
        self.y5 = [0] * sim_box.natoms_alloc
        self.z5 = [0] * sim_box.natoms_alloc

        self.sumPx = [0.0] * self.iatom_types
        self.sumPy = [0.0] * self.iatom_types
        self.sumPz = [0.0] * self.iatom_types
        self.Tscale = [0] * self.iatom_types

        ierror = 0
        for i in range(1, 20 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of alloc_atoms_MD !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def deall_atoms_MD(self, ierror):

        ialloc = [0] * 20

        if self.x1:
            self.x1.clear()
        if self.y1:
            self.y1.clear()
        if self.z1:
            self.z1.clear()

        if self.x2:
            self.x2.clear()
        if self.y2:
            self.y2.clear()
        if self.z2:
            self.z2.clear()

        if self.x3:
            self.x3.clear()
        if self.y3:
            self.y3.clear()
        if self.z3:
            self.z3.clear()

        if self.x4:
            self.x4.clear()
        if self.y4:
            self.y4.clear()
        if self.z4:
            self.z4.clear()

        if self.x5:
            self.x5.clear()
        if self.y5:
            self.y5.clear()
        if self.z5:
            self.z5.clear()

        if self.sumPx:
            self.sumPx.clear()
        if self.sumPy:
            self.sumPy.clear()
        if self.sumPz:
            self.sumPz.clear()
        if self.Tscale:
            self.Tscale.clear()

        ierror = 0
        for i in range(1, 20 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of deall_atoms_MD !

    # ! End of atoms !

#
# ------------------------------------------------------------------
#

class pot_module:
    maxconstr = 63  # ! Max number of defined    !
    # ! constraints              !
    chem_symb = [""] * 112
    ipot = [0] * 112  # ! maximum numb.elements possible !

    elem_symb = [""]
    Elem_in_com = [""]
    filename = [""]

    PotEnrg_atm, PotEnrg_glb, ecoh = 0.0, 0.0, 0.0

    ifile_numbs, nelem_in_com, n_pot_files = 0, 0, 0
    INP_STR_TYPE, iOUT_STR_TYPE, N_points_max, Nrho_points_max = 0, 0, 0, 0

    maxf = 0  # ! Max number of potential files !
    iPOT_file_type = 0  # ! 0 - plt, *.dat files !
    # ! 1 - plt, *.plt files !
    # ! 2 - LAMMPS files     !

    iPOT_file_ver = 0  # ! default: old ANN: normal Gis !
    # ! 1 - log Gis for ANN pot. !

    iPOT_func_type = 0  # ! 0 - EAM !
    # ! 1 - ADP !

    inv_calls = 0  # ! for TEST only !

    rindr, dr_min, dr_recp = 0.0, 0.0, 0.0
    drho_min, rho_max, drho_recp = 0.0, 0.0, 0.0
    r_cut_off, r2_cut_off, rin_max, avr_mass = 0.0, 0.0, 0.0, 0.0
    r_cut_short, r2_cut_short, r_max_pot = 0.0, 0.0, 0.0
    elem_radius = [0.0] * 112

    ielement = []
    natoms_of_type = []
    iZ_elem_in_com = []

    amass = [0.0]  # ! double precision !
    sum_mass = [0.0]  # ! double precision !
    gram_mol = [0.0]  # ! double precision !
    chem_pot = [0.0]  # ! double precision !
    Am_of_type = [0.0]  # ! double precision !
    pTemp = [0.0]  # ! double precision !
    E_kin = [0.0]  # ! double precision !

    r_pair_cut = [0.0]  # ! double precision !
    r2_pair_cut = [0.0]  # ! double precision !

    # !
    # ! -------------------------------------------------------------------
    # !   Changes a string to upper case
    # ! -------------------------------------------------------------------
    # !

    def conv_to_low(self, strIn, strOut, len_str):

        len_str = len(strIn.lstrip())

        # ! print('conv_to_low: len(', strIn, ')=', len_trim(strIn))

        for i in range(1, len_str + 1):
            j = ord(strIn[i])
            if ord('A') <= j <= ord('Z'):
                strOut[i] = chr(ord(strIn[i]) + 32)
            else:
                strOut[i] = strIn[i]

        # ! End of conv_to_low !

    # !
    # !--------------------------------------------------------------------
    # !

    def init_elements(self):

        self.chem_symb[0] = ''
        self.elem_radius[0] = 0.0  # ! Ang !
        self.chem_symb[1] = 'H'
        self.elem_radius[1] = 0.53
        self.chem_symb[2] = 'He'
        self.elem_radius[2] = 0.31
        self.chem_symb[3] = 'Li'
        self.elem_radius[3] = 1.45
        self.chem_symb[4] = 'Be'
        self.elem_radius[4] = 1.05
        self.chem_symb[5] = 'B'
        self.elem_radius[5] = 0.85
        self.chem_symb[6] = 'C'
        self.elem_radius[6] = 0.70
        self.chem_symb[7] = 'N'
        self.elem_radius[7] = 0.65
        self.chem_symb[8] = 'O'
        self.elem_radius[8] = 0.60
        self.chem_symb[9] = 'F'
        self.elem_radius[9] = 0.50
        self.chem_symb[10] = 'Ne'
        self.elem_radius[10] = 0.38
        self.chem_symb[11] = 'Na'
        self.elem_radius[11] = 1.80
        self.chem_symb[12] = 'Mg'
        self.elem_radius[12] = 1.50
        self.chem_symb[13] = 'Al'
        self.elem_radius[13] = 1.18
        self.chem_symb[14] = 'Si'
        self.elem_radius[14] = 1.10
        self.chem_symb[15] = 'P'
        self.elem_radius[15] = 1.00
        self.chem_symb[16] = 'S'
        self.elem_radius[16] = 1.00
        self.chem_symb[17] = 'Cl'
        self.elem_radius[17] = 1.00
        self.chem_symb[18] = 'Ar'
        self.elem_radius[18] = 0.71
        self.chem_symb[19] = 'K'
        self.elem_radius[19] = 2.20
        self.chem_symb[20] = 'Ca'
        self.elem_radius[20] = 1.80
        self.chem_symb[21] = 'Sc'
        self.elem_radius[21] = 1.60
        self.chem_symb[22] = 'Ti'
        self.elem_radius[22] = 1.40
        self.chem_symb[23] = 'V'
        self.elem_radius[23] = 1.35
        self.chem_symb[24] = 'Cr'
        self.elem_radius[24] = 1.40
        self.chem_symb[25] = 'Mn'
        self.elem_radius[25] = 1.40
        self.chem_symb[26] = 'Fe'
        self.elem_radius[26] = 1.40
        self.chem_symb[27] = 'Co'
        self.elem_radius[27] = 1.35
        self.chem_symb[28] = 'Ni'
        self.elem_radius[28] = 1.35
        self.chem_symb[29] = 'Cu'
        self.elem_radius[29] = 1.35
        self.chem_symb[30] = 'Zn'
        self.elem_radius[30] = 1.35
        self.chem_symb[31] = 'Ga'
        self.elem_radius[31] = 1.30
        self.chem_symb[32] = 'Ge'
        self.elem_radius[32] = 1.25
        self.chem_symb[33] = 'As'
        self.elem_radius[33] = 1.15
        self.chem_symb[34] = 'Se'
        self.elem_radius[34] = 1.15
        self.chem_symb[35] = 'Br'
        self.elem_radius[35] = 1.15
        self.chem_symb[36] = 'Kr'
        self.elem_radius[36] = 0.88
        self.chem_symb[37] = 'Rb'
        self.elem_radius[37] = 2.35
        self.chem_symb[38] = 'Sr'
        self.elem_radius[38] = 2.00
        self.chem_symb[39] = 'Y'
        self.elem_radius[39] = 1.85
        self.chem_symb[40] = 'Zr'
        self.elem_radius[40] = 1.55
        self.chem_symb[41] = 'Nb'
        self.elem_radius[41] = 1.45
        self.chem_symb[42] = 'Mo'
        self.elem_radius[42] = 1.45
        self.chem_symb[43] = 'Tc'
        self.elem_radius[43] = 1.35
        self.chem_symb[44] = 'Ru'
        self.elem_radius[44] = 1.30
        self.chem_symb[45] = 'Rh'
        self.elem_radius[45] = 1.35
        self.chem_symb[46] = 'Pd'
        self.elem_radius[46] = 1.40
        self.chem_symb[47] = 'Ag'
        self.elem_radius[47] = 1.60
        self.chem_symb[48] = 'Cd'
        self.elem_radius[48] = 1.55
        self.chem_symb[49] = 'In'
        self.elem_radius[49] = 1.55
        self.chem_symb[50] = 'Sn'
        self.elem_radius[50] = 1.45
        self.chem_symb[51] = 'Sb'
        self.elem_radius[51] = 1.45
        self.chem_symb[52] = 'Te'
        self.elem_radius[52] = 1.40
        self.chem_symb[53] = 'I'
        self.elem_radius[53] = 1.40
        self.chem_symb[54] = 'Xe'
        self.elem_radius[54] = 1.08
        self.chem_symb[55] = 'Cs'
        self.elem_radius[55] = 2.60
        self.chem_symb[56] = 'Ba'
        self.elem_radius[56] = 2.15
        self.chem_symb[57] = 'La'
        self.elem_radius[57] = 1.95
        self.chem_symb[58] = 'Ce'
        self.elem_radius[58] = 1.85
        self.chem_symb[59] = 'Pr'
        self.elem_radius[59] = 1.85
        self.chem_symb[60] = 'Nd'
        self.elem_radius[60] = 1.85
        self.chem_symb[61] = 'Pm'
        self.elem_radius[61] = 1.85
        self.chem_symb[62] = 'Sm'
        self.elem_radius[62] = 1.85
        self.chem_symb[63] = 'Eu'
        self.elem_radius[63] = 1.85
        self.chem_symb[64] = 'Gd'
        self.elem_radius[64] = 1.80
        self.chem_symb[65] = 'Tb'
        self.elem_radius[65] = 1.75
        self.chem_symb[66] = 'Dy'
        self.elem_radius[66] = 1.75
        self.chem_symb[67] = 'Ho'
        self.elem_radius[67] = 1.75
        self.chem_symb[68] = 'Er'
        self.elem_radius[68] = 1.75
        self.chem_symb[69] = 'Tm'
        self.elem_radius[69] = 1.75
        self.chem_symb[70] = 'Yb'
        self.elem_radius[70] = 1.75
        self.chem_symb[71] = 'Lu'
        self.elem_radius[71] = 1.75
        self.chem_symb[72] = 'Hf'
        self.elem_radius[72] = 1.55
        self.chem_symb[73] = 'Ta'
        self.elem_radius[73] = 1.45
        self.chem_symb[74] = 'W'
        self.elem_radius[74] = 1.35
        self.chem_symb[75] = 'Re'
        self.elem_radius[75] = 1.35
        self.chem_symb[76] = 'Os'
        self.elem_radius[76] = 1.30
        self.chem_symb[77] = 'Ir'
        self.elem_radius[77] = 1.35
        self.chem_symb[78] = 'Pt'
        self.elem_radius[78] = 1.35
        self.chem_symb[79] = 'Au'
        self.elem_radius[79] = 1.35
        self.chem_symb[80] = 'Hg'
        self.elem_radius[80] = 1.50
        self.chem_symb[81] = 'Tl'
        self.elem_radius[81] = 1.90
        self.chem_symb[82] = 'Pb'
        self.elem_radius[82] = 1.80
        self.chem_symb[83] = 'Bi'
        self.elem_radius[83] = 1.60
        self.chem_symb[84] = 'Po'
        self.elem_radius[84] = 1.90
        self.chem_symb[85] = 'At'
        self.elem_radius[85] = 1.27
        self.chem_symb[86] = 'Rn'
        self.elem_radius[86] = 1.20
        self.chem_symb[87] = 'Fr'
        self.elem_radius[87] = 1.94
        self.chem_symb[88] = 'Ra'
        self.elem_radius[88] = 2.15
        self.chem_symb[89] = 'Ac'
        self.elem_radius[89] = 1.95
        self.chem_symb[90] = 'Th'
        self.elem_radius[90] = 1.80
        self.chem_symb[91] = 'Pa'
        self.elem_radius[91] = 1.80
        self.chem_symb[92] = 'U'
        self.elem_radius[92] = 1.75
        self.chem_symb[93] = 'Np'
        self.elem_radius[93] = 1.75
        self.chem_symb[94] = 'Pu'
        self.elem_radius[94] = 1.75
        self.chem_symb[95] = 'Am'
        self.elem_radius[95] = 1.75
        self.chem_symb[96] = 'Cm'
        self.elem_radius[96] = 1.75
        self.chem_symb[97] = 'Bk'
        self.elem_radius[97] = 1.75
        self.chem_symb[98] = 'Cf'
        self.elem_radius[98] = 1.75
        self.chem_symb[99] = 'Es'
        self.elem_radius[99] = 1.75
        self.chem_symb[100] = 'Fm'
        self.elem_radius[100] = 1.75
        self.chem_symb[101] = 'Md'
        self.elem_radius[101] = 1.75
        self.chem_symb[102] = 'No'
        self.elem_radius[102] = 1.75
        self.chem_symb[103] = 'Lr'
        self.elem_radius[103] = 1.75
        self.chem_symb[104] = 'Rf'
        self.elem_radius[104] = 1.75
        self.chem_symb[105] = 'Db'
        self.elem_radius[105] = 1.75
        self.chem_symb[106] = 'Sg'
        self.elem_radius[106] = 1.75
        self.chem_symb[107] = 'Bh'
        self.elem_radius[107] = 1.75
        self.chem_symb[108] = 'Hs'
        self.elem_radius[108] = 1.75
        self.chem_symb[109] = 'Mt'
        self.elem_radius[109] = 1.75
        self.chem_symb[110] = 'Ds'
        self.elem_radius[110] = 1.75
        self.chem_symb[111] = 'Rg'
        self.elem_radius[111] = 1.75
        self.chem_symb[112] = 'Cn'
        self.elem_radius[112] = 1.75

        # ! End of init_elements !

    # !
    # !--------------------------------------------------------------------
    # !

    def numb_elem_Z(self, chem):

        chem_low = ""
        chem_symb_low = ""

        len_chem = len(chem.lstrip())

        numZ = 0
        self.conv_to_low(chem, chem_low, len_chem)
        for iZ in range(1, 112 + 1):
            len_symb = len(self.chem_symb[iZ].lstrip())
            self.conv_to_low(self.chem_symb[iZ], chem_symb_low, len_symb)
            if len_chem == len_symb:
                if chem_low == chem_symb_low:
                    numZ = iZ
                    break

        # ! write(6, *) 'numb_elem_Z(', chem, ')=', numZ

        return numZ
        # ! End of numb_elem_Z !

    # !
    # !---------------------------------------------------------------------
    # ! *** Gets current chem.composition ***
    # ! Includes all atoms that DO NOT have chemical constrain !
    # !---------------------------------------------------------------------
    # !

    def get_chem(self):

        # ! *** Collect atom types...
        natoms_of_type = [0]

        for n in range(1, sim_box.natoms + 1):
            ntp = atoms.ntype[n]
            natoms_of_type[ntp] += 1

        return
        # ! End of get_chem !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def alloc_pot_types(self, ierror):

        ialloc = [0] * 14

        self.elem_symb = [0] * atoms.iatom_types

        self.ielement = [0] * atoms.iatom_types
        self.natoms_of_type = [0] * atoms.iatom_types
        self.iZ_elem_in_com = [0] * atoms.iatom_types

        self.amass = [0] * atoms.iatom_types
        self.sum_mass = [0] * atoms.iatom_types
        self.gram_mol = [0] * atoms.iatom_types

        self.Am_of_type = [0] * atoms.iatom_types
        self.Elem_in_com = [0] * atoms.iatom_types

        self.r_pair_cut = [0] * atoms.ipair_types
        self.r2_pair_cut = [0] * atoms.ipair_types

        self.E_kin = [0] * atoms.iatom_types
        self.pTemp = [0] * atoms.iatom_types
        self.filename = [""] * 1

        ierror = 0
        for i in range(1, 14 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of alloc_pot_types !

    # !
    # ! ---------------------------------------------------------------------
    # !

    def deall_pot_types(self, ierror):

        ialloc = [0] * 14

        if self.elem_symb:
            self.elem_symb.clear()
        if self.filename:
            self.filename.clear()

        if self.ielement:
            self.ielement.clear()
        if self.natoms_of_type:
            self.natoms_of_type.clear()
        if self.iZ_elem_in_com:
            self.iZ_elem_in_com.clear()

        if self.amass:
            self.amass.clear()
        if self.sum_mass:
            self.sum_mass.clear()
        if self.gram_mol:
            self.gram_mol.clear()
        if self.Am_of_type:
            self.Am_of_type.clear()
        if self.Elem_in_com:
            self.Elem_in_com.clear()

        if self.r_pair_cut:
            self.r_pair_cut.clear()
        if self.r2_pair_cut:
            self.r2_pair_cut.clear()

        if self.E_kin:
            self.E_kin.clear()
        if self.pTemp:
            self.pTemp.clear()

        ierror = 0
        for i in range(1, 14 + 1):
            ierror += ialloc[i]

        return ierror
        # ! End of deall_pot_types !

    # ! End of pot_module !

#
# ------------------------------------------------------------------
#

class constants:
    # *** Set some physics constants as parameters here ***
    Boltz_Kb = 0.000086173324
    comp = 0.0001
    erg = 0.000000000001602189
    hp = 0.0041354
    cavog = 602.2045
    evdyne = 1.602189
    evkbar = 0.001602189
    evjoul = 16.02189
    pi = 4 * math.atan(1.0)
    pi2 = pi / 2.0
    eVAng2GPa = evdyne * 100.0
    atu = 0.000103642696
    tunit = math.sqrt(atu)

    # ! End of constants !
