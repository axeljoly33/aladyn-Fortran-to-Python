import sys
import operator
import numpy as np
import random
import torch
import math

import node_conf
import PROGRAM_END


# *** Set some working parameters here ***

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
Line_Executed = [0] * (Max_Lines_INI+1)
# ! = 0: line has not been executed !
# ! = 1: line has been executed !
n_of_all_moved = 0  # ! all MC moved atoms up to now !

ncell_of_atom = []

n_of_moved_atom = []

BkT = 0.0  # ! = 1.0d0 / (Boltz_Kb * T_sys) !
real_time, start_time, start_run_time = 0.0, 0.0, 0.0
T_sys, T_set, Ek_sys = 0.0, 0.0, 0.0

h = [[0.0] * (3 + 1) for i1 in range(3 + 1)]
hi = [[0.0] * (3 + 1) for i2 in range(3 + 1)]

# ! System shape matrices !
h1 = [[0.0] * (3 + 1) for i3 in range(3 + 1)]
h2 = [[0.0] * (3 + 1) for i4 in range(3 + 1)]
h3 = [[0.0] * (3 + 1) for i5 in range(3 + 1)]
h4 = [[0.0] * (3 + 1) for i6 in range(3 + 1)]
h5 = [[0.0] * (3 + 1) for i7 in range(3 + 1)]

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

U = [0.0] * (97+1)
C, CD, CM = 0.0, 0.0, 0.0
Us = [0.0] * (97+1)
Cs, CDs, CMs = 0.0, 0.0, 0.0
I97, J97, I97s, J97s = 0, 0, 0, 0

# !
# ! ********************************************************
# !                                                        $
# !  Calculates the argument of a complex number x + iy    $
# !                                                        $
# ! ********************************************************
# !

def argument( x, y, r):

    # x, y, r = 0.0, 0.0, 0.0
    xr, yr, phi = 0.0, 0.0, 0.0
    PI = 3.141592654

    r = math.sqrt(pow(x, 2) + pow(y, 2))
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

def ranmar(LEN):

    global U,I97,J97,C,buf,CD,CM

    UNI = 0.0

    for IVEC in range(1, LEN + 1):
        UNI = U[I97] - U[J97]
        if UNI < 0.0:
            UNI = UNI + 1.0
        U[I97] = UNI
        I97 = I97 - 1
        if I97 == 0:
            I97 = 97
        J97 = J97 - 1
        if J97 == 0:
            J97 = 97
        C = C - CD
        if C < 0.0:
            C = C + CM
        UNI = UNI - C
        if UNI < 0.0:
            UNI = UNI + 1.0
        buf[IVEC] = UNI

    # ! End of ranmar !

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

def rmarin(IJKL):

    global U,C,CD,CM,I97,J97

    T = 0.0

    IJ = int(IJKL / 30082)
    KL = IJKL - IJ * 30082
    I = int(((IJ / 177) % 177) + 2)
    J = (IJ % 177) + 2
    K = int(((KL / 169) % 178) + 1)
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
        U[II] = S

    C = 362436.0 / 16777216.0
    CD = 7654321.0 / 16777216.0
    CM = 16777213.0 / 16777216.0
    I97 = 97
    J97 = 33

    # ! End of rmarin !

# !
# !----------------------------------------------------------------------
# !

def matinv(a, b):

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

    return c

# !
# ! ---------------------------------------------------------------------
# !

def alloc_nodes(nodes):

    global nbuf, nbufa

    ialloc = [0] * (2 + 1)

    # ! if (mynod.eq.0) write(6, * )'alloc_nodes(', nodes, ')'

    nbuf = [0] * (nodes + 1)
    nbufa = [0] * (nodes + 1)

    ierror = 0
    for i in range(1, 2 + 1):
        ierror = ierror + ialloc[i]

    return ierror
    # ! End of alloc_nodes !

# !
# ! ---------------------------------------------------------------------
# !

def alloc_buffers():

    global buf,bufa,bufb,bufc,ibufY1,ibufY2,ibufZ1,ibufZ2
    global kbuf,kbufa,kbufb,kbufc,ncell_of_atom,nbufsize,natoms_alloc

    ialloc = [0] * (14 + 1)

    # ! if (mynod.eq.0) write(6, * )'alloc_buffers: nbufsize=', nbufsize

    buf = [0.0] * (nbufsize + 1)
    bufa = [0.0] * (nbufsize + 1)
    bufb = [0.0] * (nbufsize + 1)
    bufc = [0.0] * (nbufsize + 1)

    ibufY1 = [0] * (natoms_alloc + 1)
    ibufY2 = [0] * (natoms_alloc + 1)
    ibufZ1 = [0] * (natoms_alloc + 1)
    ibufZ2 = [0] * (natoms_alloc + 1)

    kbuf = [0] * (nbufsize + 1)
    kbufa = [0] * (nbufsize + 1)
    kbufb = [0] * (nbufsize + 1)
    kbufc = [0] * (nbufsize + 1)
    ncell_of_atom = [0] * (natoms_alloc + 1)

    ierror = 0
    for i in range(1, 14 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of alloc_buffers !

# !
# ! ---------------------------------------------------------------------
# !

def deall_buffers():

    global buf,bufa,bufb,bufc,ibufY1,ibufY2,ibufZ1,ibufZ2
    global kbuf,kbufa,kbufb,kbufc,ncell_of_atom

    ialloc = [0] * (14 + 1)

    if buf:
        buf.clear()
    if bufa:
        bufa.clear()
    if bufb:
        bufb.clear()
    if bufc:
        bufc.clear()

    if ibufY1:
        ibufY1.clear()
    if ibufY2:
        ibufY2.clear()
    if ibufZ1:
        ibufZ1.clear()
    if ibufZ2:
        ibufZ2.clear()

    if kbuf:
        kbuf.clear()
    if kbufa:
        kbufa.clear()
    if kbufb:
        kbufb.clear()
    if kbufc:
        kbufc.clear()
    if ncell_of_atom:
        ncell_of_atom.clear()

    ierror = 0
    for i in range(1, 14 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of deall_buffers !

# !
# ! ---------------------------------------------------------------------
# !

def alloc_cells():

    global id_of_cell,natoms_in_cell,n_in_cell,n_of_moved_atom,ncell_per_node,natoms_per_cell

    ialloc = [0] * (4 + 1)

    id_of_cell = [0] * (ncell_per_node + 1)
    natoms_in_cell = [0] * (ncell_per_node + 1)

    n_in_cell = [[0] * (ncell_per_node + 1) for i in range(natoms_per_cell + 1)]
    n_of_moved_atom = [0] * (ncell_per_node + 1)

    ierror = 0
    for i in range(1, 4 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of alloc_cells !

# !
# ! ---------------------------------------------------------------------
# !

def deall_cells():

    global id_of_cell,natoms_in_cell,n_in_cell,n_of_moved_atom

    ialloc = [0] * (4 + 1)

    if id_of_cell:
        id_of_cell.clear()
    if natoms_in_cell:
        natoms_in_cell.clear()
    if n_in_cell:
        n_in_cell.clear()
    if n_of_moved_atom:
        n_of_moved_atom.clear()

    ierror = 0
    for i in range(1, 4 + 1):
        ierror = ierror + ialloc[i]

    return ierror
    # ! End of deall_cells !

# !
# ! -------------------------------------------------------------------
# !

def error_check(ierror, message):

    global mynod

    if ierror != 0:
        if mynod == 0:
            print(message)
        print("The program halted on an error detected by the original Aladyn program")
        PROGRAM_END.PROGRAM_END(1)

    return
    # ! End of error_check !

# ! End of sim_box !

#
# ------------------------------------------------------------------
#
