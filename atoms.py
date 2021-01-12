import sys
import operator
import numpy as np
import random
import torch
import math
import sim_box

nbr_tot, max_nbrs = 0, 0

iatom_types, ipair_types, icomp_high, icomp_low = 0, 0, 0, 0

at_vol, at_vol_inp, sys_vol = 0.0, 0.0, 0.0

sum_dcm = [0.0] * (3 + 1)
Sys_dcmR = [0.0] * (3 + 1)
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

def alloc_atoms_sys():

    global ident,ntype,rx,ry,rz,sx,sy,sz,nbr_list,frr,Ep_of 

    ialloc = [0] * (12 + 1)

    ident = [0] * (sim_box.natoms_alloc + 1)
    ntype = [0] * (sim_box.natoms_alloc + 1)

    rx = [0] * (sim_box.natoms_alloc + 1)
    ry = [0] * (sim_box.natoms_alloc + 1)
    rz = [0] * (sim_box.natoms_alloc + 1)

    sx = [0] * (sim_box.natoms_alloc + 1)
    sy = [0] * (sim_box.natoms_alloc + 1)
    sz = [0] * (sim_box.natoms_alloc + 1)

    nbr_list = [[0] * (sim_box.natoms_alloc + 1) for i in range(sim_box.nbrs_per_atom + 1)]

    frr = [[0] * (sim_box.natoms_alloc + 1) for i in range(3 + 1)]
    Ep_of = [0] * (sim_box.natoms_alloc + 1)

    ierror = 0
    for i in range(1, 12 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of alloc_atoms_sys !

# !
# ! ---------------------------------------------------------------------
# !

def deall_atoms_sys():

    global ident,ntype,rx,ry,rz,sx,sy,sz,nbr_list,frr,Ep_of 

    ialloc = [0] * (12 + 1)

    if ident:
        ident.clear()
    if ntype:
        ntype.clear()

    if rx:
        rx.clear()
    if ry:
        ry.clear()
    if rz:
        rz.clear()

    if sx:
        sx.clear()
    if sy:
        sy.clear()
    if sz:
        sz.clear()

    if nbr_list:
        nbr_list.clear()

    if frr:
        frr.clear()
    if Ep_of:
        Ep_of.clear()

    ierror = 0
    for i in range(1, 12 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of deall_atoms_sys !

# !
# ! ---------------------------------------------------------------------
# !

def alloc_atoms_MD():

    global x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,sumPx,sumPy,sumPz,Tscale
    global iatom_types

    ialloc = [0] * (20 + 1)

    x1 = [0.0] * (sim_box.natoms_alloc + 1)
    y1 = [0.0] * (sim_box.natoms_alloc + 1)
    z1 = [0.0] * (sim_box.natoms_alloc + 1)

    x2 = [0] * (sim_box.natoms_alloc + 1)
    y2 = [0] * (sim_box.natoms_alloc + 1)
    z2 = [0] * (sim_box.natoms_alloc + 1)

    x3 = [0] * (sim_box.natoms_alloc + 1)
    y3 = [0] * (sim_box.natoms_alloc + 1)
    z3 = [0] * (sim_box.natoms_alloc + 1)

    x4 = [0] * (sim_box.natoms_alloc + 1)
    y4 = [0] * (sim_box.natoms_alloc + 1)
    z4 = [0] * (sim_box.natoms_alloc + 1)

    x5 = [0] * (sim_box.natoms_alloc + 1)
    y5 = [0] * (sim_box.natoms_alloc + 1)
    z5 = [0] * (sim_box.natoms_alloc + 1)

    sumPx = [0.0] * (iatom_types + 1)
    sumPy = [0.0] * (iatom_types + 1)
    sumPz = [0.0] * (iatom_types + 1)
    Tscale = [0] * (iatom_types + 1)

    ierror = 0
    for i in range(1, 20 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of alloc_atoms_MD !

# !
# ! ---------------------------------------------------------------------
# !

def deall_atoms_MD():

    global x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,sumPx,sumPy,sumPz,Tscale

    ialloc = [0] * (20 + 1)

    if x1:
        x1.clear()
    if y1:
        y1.clear()
    if z1:
        z1.clear()

    if x2:
        x2.clear()
    if y2:
        y2.clear()
    if z2:
        z2.clear()

    if x3:
        x3.clear()
    if y3:
        y3.clear()
    if z3:
        z3.clear()

    if x4:
        x4.clear()
    if y4:
        y4.clear()
    if z4:
        z4.clear()

    if x5:
        x5.clear()
    if y5:
        y5.clear()
    if z5:
        z5.clear()

    if sumPx:
        sumPx.clear()
    if sumPy:
        sumPy.clear()
    if sumPz:
        sumPz.clear()
    if Tscale:
        Tscale.clear()

    ierror = 0
    for i in range(1, 20 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of deall_atoms_MD !

# ! End of atoms !
