#
# ------------------------------------------------------------------
# 01-12-2021
#
# Miolecular Dynamics atoms functions for aladyn.f code.
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
# Miolecular Dynamics atoms functions for aladyn.f code
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

import sim_box


#
# ------------------------------------------------------------------
#

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

# ---------------------------------------------------------------------
#
#      END FILE  ! atoms !
#
# =====================================================================
