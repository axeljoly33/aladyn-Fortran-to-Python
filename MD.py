#
# ------------------------------------------------------------------
# 12-10-2020
#
# Miolecular Dynamics Module Unit for aladyn.f code.
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
# Miolecular Dynamics Module Unit for aladyn.f code
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

import atoms
import sim_box
import constants
import pot_module
import node_conf
import group_conf

import aladyn_IO
import aladyn_ANN
import aladyn


#
# ------------------------------------------------------------------
#

f02 = 3.0 / 20.0
f02viscous = 3.0 / 16.0
f12 = 251.0 / 360.0
f32 = 11.0 / 18.0
f42 = 1.0 / 6.0
f52 = 1.0 / 60.0
f02_wall = 3.0 / 20.0
f02_atom = 3.0 / 20.0

#
# -----------------------------------------------------------------------
# *** this is the SLOW atoms corrector step ** *
# -----------------------------------------------------------------------
#

def correct_atoms(ndof_fl):

    # integer, intent( in):: ndof_fl

    global f02_atom,f12,f32,f42,f52,f02_wall,f02_atom,f02viscous

    n, ntp = 0, 0
    dt, dtsqh, dtsqhM, sai = 0.0, 0.0, 0.0, 0.0
    am, fr_ntp = 0.0, 0.0
    x1i, xmp, afrx, Vxt, fx = 0.0, 0.0, 0.0, 0.0, 0.0
    y1i, ymp, afry, Vyt, fy = 0.0, 0.0, 0.0, 0.0, 0.0
    z1i, zmp, afrz, Vzt, fz = 0.0, 0.0, 0.0, 0.0, 0.0
    dsx = [0.0] * 3
    sum_dcm0 = [0.0] * 3

    dt = atoms.dt_step
    dtsqh = 0.5 * math.pow(dt, 2)

    for n in range(1, atoms.iatom_types + 1):
        atoms.sumPx[n] = 0.0
        atoms.sumPy[n] = 0.0
        atoms.sumPz[n] = 0.0
        pot_module.E_kin[n] = 0.0

    for n in range(1, sim_box.natoms + 1):

        ntp = atoms.ntype[n]
        am = pot_module.amass[ntp]
        dtsqhM = dtsqh / am  # ! = 0.5 * dt * dt / m with dt=dt_step !
        # ! a = - 1 / 2 * dt ^ 2 * f / m; dt = dt_step !

        fxsn = sim_box.hi11 * atoms.frr[1][n] + sim_box.hi12 * \
               atoms.frr[2][n] + sim_box.hi13 * atoms.frr[3][n]
        fysn = sim_box.hi22 * atoms.frr[2][n] + sim_box.hi23 * \
               atoms.frr[3][n]  # ! S - space !
        fzsn = sim_box.hi33 * atoms.frr[3][n]

        x1i = atoms.x1[n]
        y1i = atoms.y1[n]
        z1i = atoms.z1[n]

        xmp = atoms.x2[n] - dtsqhM * fxsn
        ymp = atoms.y2[n] - dtsqhM * fysn
        zmp = atoms.z2[n] - dtsqhM * fzsn

        # *** Nose - Hoover Thermostat,

        sai = atoms.sx[n] - xmp * f02_atom
        x1i = x1i - xmp * f12
        atoms.sx[n] = sai
        atoms.x1[n] = x1i
        atoms.x2[n] = atoms.x2[n] - xmp
        atoms.x3[n] = atoms.x3[n] - xmp * f32
        atoms.x4[n] = atoms.x4[n] - xmp * f42
        atoms.x5[n] = atoms.x5[n] - xmp * f52

        sai = atoms.sy[n] - ymp * f02_atom
        y1i = y1i - ymp * f12
        atoms.sy[n] = sai
        atoms.y1[n] = y1i
        atoms.y2[n] = atoms.y2[n] - ymp
        atoms.y3[n] = atoms.y3[n] - ymp * f32
        atoms.y4[n] = atoms.y4[n] - ymp * f42
        atoms.y5[n] = atoms.y5[n] - ymp * f52

        sai = atoms.sz[n] - zmp * f02_atom
        z1i = z1i - zmp * f12
        atoms.sz[n] = sai
        atoms.z1[n] = z1i
        atoms.z2[n] = atoms.z2[n] - zmp
        atoms.z3[n] = atoms.z3[n] - zmp * f32
        atoms.z4[n] = atoms.z4[n] - zmp * f42
        atoms.z5[n] = atoms.z5[n] - zmp * f52

        Vxt = sim_box.h11 * x1i + sim_box.h12 * y1i + sim_box.h13 * z1i
        Vyt = sim_box.h22 * y1i + sim_box.h23 * z1i
        Vzt = sim_box.h33 * z1i

        atoms.sumPx[ntp] = atoms.sumPx[ntp] + am * Vxt
        atoms.sumPy[ntp] = atoms.sumPy[ntp] + am * Vyt
        atoms.sumPz[ntp] = atoms.sumPz[ntp] + am * Vzt

        Ek_atm = am * (math.pow(Vxt, 2) + math.pow(Vyt, 2) + math.pow(Vzt, 2))
        pot_module.E_kin[ntp] = pot_module.E_kin[ntp] + Ek_atm  # ! mv ^ 2 !

        # ! do n = 1, natoms !

    return
    # ! End of correct_atoms !

#
# -----------------------------------------------------------------------
# predicts pos., vel., and higher deriv.
# Fast atoms and their neighbors are using the smallest basic dt,
# Slow atoms that do not neighbor a fast atom use the long time step.
# -----------------------------------------------------------------------
#

def predict_atoms(ndof_fl):

    # integer, intent( in):: ndof_fl

    a1i, a2i, a3i, a4i, a5i, a24, a45, a2345 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dsx = [0.0] * 3
    i = 0
    kdof = 0

    #
    # *** start predictor step of integration scheme ******************
    # *** all x, y, z coordinates are s - scaled coordinates
    # and are scale invariant(do not change with length units).
    #

    h11 = sim_box.h[1][1]
    h12 = sim_box.h[1][2]
    h13 = sim_box.h[1][3]
    h22 = sim_box.h[2][2]
    h23 = sim_box.h[2][3]
    h33 = sim_box.h[3][3]
    hi11 = sim_box.hi[1][1]
    hi12 = sim_box.hi[1][2]
    hi13 = sim_box.hi[1][3]
    hi22 = sim_box.hi[2][2]
    hi23 = sim_box.hi[2][3]
    hi33 = sim_box.hi[3][3]

    for i in range(1, sim_box.natoms + 1):
        a1i = atoms.x1[i]
        a2i = atoms.x2[i]
        a3i = atoms.x3[i]
        a4i = atoms.x4[i]
        a5i = atoms.x5[i]
        atoms.sx[i] = atoms.sx[i] + a1i + a2i + a3i + a4i + a5i
        a24 = 2.0 * a4i
        a45 = a24 + 5.0 * a5i
        a2345 = a2i + 3.0 * a3i + a45 + a24
        atoms.x1[i] = a1i + a2i + a2345
        atoms.x2[i] = a2345 + a45
        atoms.x3[i] = a3i + 2.0 * a45
        atoms.x4[i] = a45 - a4i

        a1i = atoms.y1[i]
        a2i = atoms.y2[i]
        a3i = atoms.y3[i]
        a4i = atoms.y4[i]
        a5i = atoms.y5[i]
        atoms.sy[i] = atoms.sy[i] + a1i + a2i + a3i + a4i + a5i
        a24 = 2.0 * a4i
        a45 = a24 + 5.0 * a5i
        a2345 = a2i + 3.0 * a3i + a45 + a24
        atoms.y1[i] = a1i + a2i + a2345
        atoms.y2[i] = a2345 + a45
        atoms.y3[i] = a3i + 2.0 * a45
        atoms.y4[i] = a45 - a4i

        a1i = atoms.z1[i]
        a2i = atoms.z2[i]
        a3i = atoms.z3[i]
        a4i = atoms.z4[i]
        a5i = atoms.z5[i]
        atoms.sz[i] = atoms.sz[i] + a1i + a2i + a3i + a4i + a5i
        a24 = 2.0 * a4i
        a45 = a24 + 5.0 * a5i
        a2345 = a2i + 3.0 * a3i + a45 + a24
        atoms.z1[i] = a1i + a2i + a2345
        atoms.z2[i] = a2345 + a45
        atoms.z3[i] = a3i + 2.0 * a45
        atoms.z4[i] = a45 - a4i
    # ! do i = 1, natoms !

    return
    # ! End of predict_atoms !

#
# -----------------------------------------------------------------------
# Collects Px, y, z and Ek from all nodes after correct_(slow, fast)_atoms
# -----------------------------------------------------------------------
#

def T_broadcast():

    dt = atoms.dt_step  # ! uses the basic time step !
    dtsq = math.pow(dt, 2)
    Ttemp = 1.0 / (3.0 * constants.Boltz_Kb * dtsq)  # ! 1 / (3kB.T) !

    Q_heat = atoms.Q_heat + atoms.A_fr / max(1, sim_box.natoms)
    A_fr = 0.0

    for ntp in range(1, atoms.iatom_types + 1):
        na = pot_module.natoms_of_type[ntp]  # ! do not use max(1, natoms_of_type()) !
        if na > 0:
            atoms.sumPx[ntp] = atoms.sumPx[ntp] / na
            atoms.sumPy[ntp] = atoms.sumPy[ntp] / na
            atoms.sumPz[ntp] = atoms.sumPz[ntp] / na
            pot_module.E_kin[ntp] = pot_module.E_kin[ntp] / na  # ! mv ^ 2 !
            if na > 1:
                pot_module.pTemp[ntp] = Ttemp * \
                                                    (pot_module.E_kin[ntp] -
                                                     (math.pow(atoms.sumPx[ntp], 2) +
                                                      math.pow(atoms.sumPy[ntp], 2) +
                                                      math.pow(atoms.sumPz[ntp], 2)) /
                                                     pot_module.sum_mass[ntp])
            else:
                pot_module.pTemp[ntp] = Ttemp * pot_module.E_kin[ntp]
                # ! (2 / 3kT) * E_k = mv ^ 2 / 3kT !
        #! 2Ek - P ^ 2 / M = 2(Ek - P ^ 2 / 2M) !

    # ! T = (2Ek - P ^ 2 / M) / 3Kb = 2(Ek - P ^ 2 / 2M) / 3Kb !
    # ! Ek, P, and M are the TOTAL kin.energy, momentum, and mass !

    # do ntp = 1, iatom_types
    # write(1000 + mynod, 10) ntp, pTemp(ntp), sum_mass(ntp),
    # E_kin(ntp) / (2.0d0 * dt ** 2), natoms_of_type(ntp)
    # enddo
    # 10 format('TB: pTemp(', i2, ')=', f7 .2, ' sum_mass=', f12 .8,
    # ' E_kin=', f12 .8, ' na_of_type=', i5)

    return
    # ! End of T_broadcast !

#
# ---------------------------------------------------------------------
# Uses x1(n), y1(n), and z1(n) in S - space only to calculate
# sumPxyz(n), pTemp(ntp), E_kin(ntp), T_sys and Ek_sys
# ---------------------------------------------------------------------
#

def get_T():

    dt = atoms.dt_step
    tfc = 2.0 / (3.0 * constants.Boltz_Kb)

    for i in range(1, atoms.iatom_types + 1):
        pot_module.Am_of_type[i] = 0.0
        atoms.sumPx[i] = 0.0
        atoms.sumPy[i] = 0.0
        atoms.sumPz[i] = 0.0
        pot_module.E_kin[i] = 0.0

    for i in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[i]
        Am = pot_module.amass[ntp]
        pot_module.Am_of_type[ntp] = pot_module.Am_of_type[ntp] + Am
        Vx = sim_box.h[1][1] * atoms.x1[i] + \
             sim_box.h[1][2] * atoms.y1[i] + sim_box.h[1][3] * \
             atoms.z1[i]
        Vy = sim_box.h[2][2] * atoms.y1[i] + \
             sim_box.h[2][3] * atoms.z1[i]  # ! Real units !
        Vz = sim_box.h[3][3] * atoms.z1[i]
        atoms.sumPx[ntp] = atoms.sumPx[ntp] + Am * Vx
        atoms.sumPy[ntp] = atoms.sumPy[ntp] + Am * Vy
        atoms.sumPz[ntp] = atoms.sumPz[ntp] + Am * Vz
        pot_module.E_kin[ntp] = pot_module.E_kin[ntp] + \
                                            Am * (math.pow(Vx, 2) + math.pow(Vy, 2) + math.pow(Vz, 2))  # ! mv ^ 2 !

    Ek_sys = 0.0
    avr_mass = 0.0
    for k in range(1, atoms.iatom_types + 1):
        na = pot_module.natoms_of_type[k]  # ! do not use max(1, natoms_of_type(k)) !
        if na > 0:
            atoms.sumPx[k] = atoms.sumPx[k] / na
            atoms.sumPy[k] = atoms.sumPy[k] / na
            atoms.sumPz[k] = atoms.sumPz[k] / na
            Ek_sys = Ek_sys + pot_module.E_kin[k]  # ! Sum_(mv ^ 2) !
            pot_module.E_kin[k] = pot_module.E_kin[k] / na  # ! mv ^ 2 !
            pot_module.sum_mass[k] = pot_module.Am_of_type[k]
            avr_mass = avr_mass + pot_module.sum_mass[k]
        else:
            atoms.sumPx[k] = 0.0
            atoms.sumPy[k] = 0.0
            atoms.sumPz[k] = 0.0
            pot_module.E_kin[k] = 0.0
            pot_module.sum_mass[k] = 0.0

    Ek_sys = Ek_sys / (2.0 * sim_box.natoms * math.pow(dt, 2))  # ! mv ^ 2 / 2 !
    avr_mass = avr_mass / sim_box.natoms

    T_sys = 0.0
    Ttemp = 1.0 / (3.0 * constants.Boltz_Kb * math.pow(dt, 2))
    for ntp in range(1, atoms.iatom_types + 1):
        if pot_module.natoms_of_type[ntp] > 1:  # ! more than 1 atom of type !
            pot_module.pTemp[ntp] = Ttemp * \
                                                (pot_module.E_kin[ntp] -
                                                 (math.pow(atoms.sumPx[ntp], 2) +
                                                  math.pow(atoms.sumPy[ntp], 2) +
                                                  math.pow(atoms.sumPz[ntp], 2)) /
                                                 pot_module.sum_mass[ntp])
        elif pot_module.natoms_of_type[ntp] == 1:  # ! only 1 atom of type !
            # ! (2 / 3kT) * E_k = mv ^ 2 / 3 kT !
            pot_module.pTemp[ntp] = Ttemp * pot_module.E_kin[ntp]
        else:  # ! No atoms of type ntp !
            pot_module.pTemp[ntp] = 0.0
        T_sys = T_sys + pot_module.pTemp[ntp] * pot_module.sum_mass[ntp] / \
                pot_module.amass[ntp] / sim_box.natoms

        # write(1000 + mynod, 10) ntp, natoms_of_type(ntp), pTemp(ntp), T_set,
        # amass(ntp) / atu, sum_mass(ntp), T_sys, Ek_sys
        # 10 format('get_T: natoms_of_type(', i2, ')=', i6, ' T=', f10 .2,
        # ' T_set=', f10 .2, ' mass=', f10 .3, ' sum_mass=', f10 .3,
        # ' T_sys=', f10 .2, ' Ek_sys=', f12 .8)

    # ! do ntp = 1, iatom_types !

    return
    # ! End of get_T !

#
# -----------------------------------------------------------------------
# Sets second derivatives(accelerations) according to forces,
# so that the predictor - corrector can start with supplied
# first and second derivatives.
# -----------------------------------------------------------------------
#

def initaccel():

    dtsqh = 0.5 * math.pow(atoms.dt_step, 2)

    hi11 = sim_box.hi[1][1]
    hi12 = sim_box.hi[1][2]
    hi13 = sim_box.hi[1][3]
    hi22 = sim_box.hi[2][2]
    hi23 = sim_box.hi[2][3]
    hi33 = sim_box.hi[3][3]

    for n in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[n]
        dtsqhM = dtsqh / pot_module.amass[ntp]  # ! for predict - correct !
        fxsn = hi11 * atoms.frr[1][n] + hi12 * atoms.frr[2][n] + \
               hi13 * atoms.frr[3][n]
        fysn = hi22 * atoms.frr[2][n] + hi23 * atoms.frr[3][n]  # ! S - space !
        fzsn = hi33 * atoms.frr[3][n]
        atoms.x2[n] = dtsqhM * fxsn  # ! Taylor expansion in s - space: !
        atoms.y2[n] = dtsqhM * fysn  # ! vel = dt * dr / dt  !
        atoms.z2[n] = dtsqhM * fzsn
    # ! acc = 1 / 2 * dt ^ 2 * f / m !

    return
    # ! End of initaccel !

#
# -----------------------------------------------------------------------
# Called either from read_structure with or without velocities,
# or from SIM_run when MD starts
# -----------------------------------------------------------------------
#

def init_vel( T_set0):

    #
    # *** assign initial velocities to atoms ***************************
    #
    hi11 = sim_box.hi[1][1]
    hi12 = sim_box.hi[1][2]
    hi13 = sim_box.hi[1][3]
    hi22 = sim_box.hi[2][2]
    hi23 = sim_box.hi[2][3]
    hi33 = sim_box.hi[3][3]

    tfc = 2.0 / (3.0 * constants.Boltz_Kb)

    #
    # *** use ranmar number gen.of Marsaglia and Zaman FSU - SCRI - 87 - 50
    # *** irr must be integer and originally set to an odd large number
    # *** xx, yy, zz range between 0.0 and 1.0
    #

    irr = 6751  # ! VVVV for test with escm_init only !
    sim_box.rmarin(irr + 17 * sim_box.mynod)  # ! node depndnt init.random generator !

    for i in range(1, atoms.iatom_types + 1):
        atoms.sumPx[i] = 0.0
        atoms.sumPy[i] = 0.0
        atoms.sumPz[i] = 0.0
        pot_module.E_kin[i] = 0.0

    sumx = 0.0
    sumy = 0.0
    sumz = 0.0
    sim_box.ranmar(3 * sim_box.natoms)
    ii = 1
    for i in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[i]
        Am = pot_module.amass[ntp]
        xx = sim_box.buf[ii]
        yy = sim_box.buf[ii + 1]  # ! get random velocity numbers !
        zz = sim_box.buf[ii + 2]
        ii = ii + 3
        xyz = math.sqrt(xx * xx + yy * yy + zz * zz)
        atoms.x1[i] = xx / xyz
        atoms.y1[i] = yy / xyz
        atoms.z1[i] = zz / xyz
        atoms.sumPx[ntp] = atoms.sumPx[ntp] + Am * atoms.x1[i]
        # ! in Real space !
        atoms.sumPy[ntp] = atoms.sumPy[ntp] + Am * atoms.y1[i]
        atoms.sumPz[ntp] = atoms.sumPz[ntp] + Am * atoms.z1[i]

    tot_Px = 0.0
    tot_Py = 0.0
    tot_Pz = 0.0
    tot_M = 0.0
    for ntp in range(1, atoms.iatom_types + 1):
        Am = pot_module.amass[ntp]
        na = max(1, pot_module.natoms_of_type[ntp])
        tot_Px = tot_Px + atoms.sumPx[ntp]
        tot_Py = tot_Py + atoms.sumPy[ntp]
        tot_Pz = tot_Pz + atoms.sumPz[ntp]
        tot_M = tot_M + Am * na
        atoms.sumPx[ntp] = 0.0
        atoms.sumPy[ntp] = 0.0
        atoms.sumPz[ntp] = 0.0

    #
    # *** adjust velocities such that total linear momentum is zero.
    #

    for i in range(1, sim_box.natoms + 1):  # ! V_cm = Sum_i {mi * Vi} / Sum_i {mi} !
        ntp = atoms.ntype[i]
        Am = pot_module.amass[ntp]
        atoms.x1[i] = atoms.x1[i] - tot_Px / tot_M  # ! vi = vi - V_cm !
        atoms.y1[i] = atoms.y1[i] - tot_Py / tot_M
        atoms.z1[i] = atoms.z1[i] - tot_Pz / tot_M
        vx = atoms.x1[i]
        vy = atoms.y1[i]  # ! R - space[A / ps] !
        vz = atoms.z1[i]
        # ! New momentums !
        atoms.sumPx[ntp] = atoms.sumPx[ntp] + Am * atoms.x1[i]
        # ! in Real space !
        atoms.sumPy[ntp] = atoms.sumPy[ntp] + Am * atoms.y1[i]
        atoms.sumPz[ntp] = atoms.sumPz[ntp] + Am * atoms.z1[i]
        pot_module.E_kin[ntp] = pot_module.E_kin[ntp] + Am * (vx * vx + vy * vy + vz * vz)

        # if (ident(i).eq.1) then
        # write(50, 110) i, ident(i), x1(i), y1(i), z1(i), sumPx(ntp), sumPy(ntp), sumPz(ntp), Am, E_kin(ntp)
        # endif

    pot_module.pTemp[0] = 0.0
    for i in range(1, atoms.iatom_types + 1):
        na = pot_module.natoms_of_type[i]
        if na > 0:
            # ! = 2 / 3 * Ek / Kb !
            pot_module.pTemp[i] = 0.5 * tfc * pot_module.E_kin[i] / na
            if pot_module.pTemp[i] > 0.001:
                atoms.Tscale[i] = math.sqrt(T_set0 / pot_module.pTemp[i])
            else:
                atoms.Tscale[i] = 0.0  # ! if T is too low, freeze this atom !
        else:  # ! when na = 0 !
            pot_module.pTemp[i] = 0.0  # ! No atom is rescaled to V = 0, E_kin = 0 !
            atoms.Tscale[i] = 0.0
            na = 1
        # ! if (na.gt.0)... !

        # write(50, *) 'init_vel: natoms_of_type(', i, ')=',
        # natoms_of_type(i), ' T=', pTemp(i), ' Ek=', E_kin(i) / na, ' atu=', atu,
        # ' T_0=', T_set0, ' Tscale=', Tscale(i), ' tfc=', tfc, ' mass=', amass(i),
        # ' na=', na

    # ! do i = 1, iatom_types !

    #
    # *** scale velocities to desired temp. ***********************
    #

    for i in range(1, sim_box.natoms + 1):
        ntp = atoms.ntype[i]
        atoms.x1[i] = atoms.x1[i] * atoms.Tscale[ntp]  # ! R - space !
        atoms.y1[i] = atoms.y1[i] * atoms.Tscale[ntp]
        atoms.z1[i] = atoms.z1[i] * atoms.Tscale[ntp]

    # ! Convert to S - space !

    for i in range(1, sim_box.natoms + 1):

        # if (ident(i).eq.1) then
        # write(50, 15) ident(i), x1(i), y1(i), z1(i)
        # endif
        # 15 format('init_vel1: id:', i5, ' R x1=', 3E18.10)

        atoms.x3[i] = (hi11 * atoms.x1[i] + hi12 * atoms.y1[i] + hi13 *
                                   atoms.z1[i])
        atoms.y3[i] = (hi22 * atoms.y1[i] + hi23 * atoms.z1[i])
        atoms.z3[i] = hi33 * atoms.z1[i]

        atoms.x1[i] = atoms.x3[i] * atoms.dt_step
        atoms.y1[i] = atoms.y3[i] * atoms.dt_step
        atoms.z1[i] = atoms.z3[i] * atoms.dt_step

        atoms.x2[i] = 0.0
        atoms.y2[i] = 0.0
        atoms.z2[i] = 0.0
        atoms.x3[i] = 0.0
        atoms.y3[i] = 0.0
        atoms.z3[i] = 0.0
        atoms.x4[i] = 0.0
        atoms.y4[i] = 0.0
        atoms.z4[i] = 0.0
        atoms.x5[i] = 0.0
        atoms.y5[i] = 0.0
        atoms.z5[i] = 0.0

        # if (ident(i).eq.1) then
        # write(50, 16) ident(i), x1(i), y1(i), z1(i)
        # endif
        # 16 format('init_vel2: id:', i5, ' S x1*dt=', 3E18.10)

    for i in range(1, 3 + 1):
        for j in range(1, 3 + 1):
            sim_box.h1[i][j] = 0.0
            sim_box.h2[i][j] = 0.0
            sim_box.h3[i][j] = 0.0
            sim_box.h4[i][j] = 0.0
            sim_box.h5[i][j] = 0.0

    A_fr = 0.0  # ! Reset dissipated friction energy !
    Q_heat = 0.0

    get_T()  # ! calc.pTemp(ntp), E_kin(ntp), T_sys and Ek_sys !

    return
    # ! End of init_vel !

#
# --------------------------------------------------------------------
#

def init_MD():

    global f02_atom,f12,f32,f42,f52,f02_wall,f02_atom,f02viscous


    # ! Fifth order predictor - corrector coefficients !
    f02 = 3.0 / 20.0  # ! in sim_box module !
    f02viscous = 3.0 / 16.0
    f12 = 251.0 / 360.0
    # ! f22 = 1.0 d0 !
    f32 = 11.0 / 18.0
    f42 = 1.0 / 6.0
    f52 = 1.0 / 60.0
    f02_wall = f02
    f02_atom = f02

    return
    # ! End of init_MD !

#
# ------------------------------------------------------------------
#
# END MODULE ! MD !