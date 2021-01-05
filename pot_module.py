import sys
import operator
import numpy as np
import random
import torch
import math

import atoms
import sim_box
import constants
import aladyn_IO


maxconstr = 63  # ! Max number of defined constraints !
chem_symb = [""] * (112+1)
ipot = [0] * (112+1)  # ! maximum numb.elements possible !

elem_symb = [""]
Elem_in_com = [""]
filename = ""

PotEnrg_atm, PotEnrg_glb, ecoh = 0.0, 0.0, 0.0

ifile_numbs, nelem_in_com, n_pot_files = 0, 0, 0
INP_STR_TYPE, iOUT_STR_TYPE, N_points_max, Nrho_points_max = 1, 0, 0, 0

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
elem_radius = [0.0] * (112+1)

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

def conv_to_low(strIn):

    strOut = strIn.lower()

    return strOut
    # ! End of conv_to_low !

# !
# !--------------------------------------------------------------------
# !

def init_elements():
    global chem_symb,elem_radius

    chem_symb[0] = ''
    elem_radius[0] = 0.0  # ! Ang !
    chem_symb[1] = 'H'
    elem_radius[1] = 0.53
    chem_symb[2] = 'He'
    elem_radius[2] = 0.31
    chem_symb[3] = 'Li'
    elem_radius[3] = 1.45
    chem_symb[4] = 'Be'
    elem_radius[4] = 1.05
    chem_symb[5] = 'B'
    elem_radius[5] = 0.85
    chem_symb[6] = 'C'
    elem_radius[6] = 0.70
    chem_symb[7] = 'N'
    elem_radius[7] = 0.65
    chem_symb[8] = 'O'
    elem_radius[8] = 0.60
    chem_symb[9] = 'F'
    elem_radius[9] = 0.50
    chem_symb[10] = 'Ne'
    elem_radius[10] = 0.38
    chem_symb[11] = 'Na'
    elem_radius[11] = 1.80
    chem_symb[12] = 'Mg'
    elem_radius[12] = 1.50
    chem_symb[13] = 'Al'
    elem_radius[13] = 1.18
    chem_symb[14] = 'Si'
    elem_radius[14] = 1.10
    chem_symb[15] = 'P'
    elem_radius[15] = 1.00
    chem_symb[16] = 'S'
    elem_radius[16] = 1.00
    chem_symb[17] = 'Cl'
    elem_radius[17] = 1.00
    chem_symb[18] = 'Ar'
    elem_radius[18] = 0.71
    chem_symb[19] = 'K'
    elem_radius[19] = 2.20
    chem_symb[20] = 'Ca'
    elem_radius[20] = 1.80
    chem_symb[21] = 'Sc'
    elem_radius[21] = 1.60
    chem_symb[22] = 'Ti'
    elem_radius[22] = 1.40
    chem_symb[23] = 'V'
    elem_radius[23] = 1.35
    chem_symb[24] = 'Cr'
    elem_radius[24] = 1.40
    chem_symb[25] = 'Mn'
    elem_radius[25] = 1.40
    chem_symb[26] = 'Fe'
    elem_radius[26] = 1.40
    chem_symb[27] = 'Co'
    elem_radius[27] = 1.35
    chem_symb[28] = 'Ni'
    elem_radius[28] = 1.35
    chem_symb[29] = 'Cu'
    elem_radius[29] = 1.35
    chem_symb[30] = 'Zn'
    elem_radius[30] = 1.35
    chem_symb[31] = 'Ga'
    elem_radius[31] = 1.30
    chem_symb[32] = 'Ge'
    elem_radius[32] = 1.25
    chem_symb[33] = 'As'
    elem_radius[33] = 1.15
    chem_symb[34] = 'Se'
    elem_radius[34] = 1.15
    chem_symb[35] = 'Br'
    elem_radius[35] = 1.15
    chem_symb[36] = 'Kr'
    elem_radius[36] = 0.88
    chem_symb[37] = 'Rb'
    elem_radius[37] = 2.35
    chem_symb[38] = 'Sr'
    elem_radius[38] = 2.00
    chem_symb[39] = 'Y'
    elem_radius[39] = 1.85
    chem_symb[40] = 'Zr'
    elem_radius[40] = 1.55
    chem_symb[41] = 'Nb'
    elem_radius[41] = 1.45
    chem_symb[42] = 'Mo'
    elem_radius[42] = 1.45
    chem_symb[43] = 'Tc'
    elem_radius[43] = 1.35
    chem_symb[44] = 'Ru'
    elem_radius[44] = 1.30
    chem_symb[45] = 'Rh'
    elem_radius[45] = 1.35
    chem_symb[46] = 'Pd'
    elem_radius[46] = 1.40
    chem_symb[47] = 'Ag'
    elem_radius[47] = 1.60
    chem_symb[48] = 'Cd'
    elem_radius[48] = 1.55
    chem_symb[49] = 'In'
    elem_radius[49] = 1.55
    chem_symb[50] = 'Sn'
    elem_radius[50] = 1.45
    chem_symb[51] = 'Sb'
    elem_radius[51] = 1.45
    chem_symb[52] = 'Te'
    elem_radius[52] = 1.40
    chem_symb[53] = 'I'
    elem_radius[53] = 1.40
    chem_symb[54] = 'Xe'
    elem_radius[54] = 1.08
    chem_symb[55] = 'Cs'
    elem_radius[55] = 2.60
    chem_symb[56] = 'Ba'
    elem_radius[56] = 2.15
    chem_symb[57] = 'La'
    elem_radius[57] = 1.95
    chem_symb[58] = 'Ce'
    elem_radius[58] = 1.85
    chem_symb[59] = 'Pr'
    elem_radius[59] = 1.85
    chem_symb[60] = 'Nd'
    elem_radius[60] = 1.85
    chem_symb[61] = 'Pm'
    elem_radius[61] = 1.85
    chem_symb[62] = 'Sm'
    elem_radius[62] = 1.85
    chem_symb[63] = 'Eu'
    elem_radius[63] = 1.85
    chem_symb[64] = 'Gd'
    elem_radius[64] = 1.80
    chem_symb[65] = 'Tb'
    elem_radius[65] = 1.75
    chem_symb[66] = 'Dy'
    elem_radius[66] = 1.75
    chem_symb[67] = 'Ho'
    elem_radius[67] = 1.75
    chem_symb[68] = 'Er'
    elem_radius[68] = 1.75
    chem_symb[69] = 'Tm'
    elem_radius[69] = 1.75
    chem_symb[70] = 'Yb'
    elem_radius[70] = 1.75
    chem_symb[71] = 'Lu'
    elem_radius[71] = 1.75
    chem_symb[72] = 'Hf'
    elem_radius[72] = 1.55
    chem_symb[73] = 'Ta'
    elem_radius[73] = 1.45
    chem_symb[74] = 'W'
    elem_radius[74] = 1.35
    chem_symb[75] = 'Re'
    elem_radius[75] = 1.35
    chem_symb[76] = 'Os'
    elem_radius[76] = 1.30
    chem_symb[77] = 'Ir'
    elem_radius[77] = 1.35
    chem_symb[78] = 'Pt'
    elem_radius[78] = 1.35
    chem_symb[79] = 'Au'
    elem_radius[79] = 1.35
    chem_symb[80] = 'Hg'
    elem_radius[80] = 1.50
    chem_symb[81] = 'Tl'
    elem_radius[81] = 1.90
    chem_symb[82] = 'Pb'
    elem_radius[82] = 1.80
    chem_symb[83] = 'Bi'
    elem_radius[83] = 1.60
    chem_symb[84] = 'Po'
    elem_radius[84] = 1.90
    chem_symb[85] = 'At'
    elem_radius[85] = 1.27
    chem_symb[86] = 'Rn'
    elem_radius[86] = 1.20
    chem_symb[87] = 'Fr'
    elem_radius[87] = 1.94
    chem_symb[88] = 'Ra'
    elem_radius[88] = 2.15
    chem_symb[89] = 'Ac'
    elem_radius[89] = 1.95
    chem_symb[90] = 'Th'
    elem_radius[90] = 1.80
    chem_symb[91] = 'Pa'
    elem_radius[91] = 1.80
    chem_symb[92] = 'U'
    elem_radius[92] = 1.75
    chem_symb[93] = 'Np'
    elem_radius[93] = 1.75
    chem_symb[94] = 'Pu'
    elem_radius[94] = 1.75
    chem_symb[95] = 'Am'
    elem_radius[95] = 1.75
    chem_symb[96] = 'Cm'
    elem_radius[96] = 1.75
    chem_symb[97] = 'Bk'
    elem_radius[97] = 1.75
    chem_symb[98] = 'Cf'
    elem_radius[98] = 1.75
    chem_symb[99] = 'Es'
    elem_radius[99] = 1.75
    chem_symb[100] = 'Fm'
    elem_radius[100] = 1.75
    chem_symb[101] = 'Md'
    elem_radius[101] = 1.75
    chem_symb[102] = 'No'
    elem_radius[102] = 1.75
    chem_symb[103] = 'Lr'
    elem_radius[103] = 1.75
    chem_symb[104] = 'Rf'
    elem_radius[104] = 1.75
    chem_symb[105] = 'Db'
    elem_radius[105] = 1.75
    chem_symb[106] = 'Sg'
    elem_radius[106] = 1.75
    chem_symb[107] = 'Bh'
    elem_radius[107] = 1.75
    chem_symb[108] = 'Hs'
    elem_radius[108] = 1.75
    chem_symb[109] = 'Mt'
    elem_radius[109] = 1.75
    chem_symb[110] = 'Ds'
    elem_radius[110] = 1.75
    chem_symb[111] = 'Rg'
    elem_radius[111] = 1.75
    chem_symb[112] = 'Cn'
    elem_radius[112] = 1.75

    return
    # ! End of init_elements !

# !
# !--------------------------------------------------------------------
# !

def numb_elem_Z(chem):
    global chem_symb
    chem_low = ""
    chem_symb_low = ""

    len_chem = len(chem.lstrip())

    numZ = 0
    chem_low = conv_to_low(chem)
    for iZ in range(1, 112 + 1):
        len_symb = len(chem_symb[iZ].lstrip())
        chem_symb_low = conv_to_low(chem_symb[iZ])
        if len_chem == len_symb:
            if chem_low == chem_symb_low:
                numZ = iZ
                break

    # ! write(6, *) 'numb_elem_Z(', chem, ')=', numZ

    numb_elem_Z = numZ

    return numb_elem_Z
    # ! End of numb_elem_Z !

# !
# !---------------------------------------------------------------------
# ! *** Gets current chem.composition ***
# ! Includes all atoms that DO NOT have chemical constrain !
# !---------------------------------------------------------------------
# !

def get_chem():

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

def alloc_pot_types():

    global elem_symb,ielement,natoms_of_type,iZ_elem_in_com,amass,sum_mass,gram_mol,Am_of_type,Elem_in_com,r_pair_cut
    global r2_pair_cut,E_kin,pTemp,filename

    ialloc = [0] * (14 + 1)

    elem_symb = [""] * (atoms.iatom_types + 1)

    ielement = [0] * (atoms.iatom_types + 1)
    natoms_of_type = [0] * (atoms.iatom_types + 1)
    iZ_elem_in_com = [0] * (atoms.iatom_types + 1)

    amass = [0.0] * (atoms.iatom_types + 1)
    sum_mass = [0.0] * (atoms.iatom_types + 1)
    gram_mol = [0.0] * (atoms.iatom_types + 1)

    Am_of_type = [0.0] * (atoms.iatom_types + 1)
    Elem_in_com = [""] * (atoms.iatom_types + 1)

    r_pair_cut = [0.0] * (atoms.ipair_types + 1)
    r2_pair_cut = [0.0] * (atoms.ipair_types + 1)

    E_kin = [0.0] * (atoms.iatom_types + 1)
    pTemp = [0.0] * (atoms.iatom_types + 1)
    filename = ""

    ierror = 0
    for i in range(1, 14 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of alloc_pot_types !

# !
# ! ---------------------------------------------------------------------
# !

def deall_pot_types():

    global elem_symb,filename,ielement,natoms_of_type,iZ_elem_in_com,amass,sum_mass,gram_mol,Am_of_type,Elem_in_com
    global r_pair_cut,r2_pair_cut,E_kin,pTemp

    ialloc = [0] * (14 + 1)

    if elem_symb:
        elem_symb.clear()
    if filename:
        filename = ""

    if ielement:
        ielement.clear()
    if natoms_of_type:
        natoms_of_type.clear()
    if iZ_elem_in_com:
        iZ_elem_in_com.clear()

    if amass:
        amass.clear()
    if sum_mass:
        sum_mass.clear()
    if gram_mol:
        gram_mol.clear()
    if Am_of_type:
        Am_of_type.clear()
    if Elem_in_com:
        Elem_in_com.clear()

    if r_pair_cut:
        r_pair_cut.clear()
    if r2_pair_cut:
        r2_pair_cut.clear()

    if E_kin:
        E_kin.clear()
    if pTemp:
        pTemp.clear()

    ierror = 0
    for i in range(1, 14 + 1):
        ierror += ialloc[i]

    return ierror
    # ! End of deall_pot_types !

# ! End of pot_module !
