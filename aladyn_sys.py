#
# ------------------------------------------------------------------
# 12-10-2020
#
# System Module Unit for the ParaGrandMC.f parallel Monte Carlo code.
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
# 06-01-2016
#
# System Module Unit for the ParaGrandMC.f parallel Monte Carlo code
#
# Vesselin Yamakov
# National Institute of Aerospace
# 100 Exploration Way,
# Hampton, VA 23666 
# phone: (757)-864-2850
# fax:   (757)-864-8911
# e-mail: yamakov@nianet.org
#
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
# Renames some OpenMP functions
# ------------------------------------------------------------------
#

import sys
import operator
import numpy as np
import random
import torch
import math
#import aladyn_mods
#import aladyn_MD
#import aladyn_IO
#import aladyn_ANN
#import aladyn


#
# ------------------------------------------------------------------
#

def GET_NUM_PROCS():

    GET_NUM_PROCS = OMP_GET_NUM_PROCS()

    return GET_NUM_PROCS
    # ! end function !

#
# ------------------------------------------------------------------
#

def GET_NUM_THREADS():

    GET_NUM_THREADS = omp_get_num_threads()

    return GET_NUM_THREADS
    # ! end function !
#
# ------------------------------------------------------------------
#

def GET_THREAD_NUM():

    GET_THREAD_NUM = OMP_get_thread_num()

    return GET_THREAD_NUM
    # ! end function !

#
# ------------------------------------------------------------------
#

def GET_WTIME():

    GET_WTIME = OMP_get_wtime()

    return GET_WTIME
    # ! end function !

#
# ------------------------------------------------------------------
#

def GET_MAX_THREADS():

    GET_MAX_THREADS = OMP_GET_MAX_THREADS()

    return GET_MAX_THREADS
    # ! end function !

#
# ------------------------------------------------------------------
#

def SET_NUM_THREADS(num_thrds):

    # integer, intent(in) :: num_thrds

    OMP_SET_NUM_THREADS(num_thrds)

    return
    # ! end subroutine !

#
# ------------------------------------------------------------------
#
    # END MODULE ! sys_OMP !
#
# =====================================================================
# Overwrites some OpenACC functions when ACC is not available
# ------------------------------------------------------------------
#