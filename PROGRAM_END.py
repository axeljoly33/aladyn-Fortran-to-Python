import sys

import atoms
import aladyn_ANN
import sim_box
import pot_module


def PROGRAM_END(ierr):
    #use sys_OMP
    #use sys_ACC
    #use sim_box
    #use pot_module
    #use atoms
    #use ANN

    ierror = 0

    ierror = atoms.deall_atoms_sys()
    ierror = atoms.deall_atoms_MD()

    ierror = aladyn_ANN.deall_types_ANN()
    ierror = aladyn_ANN.deall_atoms_ANN()

    ierror = sim_box.deall_buffers()
    ierror = pot_module.deall_pot_types()
    ierror = sim_box.deall_cells()

    if ierr != 0:
        sys.exit("")    # ! PROGRAM_END was called due to an error !
                        # ! STOP the execution, otherwise continue !
    return
    # ! End of PROGRAM_END !
