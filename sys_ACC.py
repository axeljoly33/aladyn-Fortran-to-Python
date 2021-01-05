devicetype = 0
acc_device_current = 0
My_GPU_mem, My_GPU_free_mem = 0, 0

#
# ------------------------------------------------------------------
#

def get_device_type():
    global devicetype

    get_device_type = 0

    return get_device_type
    # ! end function !

#
# ------------------------------------------------------------------
#

def get_num_devices(idev_type):
    global get_num_devices

    get_num_devices = 0

    return get_num_devices
    # ! end function !

#
# ------------------------------------------------------------------
#

def set_device_num(my_device, dev_type):

    # integer, intent(in) :: my_device, dev_type

    return
    # ! end subroutine !

#
# ------------------------------------------------------------------
#

def get_gpu_mem( My_GPU):

    #integer, intent(in) :: My_GPU
    global My_GPU_mem

    get_gpu_mem = 0

    return get_gpu_mem
    # ! end function !

#
# ------------------------------------------------------------------
#

def get_gpu_free_mem( My_GPU):

    #integer, intent(in) :: My_GPU
    global My_GPU_free_mem

    get_gpu_free_mem = 0

    return get_gpu_free_mem
    # ! end function !

#
# ------------------------------------------------------------------
#
def GPU_Init( my_device):

    #integer, intent(in) :: my_device
    global My_GPU_mem,My_GPU_free_mem

    My_GPU_mem = 0
    My_GPU_free_mem = 0

    return