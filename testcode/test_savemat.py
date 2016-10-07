import scipy.io as sio
import numpy as np
test = np.zeros((3,4))
test_dict = {'data':test}
sio.savemat('test.mat',test_dict)
#sio.savemat(file_name, mdict, appendmat, format, long_field_names, do_compression, oned_as)