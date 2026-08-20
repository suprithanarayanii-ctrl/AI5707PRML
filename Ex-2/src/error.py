import numpy as np
def frobeniuserror(a,b):
    # norm for error
    return np.linalg.norm(a-b,ord="fro")