import numpy as np
def frobeniuserror(a,b):
    # diff between org and reconst matrix
    return np.linalg.norm(a-b,ord="fro")