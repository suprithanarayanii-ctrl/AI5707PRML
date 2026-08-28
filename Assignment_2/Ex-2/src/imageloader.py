import numpy as np
from PIL import Image
def loadimage(a):
    # small grayscale
    x=Image.open(a).convert("L")
    o=np.array(x,dtype=float)
    x=x.resize((100,75))
    r=np.array(x,dtype=float)
    return o,r