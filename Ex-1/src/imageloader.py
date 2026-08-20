import numpy as np
from PIL import Image
def loadimage(a):
    # img to grayscale, resize
    x=Image.open(a).convert("L")
    x=x.resize((100,100))
    return np.array(x,dtype=float)