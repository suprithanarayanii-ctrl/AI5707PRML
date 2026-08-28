import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from imageloader import loadimage
from error import frobeniuserror
from svdreconstruction import svd,svdreconstruction
def analyzerectangularimage(a):
    o,a=loadimage(a)
    print("Original image size:",o.shape,flush=True)
    print("Resized image size:",a.shape,flush=True)
    print("Calculating SVD...",flush=True)
    u,s,v=svd(a)
    print("SVD completed.",flush=True)
    n=len(s)
    print("Number of components:",n,flush=True)
    e=[]
    for k in range(1,n+1):
        x=svdreconstruction(a,k)
        y=frobeniuserror(a,x)
        e.append(y)
    k=np.arange(1,n+1)
    plt.figure(figsize=(10,6))
    plt.plot(k,e,marker="o",label="SVD")
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Frobenius Reconstruction Error")
    plt.title("SVD Reconstruction Error vs Number of Components")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    z=[5,20,70]
    for k in z:
        if k>n:
            continue
        print("Creating reconstruction for k =",k,flush=True)
        x=svdreconstruction(a,k)
        y=np.abs(a-x)
        xd=Image.fromarray(np.clip(x,0,255).astype(np.uint8))
        xd=xd.resize((o.shape[1],o.shape[0]),Image.Resampling.BICUBIC)
        xd=np.array(xd)
        yd=Image.fromarray(np.clip(y,0,255).astype(np.uint8))
        yd=yd.resize((o.shape[1],o.shape[0]),Image.Resampling.NEAREST)
        yd=np.array(yd)
        f=plt.figure(figsize=(18,6))
        a1=f.add_subplot(1,3,1)
        a2=f.add_subplot(1,3,2)
        a3=f.add_subplot(1,3,3)
        a1.imshow(o,cmap="gray")
        a1.set_title("Original")
        a1.axis("off")
        a2.imshow(xd,cmap="gray")
        a2.set_title("SVD Reconstruction, k="+str(k))
        a2.axis("off")
        a3.imshow(yd,cmap="gray")
        a3.set_title("SVD Error, k="+str(k))
        a3.axis("off")
        plt.tight_layout()
        plt.show()
    print("Analysis finished.",flush=True)