import numpy as np
import matplotlib.pyplot as plt
from imageloader import loadimage
from svdreconstruction import svdreconstruction
from evdreconstruction import evdreconstruction
from error import frobeniuserror
def analyzeimage(a,is_square=True):
    a=loadimage(a)
    print("Image size:",a.shape,flush=True)
    n=min(a.shape)
    k=list(range(1,n+1))
    es=[]
    for i in k:
        print("SVD reconstruction for k =",i,flush=True)
        x=svdreconstruction(a,i)
        e=frobeniuserror(a,x)
        es.append(e)
    ee=[]
    ev=None
    p=None
    if is_square:
        print("Calculating EVD...",flush=True)
        ev,p=np.linalg.eig(a)
        print("EVD completed.",flush=True)
        for i in k:
            print("EVD reconstruction for k =",i,flush=True)
            x=evdreconstruction(a,ev,p,i)
            e=frobeniuserror(a,x)
            ee.append(e)
    plt.figure(figsize=(10,6))
    plt.plot(k,es,marker="o",label="SVD")
    if is_square:
        plt.plot(k,ee,marker="s",label="EVD")
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Frobenius Reconstruction Error")
    plt.title("Reconstruction Error vs Number of Components")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    z=[5,20,60]
    for i in z:
        if i>n:
            continue
        print("Creating reconstruction for k =",i,flush=True)
        x=svdreconstruction(a,i)
        ex=np.abs(a-x)
        if is_square:
            y=evdreconstruction(a,ev,p,i)
            ey=np.abs(a-y)
            f=plt.figure(figsize=(20,5))
            a1=f.add_subplot(1,5,1)
            a2=f.add_subplot(1,5,2)
            a3=f.add_subplot(1,5,3)
            a4=f.add_subplot(1,5,4)
            a5=f.add_subplot(1,5,5)
            a1.imshow(a,cmap="gray")
            a1.set_title("Original")
            a1.axis("off")
            a2.imshow(x,cmap="gray")
            a2.set_title("SVD, k="+str(i))
            a2.axis("off")
            a3.imshow(y,cmap="gray")
            a3.set_title("EVD, k="+str(i))
            a3.axis("off")
            a4.imshow(ex,cmap="gray")
            a4.set_title("SVD Error")
            a4.axis("off")
            a5.imshow(ey,cmap="gray")
            a5.set_title("EVD Error")
            a5.axis("off")
            plt.tight_layout()
            plt.show()
    print("Analysis finished.",flush=True)