import math
from matrixoperations import multiply,transpose,diagonal
def jacobi(a):
    n=len(a)
    v=[]
    for i in range(n):
        x=[]
        for j in range(n):
            if i==j:
                x.append(1.0)
            else:
                x.append(0.0)
        v.append(x)
    for z in range(100):
        m=0
        for p in range(n-1):
            for q in range(p+1,n):
                if abs(a[p][q])>m:
                    m=abs(a[p][q])
        if m<1e-10:
            break
        for p in range(n-1):
            for q in range(p+1,n):
                if abs(a[p][q])<1e-12:
                    continue
                t=0.5*math.atan2(2*a[p][q],a[q][q]-a[p][p])
                c=math.cos(t)
                s=math.sin(t)
                for i in range(n):
                    if i!=p and i!=q:
                        x=a[i][p]
                        y=a[i][q]
                        a[i][p]=c*x-s*y
                        a[p][i]=a[i][p]
                        a[i][q]=s*x+c*y
                        a[q][i]=a[i][q]
                x=a[p][p]
                y=a[q][q]
                z=a[p][q]
                a[p][p]=c*c*x-2*s*c*z+s*s*y
                a[q][q]=s*s*x+2*s*c*z+c*c*y
                a[p][q]=0
                a[q][p]=0
                for i in range(n):
                    x=v[i][p]
                    y=v[i][q]
                    v[i][p]=c*x-s*y
                    v[i][q]=s*x+c*y
    e=[]
    for i in range(n):
        e.append(a[i][i])
    return e,v
def sortvalues(a,b):
    n=len(a)
    x=[]
    for i in range(n):
        y=[]
        for j in range(n):
            y.append(b[j][i])
        x.append((a[i],y))
    for i in range(n):
        p=i
        for j in range(i+1,n):
            if x[j][0]>x[p][0]:
                p=j
        x[i],x[p]=x[p],x[i]
    a=[]
    b=[]
    for i in range(n):
        a.append(x[i][0])
        b.append(x[i][1])
    z=[]
    for i in range(n):
        y=[]
        for j in range(n):
            y.append(b[j][i])
        z.append(y)
    return a,z
def svd(a):
    m=len(a)
    n=len(a[0])
    # svd
    at=transpose(a)
    ata=multiply(at,a)
    e,v=jacobi(ata)
    e,v=sortvalues(e,v)
    for i in range(len(e)):
        if e[i]<0:
            e[i]=0
    s=[]
    for i in range(len(e)):
        s.append(math.sqrt(e[i]))
    r=min(m,n)
    s=s[:r]
    v1=[]
    for i in range(n):
        x=[]
        for j in range(r):
            x.append(v[i][j])
        v1.append(x)
    u=[]
    for i in range(m):
        u.append([0]*r)
    # U
    for i in range(r):
        if s[i]>1e-12:
            for j in range(m):
                z=0
                for k in range(n):
                    z+=a[j][k]*v1[k][i]
                u[j][i]=z/s[i]
    return u,s,transpose(v1)
def svdreconstruction(a,k):
    u,s,v=svd(a)
    if k>len(s):
        k=len(s)
    u1=[]
    for i in range(len(u)):
        x=[]
        for j in range(k):
            x.append(u[i][j])
        u1.append(x)
    s1=diagonal(s[:k])
    v1=[]
    for i in range(k):
        x=[]
        for j in range(len(v[0])):
            x.append(v[i][j])
        v1.append(x)
    x=multiply(u1,s1)
    x=multiply(x,v1)
    return x