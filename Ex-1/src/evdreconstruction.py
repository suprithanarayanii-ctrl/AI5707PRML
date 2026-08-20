from matrixoperations import multiply,inverse,diagonal
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
            if abs(x[j][0])>abs(x[p][0]):
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
def evdreconstruction(a,b,c,k):
    n=len(b)
    b,c=sortvalues(b,c)
    u=[False]*n
    x=[]
    # conjugate pairs
    for i in range(n):
        if u[i]:
            continue
        v=b[i]
        if abs(v.imag)>1e-10:
            f=False
            for j in range(n):
                if i==j or u[j]:
                    continue
                if abs(b[j]-v.conjugate())<1e-10:
                    x.append({"indices":[i,j],"value":abs(v)})
                    u[i]=True
                    u[j]=True
                    f=True
                    break
            if not f:
                x.append({"indices":[i],"value":abs(v)})
                u[i]=True
        else:
            x.append({"indices":[i],"value":abs(v)})
            u[i]=True
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if x[j]["value"]>x[i]["value"]:
                x[i],x[j]=x[j],x[i]
    s=[]
    q=0
    for z in x:
        m=len(z["indices"])
        if q+m>k:
            continue
        for i in z["indices"]:
            s.append(i)
        q+=m
    l=[]
    for i in range(n):
        if i in s:
            l.append(b[i])
        else:
            l.append(0)
    d=diagonal(l)
    q=c
    qi=inverse(q)
    # A = QΛQ^-1
    z=multiply(q,d)
    z=multiply(z,qi)
    for i in range(len(z)):
        for j in range(len(z[0])):
            if abs(z[i][j].imag)<1e-10:
                z[i][j]=z[i][j].real
    return z