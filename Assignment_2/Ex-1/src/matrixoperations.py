def multiply(a,b):
    r=len(a)
    c=len(a[0])
    rb=len(b)
    cb=len(b[0])
    if c!=rb:
        print("Matrices cannot be multiplied")
        return None
    x=[]
    for i in range(r):
        y=[]
        for j in range(cb):
            s=0
            for k in range(c):
                s+=a[i][k]*b[k][j]
            y.append(s)
        x.append(y)
    return x
def transpose(a):
    r=len(a)
    c=len(a[0])
    x=[]
    for j in range(c):
        y=[]
        for i in range(r):
            y.append(a[i][j])
        x.append(y)
    return x
def inverse(a):
    n=len(a)
    x=[]
    for i in range(n):
        y=[]
        for j in range(n):
            y.append(complex(a[i][j]))
        for j in range(n):
            y.append(1 if i==j else 0)
        x.append(y)
    for i in range(n):
        p=i
        for j in range(i+1,n):
            if abs(x[j][i])>abs(x[p][i]):
                p=j
        if abs(x[p][i])<1e-10:
            print("Matrix is not invertible")
            return None
        x[i],x[p]=x[p],x[i]
        q=x[i][i]
        for j in range(2*n):
            x[i][j]/=q
        for j in range(n):
            if j==i:
                continue
            f=x[j][i]
            for k in range(2*n):
                x[j][k]-=f*x[i][k]
    z=[]
    for i in range(n):
        y=[]
        for j in range(n):
            y.append(x[i][j+n])
        z.append(y)
    return z
def diagonal(a):
    n=len(a)
    x=[]
    for i in range(n):
        y=[]
        for j in range(n):
            if i==j:
                y.append(a[i])
            else:
                y.append(0)
        x.append(y)
    return x