def transpose(A):
  row=len(A)
  col=len(A[0])
  At=[[0 for _ in range(row)]for _ in range(col)]
  for i in range(row):
    for j in range(col):
      At[j][i]=A[i][j]
  return At

def multiply(A,B):
  rowA=len(A)
  colA=len(A[0])
  rowB=len(B)
  colB=len(B[0])
  if colA!=rowB:
    raise ValueError("Multplication cant be performed")
  result=[[0 for _ in range(colB)]for _ in range(rowA)]
  for i in range(rowA):
    for j in range(colB):
      for k in range(colA):
          result[i][j]+=A[i][k]*B[k][j]
  return result

def minor_matrix(A,r,c):
  row=len(A)
  col=len(A[0])
  minor=[]
  for i in range(row):
    if i==r:
      continue
    rowMat=[]
    for j in range(col):
      if j==c:
        continue
      rowMat.append(A[i][j])
    minor.append(rowMat)
  return minor

def determinant(A):
  if len(A)==1:
     return A[0][0]
  if len(A)==2:
    det=A[0][0]*A[1][1] - A[1][0]*A[0][1]
    return det
  det=0
  for j in range(len(A[0])):
    minor=minor_matrix(A,0,j)
    minor_det=determinant(minor)
    det+= ((-1)**j)*A[0][j]*minor_det
  return det

def inverse(A):
  row=len(A)
  col=len(A[0])
  det=determinant(A)
  if det==0:
    raise ValueError("Inverse cannot be calculated")
  cofactor=[[0 for _ in range(col)]for _ in range(row)]
  for i in range(row):
    for j in range(col):
      minor=minor_matrix(A,i,j)
      minor_det=determinant(minor)
      cofactor[i][j]=((-1)**(i+j))*minor_det
  adj=transpose(cofactor)
  A_inv=[[0 for _ in range(col)]for _ in range(row)]
  for i in range(row):
    for j in range(col):
      A_inv[i][j]=adj[i][j]/det
  return A_inv