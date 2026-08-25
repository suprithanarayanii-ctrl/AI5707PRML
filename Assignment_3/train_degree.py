from matrix_operations import transpose
from matrix_operations import multiply
from matrix_operations import inverse

#System of equations Y=XW
def train_degree(degree,train):
  x_train=[]
  for line in train:
    x_train.append(line[0])

  # X matrix
  x_train_poly=[]
  for x in x_train:
    value=[]
    for power in range(degree+1):
      value.append(x**power)
    x_train_poly.append(value)
  print(f"x_train_poly={len(x_train_poly)}*{len(x_train_poly[0])}")

  #Y matrix
  y_train=[]
  for line in train:
    y_train.append([line[1]])
  print(f"y_train={len(y_train)}*{len(y_train[0])}")

  #W matrix w=(Xt X)-1 Xty
  Xt=transpose(x_train_poly)
  XtX=multiply(Xt,x_train_poly)
  inv=inverse(XtX)
  W=multiply(multiply(inv,Xt),y_train)
  return W