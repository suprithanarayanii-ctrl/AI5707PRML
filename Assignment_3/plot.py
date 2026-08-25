import matplotlib.pyplot as plt
from matrix_operations import multiply

def mseplot(valid_results,train_results):
  degree=[]
  mse_values_train=[] 
  for result in train_results:
    degree.append(result[0])
    mse_values_train.append(result[1])

  mse_values_vald=[] 
  for result in valid_results:
    mse_values_vald.append(result[1])

  plt.plot(degree,mse_values_train,marker='o',color="red",label="Training data MSE")
  plt.plot(degree,mse_values_vald,marker='o',label="Validation data MSE")
  plt.xlabel("Polynomial Degree")
  plt.ylabel("MSE")
  plt.title("Effect of Polynomial Degree on MSE")
  plt.legend()
  plt.grid()
  plt.show()

def curveplot(W_results,train):
  x_train=[]
  y_train=[]
  for line in train:
    x_train.append(line[0])
    y_train.append(line[1])

  x_sorted=sorted(x_train)
  drange=len(W_results)
  rows=(drange+4)//5
  fig, axes = plt.subplots(rows, 5, figsize=(20, 4*rows))
  if rows==1:
    axes=[axes]
  for i in range(drange):
    degree=i+1
    W=W_results[i]
    x_train_poly=[]
    for x in x_sorted:
      value=[]
      for power in range(degree+1):
        value.append(x**power)
      x_train_poly.append(value)

    # Predict y
    y_hat=multiply(x_train_poly,W)
    y_pred=[]
    for value in y_hat:
      y_pred.append(value[0])
    ax=axes[i//5][i%5]
    ax.scatter(x_train,y_train,s=2)
    ax.plot(x_sorted,y_pred,color="red")
    ax.set_title(f"Degree {degree}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()

  plt.tight_layout()
  plt.show()
    