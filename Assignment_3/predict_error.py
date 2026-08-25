from matrix_operations import multiply

def predict_error(degree,data,W):
  x_data=[]
  for line in data:
    x_data.append(line[0])

  # X matrix
  x_poly=[]
  for x in x_data:
    value=[]
    for power in range(degree+1):
      value.append(x**power)
    x_poly.append(value)
  print(f"x_poly={len(x_poly)}*{len(x_poly[0])}")

  # Predicted y values
  y_hat=multiply(x_poly,W)

  #Actual y values
  y_data=[]
  for line in data:
    y_data.append([line[1]])
  print(f"size of y_hat={len(y_hat)}*{len(y_hat[0])}")
  print(f"size of y_data={len(y_data)}*{len(y_data[0])}")

  # error
  SSE=0
  for i in range(len(y_data)):
    error=y_data[i][0]-y_hat[i][0]
    squared_error=error**2
    SSE+=squared_error
  MSE=SSE/len(y_data)
  return MSE
