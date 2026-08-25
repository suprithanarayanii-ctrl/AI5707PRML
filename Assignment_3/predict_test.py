from matrix_operations import multiply

def predict_test(degree,test,W):
  x_test=[]
  for line in test:
    x_test.append(line[0])

  # X matrix
  x_test_poly=[]
  for x in x_test:
    value=[]
    for power in range(degree+1):
      value.append(x**power)
    x_test_poly.append(value)
  print(f"x_test_poly={len(x_test_poly)}*{len(x_test_poly[0])}")

  # Predicted y values
  y_hat=multiply(x_test_poly,W)

  #Actual y values
  y_test=[]
  for line in test:
    y_test.append([line[1]])
  print(f"size of y_hat={len(y_hat)}*{len(y_hat[0])}")
  print(f"size of y_test={len(y_test)}*{len(y_test[0])}")

  # error
  SSE=0
  for i in range(len(y_test)):
    error=y_test[i][0]-y_hat[i][0]
    squared_error=error**2
    SSE+=squared_error
  MSE=SSE/len(y_test)
  return MSE
