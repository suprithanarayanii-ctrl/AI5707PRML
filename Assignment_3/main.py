import random
from train_degree import train_degree
from predict_error import predict_error
from predict_test import predict_test
from plot import mseplot
from plot import curveplot

# reading the file 
file=open('noisy_5.txt','r');
x_list=[]
y_list=[]
for line in file:
  parts=line.split()
  x=float(parts[0])
  y=float(parts[1])
  x_list.append(x)
  y_list.append(y)

print(f"size of x = {len(x_list)}")
print(f"size of y = {len(y_list)}")

# create (x,y) pairs
data=[]
for i in range(len(x_list)):
  data.append([x_list[i],y_list[i]])

# shuffle the data
random.shuffle(data)

# split data
train = data[:6000]
validation = data[6000:8000]
test = data[8000:]
print(f"--------------------SPLIT DATASET--------------------------------")
print(f"size of train={len(train)}")
print(f"size of validation={len(validation)}")
print(f"size of test={len(test)}")

#Learn W
drange=10
train_results=[]
valid_results=[]
W_results=[]
for i in range(1,drange+1):
  print(f"---------------------DEGREE {i}------------------------------------")
  print(f"------LEARN W ON TRAIN-------------")
  W=train_degree(i,train)
  W_results.append(W)
  print(f"size of w={len(W)}*{len(W[0])}")
  # Predict Training data for showing overfitting
  MSE=predict_error(i,train,W)
  train_results.append([i,MSE])
  #Predict Validation
  print(f"------PREDICT ON VALIDATION---------")
  MSE=predict_error(i,validation,W)
  print(f"MSE={MSE}")
  valid_results.append([i,MSE])

curveplot(W_results,train)

# Effect of Poly degree
print(f"-----------------------EFFECT POLY DEGREE----------------------------------")
print(f"Degree \t MSE")
for i in range(drange):
  print(f"{i+1}\t{valid_results[i][1]}")
mseplot(valid_results,train_results)

# Best Degree
best_degree=valid_results[0][0]
min_mse=valid_results[0][1]
for mse in valid_results:
  if mse[1]<min_mse:
    best_degree=mse[0]
    min_mse=mse[1]
print(f"-------------------------BEST DEGREE-----------------------------")
print(f"Best Degree = {best_degree}")
print(f"Minimum validation MSE = {min_mse}")
#Predict Test
print(f"--------------------------TEST DATA------------------------------------")
W=W_results[best_degree-1]
test_MSE=predict_test(best_degree,test,W)
print(f"Degree={best_degree} \t Test MSE={test_MSE}")

file.close()