import numpy as np

class KNN:

  def __init__(self,k=3,batch_size=64):
    self.k=k
    self.batch_size=batch_size
    self.X_train=None
    self.y_train=None

  def fit(self,X,y):
    self.X_train=np.asarray(X,dtype=np.float64)
    self.y_train=np.asarray(y)

    if self.k<1 or self.k>len(self.X_train):
      raise ValueError("Invalid value of k")
    return self

  def predict(self,X):
    X=np.asarray(X,dtype=np.float64)
    predictions=[]
    train_norms=np.sum(self.X_train**2,axis=1)
    for start in range(0,len(X),self.batch_size):
        X_batch=X[start:start+self.batch_size]
        test_norms=np.sum(X_batch**2,axis=1)
        distances=(test_norms[:,None]+train_norms[None,:]-2*np.dot(X_batch,self.X_train.T))
        distances=np.maximum(distances,0)
        nearest_indices=np.argpartition(distances,self.k-1,axis=1)[:,:self.k]
        for indices in nearest_indices:
          neighbor_labels=self.y_train[indices]
          classes,counts=np.unique(neighbor_labels,return_counts=True)
          predicted_label=classes[np.argmax(counts)]
          predictions.append(predicted_label)
    return np.asarray(predictions)