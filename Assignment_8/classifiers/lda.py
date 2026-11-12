import numpy as np

class LDA:

  def __init__(self,regularization=1e-6):
    self.regularization=regularization
    self.classes=None
    self.means=None
    self.weights=None
    self.bias=None

  def fit(self,X,y):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y)

    self.classes,counts=np.unique(y,return_counts=True)
    n_samples,n_features=X.shape
    n_classes=len(self.classes)
    self.means=np.zeros((n_classes,n_features))
    within_scatter=np.zeros((n_features,n_features))
    for i,class_label in enumerate(self.classes):
      X_class=X[y==class_label]
      self.means[i]=np.mean(X_class,axis=0)
      centered=X_class-self.means[i]
      within_scatter+=centered.T@centered
    covariance=within_scatter/(n_samples-n_classes)
    scale=np.trace(covariance)/n_features
    covariance+=(self.regularization*scale*np.eye(n_features))
    self.weights=np.linalg.solve(covariance,self.means.T).T
    priors=counts/n_samples
    self.bias=(-0.5*np.sum(self.weights*self.means,axis=1)+np.log(priors))
    return self

  def predict(self,X):
    X=np.asarray(X,dtype=np.float64)
    scores=X@self.weights.T+self.bias
    predicted_indices=np.argmax(scores,axis=1)
    return self.classes[predicted_indices]