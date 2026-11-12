import numpy as np

class GMM:
  def __init__(self,n_components=2,max_iter=100,tolerance=1e-4):
    self.n_components=n_components
    self.max_iter=max_iter
    self.tolerance=tolerance
    self.means=None
    self.variances=None
    self.weights=None

  def log_gaussian(self,X):
    n_features=X.shape[1]
    difference=X[:,None,:]-self.means[None,:,:]
    log_det=np.sum(np.log(self.variances),axis=1)
    exponent=np.sum((difference**2)/self.variances[None,:,:],axis=2)
    log_probability=-0.5*(n_features*np.log(2*np.pi)+log_det[None,:]+exponent)
    return log_probability

  def e_step(self,X):
    log_probability=self.log_gaussian(X)
    log_probability+=np.log(self.weights+1e-12)

    maximum=np.max(log_probability,axis=1,keepdims=True)
    probability=np.exp(log_probability-maximum)
    responsibilities=probability/np.sum(probability,axis=1,keepdims=True)
    return responsibilities

  def m_step(self,X,responsibilities):
    n_samples,n_features=X.shape
    component_counts=np.sum(responsibilities,axis=0)
    component_counts=np.maximum(component_counts,1e-10)
    self.weights=component_counts/n_samples
    self.means=(responsibilities.T@X)/component_counts[:,None]
    for k in range(self.n_components):
      difference=X-self.means[k]
      self.variances[k]=np.sum(responsibilities[:,k,None]*difference**2,axis=0)/component_counts[k]
    self.variances+=1e-6

  def fit(self,X):
    X=np.asarray(X,dtype=np.float64)
    n_samples,n_features=X.shape
    if self.n_components>n_samples:
      raise ValueError("Too many components")
    np.random.seed(42)
    indices=np.random.choice(n_samples,self.n_components,replace=False)
    self.means=X[indices].copy()
    self.variances=np.zeros((self.n_components,n_features))
    global_variance=np.var(X,axis=0)+1e-6
    for k in range(self.n_components):
      self.variances[k]=global_variance
    self.weights=np.ones(self.n_components)/self.n_components
    previous_likelihood=-np.inf
    for iteration in range(self.max_iter):
      responsibilities=self.e_step(X)
      log_probability=self.log_gaussian(X)
      log_probability+=np.log(self.weights+1e-12)
      maximum=np.max(log_probability,axis=1,keepdims=True)
      likelihood=np.sum(maximum[:,0]+np.log(np.sum(np.exp(log_probability-maximum),axis=1)))
      # M step
      self.m_step(X,responsibilities)
      if iteration>0:
        if abs(likelihood-previous_likelihood)<self.tolerance:
          break
      previous_likelihood=likelihood
    return self

  def predict_log_probability(self,X):
    log_probability=self.log_gaussian(X)
    log_probability+=np.log(self.weights+1e-12)
    maximum=np.max(log_probability,axis=1,keepdims=True)
    return (maximum[:,0]+np.log(np.sum(np.exp(log_probability-maximum),axis=1)))

class GMMClassifier:
  def __init__(self,n_components=2,max_iter=100):
    self.n_components=n_components
    self.max_iter=max_iter
    self.classes=None
    self.models={}
    self.priors={}

  def fit(self,X,y):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y)
    self.classes,counts=np.unique(y,return_counts=True)
    for class_label,count in zip(self.classes,counts):
      X_class=X[y==class_label]
      model=GMM(n_components=self.n_components,max_iter=self.max_iter)
      model.fit(X_class)
      self.models[class_label]=model
      self.priors[class_label]=count/len(y)
    return self

  def predict(self,X):
    X=np.asarray(X,dtype=np.float64)
    scores=[]
    for class_label in self.classes:
      model=self.models[class_label]
      log_likelihood=model.predict_log_probability(X)
      score=(log_likelihood+np.log(self.priors[class_label]))
      scores.append(score)
    scores=np.column_stack(scores)
    predictions=np.argmax(scores,axis=1)
    return self.classes[predictions]
