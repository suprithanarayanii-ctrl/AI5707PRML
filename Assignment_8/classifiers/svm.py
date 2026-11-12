import numpy as np
class SVM:
    def __init__(self,learning_rate=0.01,regularization=1.0,epochs=1000,batch_size=256):
        self.learning_rate=learning_rate
        self.regularization=regularization
        self.epochs=epochs
        self.batch_size=batch_size
        self.classes=None
        self.weights=None
        self.bias=None
        self.alphas=[]
    def fit_binary(self,X,y):
        n_samples,n_features=X.shape
        C=self.regularization
        alpha=np.zeros(n_samples)
        w=np.zeros(n_features)
        b=0.0
        rng=np.random.default_rng(42)
        tol=1e-3
        for epoch in range(self.epochs):
            changed=0
            for i in rng.permutation(n_samples):
                Ei=np.dot(w,X[i])+b-y[i]
                yiEi=y[i]*Ei
                if not ((yiEi < -tol and alpha[i] < C) or (yiEi > tol and alpha[i] > 0)):
                    continue
                j=rng.integers(n_samples)
                if j==i:
                    j=(j+1)%n_samples
                Ej=np.dot(w,X[j])+b-y[j]
                old_ai=alpha[i]
                old_aj=alpha[j]
                if y[i]!=y[j]:
                    L=max(0,old_aj-old_ai)
                    H=min(C,C+old_aj-old_ai)
                else:
                    L=max(0,old_ai+old_aj-C)
                    H=min(C,old_ai+old_aj)
                if L==H:
                    continue
                Kii=np.dot(X[i],X[i])
                Kjj=np.dot(X[j],X[j])
                Kij=np.dot(X[i],X[j])
                eta=2*Kij-Kii-Kjj
                if eta>=0:
                    continue
                alpha[j]-=y[j]*(Ei-Ej)/eta
                alpha[j]=np.clip(alpha[j],L,H)
                if abs(alpha[j]-old_aj)<1e-5:
                    alpha[j]=old_aj
                    continue
                alpha[i]+=y[i]*y[j]*(old_aj-alpha[j])
                b1=b-Ei-y[i]*(alpha[i]-old_ai)*Kii-y[j]*(alpha[j]-old_aj)*Kij
                b2=b-Ej-y[i]*(alpha[i]-old_ai)*Kij-y[j]*(alpha[j]-old_aj)*Kjj
                if 0<alpha[i]<C:
                    b=b1
                elif 0<alpha[j]<C:
                    b=b2
                else:
                    b=(b1+b2)/2
                w+=(alpha[i]-old_ai)*y[i]*X[i]+(alpha[j]-old_aj)*y[j]*X[j]
                changed+=1
            if changed==0:
                break
        return w,b,alpha
    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y)
        self.classes=np.unique(y)
        n_classes=len(self.classes)
        n_features=X.shape[1]
        self.weights=np.zeros((n_classes,n_features))
        self.bias=np.zeros(n_classes)
        self.alphas=[]
        for class_index,class_label in enumerate(self.classes):
            binary_y=np.where(y==class_label,1.0,-1.0)
            w,b,alpha=self.fit_binary(X,binary_y)
            self.weights[class_index]=w
            self.bias[class_index]=b
            self.alphas.append(alpha)
        return self
    def predict(self,X):
        X=np.asarray(X,dtype=np.float64)
        scores=X@self.weights.T+self.bias
        predicted_indices=np.argmax(scores,axis=1)
        return self.classes[predicted_indices]
