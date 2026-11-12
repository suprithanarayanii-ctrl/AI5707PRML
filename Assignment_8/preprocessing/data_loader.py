import os
import pickle
import numpy as np
import matplotlib.image as mpimg

class DataLoader:
  # Load all 4 datasets and convert them into suitable type
  def __init__(self,data_dir="datasets",random_state=42):
    self.data_dir=data_dir
    self.random_state=random_state

  # Common random Train test split
    # Common random Train test split
  def random_split(self, X, y, test_ratio=0.2, random_state=None):
    if random_state is None:
      random_state = self.random_state

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    for label in np.unique(y):
      indices = np.where(y == label)[0]
      indices = rng.permutation(indices)

      test_count = int(len(indices) * test_ratio)

      train_indices.extend(indices[test_count:])
      test_indices.extend(indices[:test_count])

    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)

    return (
      X[train_indices],
      X[test_indices],
      y[train_indices],
      y[test_indices]
    )
  # Synthetic Dataset
  def load_synthetic(self):
    file_path=os.path.join(self.data_dir,"synthetic","team5","data.pkl")
    with open(file_path,"rb") as file:
      data=pickle.load(file)
    X=np.asarray(data["x"],dtype=np.float64)
    y=np.asarray(data["y"],dtype=np.int64)

    # validate feature dim
    assert X.ndim==2 and X.shape[1]==16
    assert len(X)==len(y)

    X_train,X_test,y_train,y_test=self.random_split(X,y)
    return {
      "X_train":X_train,
      "y_train":y_train,
      "X_test":X_test,
      "y_test":y_test
    }

  # MNIST Dataset
  def read_image(self,image_path):
    image=mpimg.imread(image_path)

    # convert RGB to Greyscale
    if image.ndim==3:
      if image.shape[2]==4:
        image=image[:,:,:3]
      image=(0.299*image[:,:,0]+0.587*image[:,:,1]+0.114*image[:,:,2])

    # Flattern
    image=image.reshape(-1)
    return image.astype(np.float64)

  def load_mnist_split(self,split_name):
    X=[]
    y=[]
    split_path=os.path.join(self.data_dir,"mnist",split_name)
    classes=[0,1,3]
    for label in classes:
      class_path=os.path.join(split_path,str(label))
      if not os.path.isdir(class_path):
        raise FileNotFoundError(f"Missing directory: {class_path}")
      image_files=sorted(os.listdir(class_path))
      for filename in image_files:
        if not filename.lower().endswith((".png",".jpg",".jpeg",".bmp")):
          continue
        image_path=os.path.join(class_path,filename)
        image_vector=self.read_image(image_path)
        X.append(image_vector)
        y.append(label)
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.int64)
    return X,y

  def load_mnist(self):
    X_train,y_train=self.load_mnist_split("train")
    X_val,y_val=self.load_mnist_split("dev")
    X_test,y_test=self.load_mnist_split("test")

    # all img same dim
    if not(X_train.shape[1]==X_val.shape[1]==X_test.shape[1]):
      raise ValueError("MNIST image dimensions do nt match")
    return {
      "X_train":X_train,
      "y_train":y_train,
      "X_val":X_val,
      "y_val":y_val,
      "X_test":X_test,
      "y_test":y_test
    }
  
  # TI Digits Dataset
  def extract_sequence_features(self,sequence):
    sequence=np.asarray(sequence,dtype=np.float64)
    # each frame has 39 MFCC features
    if sequence.ndim!=2 or sequence.shape[1]!=39:
      raise ValueError(f"Expected sequence shape (T,39), "f"received {sequence.shape}")
    
    mean_features=np.mean(sequence,axis=0)
    std_features=np.std(sequence,axis=0)
    features=np.concatenate([mean_features,std_features])
    return features

  def load_ti_digits(self):
    digit_path=os.path.join(self.data_dir,"ti_digits","ExtractedData")
    assigned_digits=[2,6,7]
    X=[]
    y=[]
    for digit in assigned_digits:
      file_path=os.path.join(digit_path,f"{digit}.npz")
      with np.load(file_path,allow_pickle=True) as data:
        sequence_keys=sorted(data.files,key=lambda key:int(key.split("_")[1]))
        for key in sequence_keys:
          sequence=data[key]
          features=self.extract_sequence_features(sequence)
          X.append(features)
          y.append(digit)
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.int64)
    X_train,X_test,y_train,y_test=self.random_split(X,y)
    return {
      "X_train":X_train,
      "y_train":y_train,
      "X_test":X_test,
      "y_test":y_test
    }

  # img feature dataset
  def load_image_dataset(self):
    image_path=os.path.join(self.data_dir,"image")
    def read_csv(filename):
      path=os.path.join(image_path,filename)
      return np.loadtxt(path,delimiter=",",dtype=np.float64)
    X_train=read_csv("train_data.csv")
    y_train=read_csv("train_label.csv")
    X_val=read_csv("val_data.csv")
    y_val=read_csv("val_label.csv")
    X_test=read_csv("test_data.csv")
    y_test=read_csv("test_label.csv")

    y_train=y_train.reshape(-1).astype(np.int64)
    y_val=y_val.reshape(-1).astype(np.int64)
    y_test=y_test.reshape(-1).astype(np.int64)

    assert X_train.shape[1]==36
    assert X_val.shape[1]==36
    assert X_test.shape[1]==36

    return {
      "X_train":X_train,
      "y_train":y_train,
      "X_val":X_val,
      "y_val":y_val,
      "X_test":X_test,
      "y_test":y_test
    }