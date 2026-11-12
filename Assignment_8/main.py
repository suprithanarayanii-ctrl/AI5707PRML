from preprocessing.data_loader import DataLoader
import numpy as np


def print_dataset_info(name,data):
  print("\n"+"#"*50)
  print(name)
  print("#"*50)

  for key,value in data.items():
    print(f"{key}:{value.shape}")
  print(f"Training classes: {np.unique(data["y_train"])}")
  print(f"Training samples: {len(data["y_train"])}")
  print(f"Testing samples: {len(data["y_test"])}")

def main():
  loader=DataLoader(data_dir="datasets",random_state=42)
  # 1. synthetic
  synthetic=loader.load_synthetic()
  print_dataset_info("SYNTHETIC DATASET",synthetic)


  # 2. MNIST
  mnist=loader.load_mnist()
  print_dataset_info("MNIST DAATSET",mnist)

  # 3. TI Digits
  ti_digits=loader.load_ti_digits()
  print_dataset_info("TI DIGITS DATASET",ti_digits)

  # 4. image feature dataset
  image=loader.load_image_dataset()
  print_dataset_info("IMAGE FEATURE DATASET",image)

  print("\n"+"#"*50)
  print("NORMALIZATION")
  print("#"*50)  
  
  datasets = {
    "Synthetic": synthetic["X_train"],
    "MNIST": mnist["X_train"],
    "TI Digits": ti_digits["X_train"],
    "Image": image["X_train"]
}

  for name, X in datasets.items():
      print(f"\n{name}")
      print("Shape:", X.shape)
      print("Mean:", np.mean(X))
      print("Std:", np.std(X))

if __name__ ==  "__main__":
  main()