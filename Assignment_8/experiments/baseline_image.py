
import numpy as np
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.results_logger import save_results
from preprocessing.data_loader import DataLoader

from classifiers.knn import KNN
from classifiers.lda import LDA
from classifiers.gmm import GMMClassifier
from classifiers.svm import SVM


loader = DataLoader()

data = loader.load_image_dataset()


X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

print("IMAGE FEATURE ORIGINAL FEATURES")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

knn = KNN(k=3)
knn.fit(X_train, y_train)
knn_accuracy = np.mean(knn.predict(X_test) == y_test) * 100

lda = LDA()
lda.fit(X_train, y_train)
lda_accuracy = np.mean(lda.predict(X_test) == y_test) * 100

gmm = GMMClassifier(n_components=2)
gmm.fit(X_train, y_train)
gmm_accuracy = np.mean(gmm.predict(X_test) == y_test) * 100

svm = SVM(
    learning_rate=0.01,
    regularization=0.001,
    epochs=30,
    batch_size=256
)
svm.fit(X_train, y_train)
svm_accuracy = np.mean(svm.predict(X_test) == y_test) * 100

print("\nIMAGE FEATURE BASELINE RESULTS")
print("KNN:", round(knn_accuracy, 3), "%")
print("LDA:", round(lda_accuracy, 3), "%")
print("GMM:", round(gmm_accuracy, 3), "%")
print("SVM:", round(svm_accuracy, 3), "%")
save_results(
    dataset="ImageFeature",
    experiment="Baseline",
    components="Original",
    explained_variance="",
    results={
        "KNN": knn_accuracy,
        "LDA": lda_accuracy,
        "GMM": gmm_accuracy,
        "SVM": svm_accuracy
    }
)