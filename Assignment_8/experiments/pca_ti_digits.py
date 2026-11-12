
import numpy as np
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.results_logger import save_results
from preprocessing.data_loader import DataLoader
from preprocessing.pca import PCA

from classifiers.knn import KNN
from classifiers.lda import LDA
from classifiers.gmm import GMMClassifier
from classifiers.svm import SVM


loader = DataLoader()

data= loader.load_ti_digits()
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

print("TI DIGITS DATASET")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

dimensions = [10, 20, 40, 60, 75]

for k in dimensions:

    print("\n================================")
    print("PCA Components:", k)
    print("================================")

    pca = PCA(n_components=k)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    variance = np.sum(pca.explained_variance_ratio)

    print("Training shape:", X_train_pca.shape)
    print("Testing shape:", X_test_pca.shape)
    print("Explained variance:", round(variance * 100, 2), "%")

    # KNN
    knn = KNN(k=3)
    knn.fit(X_train_pca, y_train)
    knn_predictions = knn.predict(X_test_pca)
    knn_accuracy = np.mean(knn_predictions == y_test) * 100

    # LDA
    lda = LDA()
    lda.fit(X_train_pca, y_train)
    lda_predictions = lda.predict(X_test_pca)
    lda_accuracy = np.mean(lda_predictions == y_test) * 100

    # GMM
    gmm = GMMClassifier(n_components=2)
    gmm.fit(X_train_pca, y_train)
    gmm_predictions = gmm.predict(X_test_pca)
    gmm_accuracy = np.mean(gmm_predictions == y_test) * 100

    # SVM
    svm = SVM(
        learning_rate=0.01,
        regularization=0.001,
        epochs=30,
        batch_size=256
    )
    svm.fit(X_train_pca, y_train)
    svm_predictions = svm.predict(X_test_pca)
    svm_accuracy = np.mean(svm_predictions == y_test) * 100

    print("\nRESULTS")
    print("KNN:", round(knn_accuracy, 3), "%")
    print("LDA:", round(lda_accuracy, 3), "%")
    print("GMM:", round(gmm_accuracy, 3), "%")
    print("SVM:", round(svm_accuracy, 3), "%")
    save_results(
    dataset="TI Digits",
    experiment="PCA",
    components=k,
    explained_variance=round(
        np.sum(pca.explained_variance_ratio) * 100, 2
    ),
    results={
        "KNN": knn_accuracy,
        "LDA": lda_accuracy,
        "GMM": gmm_accuracy,
        "SVM": svm_accuracy
    }
    )
