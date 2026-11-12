
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "results/experiment_results.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read CSV
with open(CSV_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

datasets = ["Synthetic", "MNIST", "TI Digits", "ImageFeature"]
classifiers = ["KNN", "LDA", "GMM", "SVM"]

for dataset in datasets:

    dataset_rows = [
        row for row in rows
        if row["Dataset"] == dataset
    ]

    baseline = {}
    pca_data = {}

    # Extract baseline and PCA results
    for row in dataset_rows:

        classifier = row["Classifier"]
        accuracy = float(row["Accuracy"])

        if row["Experiment"] == "Baseline":
            baseline[classifier] = accuracy

        elif row["Experiment"] == "PCA":
            k = int(row["Components"])

            if classifier not in pca_data:
                pca_data[classifier] = {}

            pca_data[classifier][k] = accuracy

    # Check that results are available
    if not baseline or not pca_data:
        print("Missing results for:", dataset)
        continue

    plt.figure(figsize=(10, 6))

    for classifier in classifiers:

        if classifier not in baseline:
            continue

        if classifier not in pca_data:
            continue

        components = sorted(pca_data[classifier].keys())

        accuracies = [
            pca_data[classifier][k]
            for k in components
        ]

        # PCA accuracy
        line, = plt.plot(
            components,
            accuracies,
            marker="o",
            linewidth=2,
            label=classifier + " - PCA"
        )

        # Baseline accuracy
        plt.axhline(
            y=baseline[classifier],
            linestyle="--",
            color=line.get_color(),
            alpha=0.7,
            label=classifier + " - Baseline"
        )

    plt.title(dataset + " - PCA Classification Accuracy")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Classification Accuracy (%)")

    plt.ylim(0, 105)
    plt.xticks(sorted(set(
        int(row["Components"])
        for row in dataset_rows
        if row["Experiment"] == "PCA"
    )))

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()

    filename = dataset.lower().replace(" ", "_") + "_pca_accuracy.png"

    save_path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Graph saved:", save_path)

print("\nAll graphs generated from CSV successfully.")
