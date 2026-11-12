
import os
import csv

RESULTS_DIR = "results"
CSV_FILE = os.path.join(RESULTS_DIR, "experiment_results.csv")

FIELDS = [
    "Dataset",
    "Experiment",
    "Components",
    "Explained_Variance",
    "Classifier",
    "Accuracy"
]


def save_results(dataset, experiment, components,
                 explained_variance, results):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Read existing results
    rows = []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

    # Remove previous entries for the same experiment
    # so rerunning a script updates its results
    rows = [
        row for row in rows
        if not (
            row["Dataset"] == dataset
            and row["Experiment"] == experiment
            and row["Components"] == str(components)
        )
    ]

    # Add new results
    for classifier, accuracy in results.items():

        rows.append({
            "Dataset": dataset,
            "Experiment": experiment,
            "Components": components,
            "Explained_Variance": explained_variance,
            "Classifier": classifier,
            "Accuracy": round(float(accuracy), 3)
        })

    # Write updated CSV
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Results saved to:", CSV_FILE)
