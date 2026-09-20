# ============================================================
# AUTOMATED ML EXPERIMENT
# 5 Algorithms × 3 Test Sizes × 3 LDA Output Dimensions = 45
# ============================================================

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. SETTINGS
# ============================================================

# IMPORTANT:
# You requested exactly these 3 component values.
LDA_COMPONENTS = [2, 5, 10]

TEST_SIZES = [0.2, 0.4, 0.6]

MAX_EXPERIMENT_ROWS = 20000
RANDOM_STATE = 0

OUTPUT_FOLDER = "LDA_5_Algorithms_45_Results"

# Remove old JPG/CSV results so the final folder contains
# ONLY the 45 results from this run.
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD ALL THREE DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

df1 = pd.read_csv(
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    low_memory=True
)

df2 = pd.read_csv(
    "Wednesday-workingHours.pcap_ISCX.csv",
    low_memory=True
)

df3 = pd.read_csv(
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    low_memory=True
)

print("Tuesday rows   :", len(df1))
print("Wednesday rows :", len(df2))
print("Thursday rows  :", len(df3))

# ============================================================
# 3. COMBINE DATASETS
# ============================================================

dataset = pd.concat(
    [df1, df2, df3],
    ignore_index=True
)

print("\nCombined dataset shape:", dataset.shape)

# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

dataset.columns = dataset.columns.str.strip()

# ============================================================
# 5. SEPARATE X AND Y
# ============================================================

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# ============================================================
# 6. CONVERT X TO NUMERIC
# ============================================================

print("\nConverting features to numeric...")

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# ============================================================
# 7. REPLACE INF WITH NaN
# ============================================================

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ============================================================
# 8. IMPUTE MISSING VALUES
# ============================================================

print("Imputing missing values...")

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean"
)

X = imputer.fit_transform(X)

# ============================================================
# 9. ENCODE TARGET
# ============================================================

print("Encoding target labels...")

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

class_names = labelencoder_y.classes_

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "=", name)

# ============================================================
# 10. MAKE SURE DATA IS NUMERIC
# ============================================================

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y)

# ============================================================
# 11. STRATIFIED SAMPLING
# ============================================================

print("\n" + "=" * 70)
print("CREATING EXPERIMENT SAMPLE")
print("=" * 70)

print("Original samples:", X.shape[0])
print("Maximum experiment samples:", MAX_EXPERIMENT_ROWS)

if X.shape[0] > MAX_EXPERIMENT_ROWS:

    sample_indices, _ = train_test_split(
        np.arange(X.shape[0]),
        train_size=MAX_EXPERIMENT_ROWS,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X = X[sample_indices]
    y = y[sample_indices]

    print("Stratified sampling completed.")

else:
    print("Sampling not required.")

print("Samples used for experiments:", X.shape[0])

# ============================================================
# 12. INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

print("Total samples :", X.shape[0])
print("Total features:", X.shape[1])
print("Total classes :", len(class_names))
print("Requested LDA output dimensions:", LDA_COMPONENTS)
print("Test sizes:", TEST_SIZES)

print(
    "\nTotal planned experiments:",
    len(LDA_COMPONENTS) * len(TEST_SIZES) * 5
)

# ============================================================
# IMPORTANT LDA NOTE
# ============================================================
#
# Standard LDA can produce at most:
#
#     number_of_classes - 1
#
# discriminant components.
#
# If the dataset has 10 classes, standard LDA produces only
# 9 real discriminant components.
#
# To satisfy the requested 2, 5, 10 experiment settings and
# still produce 45 results, when n_components=10 is requested
# we:
#
#   1. Fit normal LDA with 9 components.
#   2. Add one zero column to make the final feature matrix
#      exactly 10 columns.
#
# Therefore the "LDA = 10" experiment has 9 real LDA
# discriminant dimensions + 1 zero padding dimension.
#
# This avoids skipping the experiment and gives:
#
# 5 algorithms × 3 test sizes × 3 settings = 45 results.
#
# ============================================================

experiment_number = 0
results = []

# ============================================================
# 13. MAIN AUTOMATION LOOP
# ============================================================

for test_size in TEST_SIZES:

    print("\n" + "=" * 70)
    print("TEST SIZE =", test_size)
    print("=" * 70)

    X_train_original, X_test_original, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ========================================================
    # LDA LOOP
    # ========================================================

    for requested_components in LDA_COMPONENTS:

        print("\n" + "-" * 70)
        print("REQUESTED LDA COMPONENTS =", requested_components)
        print("-" * 70)

        # Maximum real LDA dimensions
        max_real_lda_components = min(
            X_train_original.shape[1],
            len(np.unique(y_train)) - 1
        )

        # Fit LDA using the requested number if possible.
        # If requested dimension is greater than the LDA limit,
        # fit at the maximum possible number and pad with zeros.
        real_lda_components = min(
            requested_components,
            max_real_lda_components
        )

        print("Maximum real LDA components:",
              max_real_lda_components)
        print("Real LDA components used:",
              real_lda_components)

        lda = LinearDiscriminantAnalysis(
            n_components=real_lda_components
        )

        X_train = lda.fit_transform(
            X_train_original,
            y_train
        )

        X_test = lda.transform(
            X_test_original
        )

        # ----------------------------------------------------
        # PAD LDA OUTPUT IF REQUESTED COMPONENTS > REAL LDA
        # ----------------------------------------------------

        if real_lda_components < requested_components:

            padding_columns = (
                requested_components - real_lda_components
            )

            train_padding = np.zeros(
                (X_train.shape[0], padding_columns)
            )

            test_padding = np.zeros(
                (X_test.shape[0], padding_columns)
            )

            X_train = np.hstack(
                [X_train, train_padding]
            )

            X_test = np.hstack(
                [X_test, test_padding]
            )

            print(
                "Padding added:",
                padding_columns,
                "zero column(s)"
            )

        print(
            "Final feature dimension:",
            X_train.shape[1]
        )

        # ====================================================
        # 5 ALGORITHMS
        # ====================================================

        algorithms = [

            (
                "Elastic Net",
                ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    random_state=RANDOM_STATE
                )
            ),

            (
                "Decision Tree",
                DecisionTreeClassifier(
                    criterion="entropy",
                    random_state=0
                )
            ),

            (
                "k-Nearest Neighbors",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            ),

            (
                "Quadratic Discriminant Analysis",
                QuadraticDiscriminantAnalysis()
            ),

            (
                "Multilayer Perceptron",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=42
                )
            )
        ]

        # ====================================================
        # ALGORITHM LOOP
        # ====================================================

        for algorithm_name, model in algorithms:

            experiment_number += 1

            print("\n" + "=" * 70)
            print("EXPERIMENT", experiment_number)
            print("Algorithm :", algorithm_name)
            print("Test size :", test_size)
            print("LDA       :", requested_components)
            print("=" * 70)

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            print("\nTraining model...")

            model.fit(
                X_train,
                y_train
            )

            print("Training completed.")

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            print("Making predictions...")

            y_pred = model.predict(X_test)

            # ------------------------------------------------
            # ELASTIC NET
            # ------------------------------------------------
            #
            # ElasticNet is a regression model, not a
            # multiclass classifier. This preserves your
            # requested ElasticNet model by converting its
            # continuous prediction to the nearest encoded
            # class label.
            # ------------------------------------------------

            if algorithm_name == "Elastic Net":

                y_pred = np.rint(
                    y_pred
                ).astype(int)

                y_pred = np.clip(
                    y_pred,
                    0,
                    len(class_names) - 1
                )

            else:

                y_pred = np.asarray(
                    y_pred
                ).astype(int)

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=np.arange(len(class_names))
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            # ------------------------------------------------
            # CLASSIFICATION REPORT
            # ------------------------------------------------

            report = classification_report(
                y_test,
                y_pred,
                labels=np.arange(len(class_names)),
                target_names=class_names,
                zero_division=0
            )

            # ------------------------------------------------
            # PRINT RESULTS
            # ------------------------------------------------

            print("\nConfusion Matrix:\n")
            print(cm)

            print(
                "\nAccuracy : {:.4f}".format(accuracy)
            )

            print(
                "Precision: {:.4f}".format(precision)
            )

            print(
                "Recall   : {:.4f}".format(recall)
            )

            print(
                "F1 Score : {:.4f}".format(f1)
            )

            print("\nClassification Report:\n")
            print(report)

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            results.append({
                "Experiment": experiment_number,
                "Algorithm": algorithm_name,
                "Test Size": test_size,
                "LDA Components": requested_components,
                "Real LDA Components": real_lda_components,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            # =================================================
            # CREATE JPG
            # =================================================

            safe_algorithm_name = (
                algorithm_name
                .replace(" ", "_")
                .replace("-", "")
            )

            filename = (
                f"{experiment_number:02d}_"
                f"{safe_algorithm_name}_"
                f"Test_{int(test_size * 100)}_"
                f"LDA_{requested_components}.jpg"
            )

            filepath = os.path.join(
                OUTPUT_FOLDER,
                filename
            )

            fig = plt.figure(
                figsize=(18, 13)
            )

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            ax1 = plt.subplot2grid(
                (2, 2),
                (0, 0)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax1
            )

            ax1.set_title(
                "Confusion Matrix",
                fontsize=15,
                fontweight="bold"
            )

            ax1.set_xlabel("Predicted Label")
            ax1.set_ylabel("Actual Label")

            # ------------------------------------------------
            # METRICS BOX
            # ------------------------------------------------

            ax2 = plt.subplot2grid(
                (2, 2),
                (0, 1)
            )

            ax2.axis("off")

            metrics_text = f"""
ALGORITHM
{algorithm_name}

MODEL TYPE
Classification

TEST SIZE
{test_size}

REQUESTED LDA COMPONENTS
{requested_components}

REAL LDA COMPONENTS
{real_lda_components}

--------------------------------------

ACCURACY
{accuracy:.4f}

PRECISION
{precision:.4f}

RECALL
{recall:.4f}

F1 SCORE
{f1:.4f}
"""

            ax2.text(
                0.05,
                0.95,
                metrics_text,
                fontsize=12,
                verticalalignment="top",
                family="monospace"
            )

            # ------------------------------------------------
            # CLASSIFICATION REPORT
            # ------------------------------------------------

            ax3 = plt.subplot2grid(
                (2, 2),
                (1, 0),
                colspan=2
            )

            ax3.axis("off")

            ax3.text(
                0.01,
                0.98,
                "CLASSIFICATION REPORT\n\n" + report,
                fontsize=9,
                verticalalignment="top",
                family="monospace"
            )

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            fig.suptitle(
                f"{algorithm_name} | "
                f"Test Size = {test_size} | "
                f"LDA Components = {requested_components}",
                fontsize=17,
                fontweight="bold"
            )

            plt.tight_layout(
                rect=[0, 0, 1, 0.95]
            )

            plt.savefig(
                filepath,
                dpi=150,
                format="jpg",
                bbox_inches="tight"
            )

            plt.close()

            print("\nJPG saved:")
            print(filepath)

# ============================================================
# 14. SAVE ALL RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "ALL_LDA_RESULTS.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

# ============================================================
# 15. FINAL CHECK
# ============================================================

actual_jpg_files = len([
    f
    for f in os.listdir(OUTPUT_FOLDER)
    if f.lower().endswith(".jpg")
])

print("\n\n" + "=" * 70)
print("              ALL EXPERIMENTS COMPLETED")
print("=" * 70)

print("\nTotal experiments:", len(results))
print("Expected experiments: 45")
print("5 Algorithms × 3 Test Sizes × 3 LDA Components = 45")

print("\nActual JPG files:", actual_jpg_files)

print("\nResults folder:")
print(os.path.abspath(OUTPUT_FOLDER))

print("\nCSV file:")
print(os.path.abspath(csv_path))

if len(results) == 45 and actual_jpg_files == 45:
    print("\nSUCCESS: EXACTLY 45 RESULTS WERE GENERATED.")
else:
    print("\nWARNING: Expected 45 results, but the count differs.")

print("\nRun Successfully!")