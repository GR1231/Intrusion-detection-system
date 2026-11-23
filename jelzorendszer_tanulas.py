import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT_DIR = Path("tanulas_eredmenyek")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_SIZE = 0.20
RANDOM_STATE = 42

#Betoltes
def load_dataset(path):
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fajl nem talalhato: {file_path.resolve()}\n")

    print("Adathalmaz betoltese...")

    #Adat elofeldolgozasa
    df = pd.read_csv(file_path, low_memory=False)

    df.columns = df.columns.str.strip() #szokoz eltavolitas

    df = df.loc[:, ~df.columns.duplicated()].copy() #duplikalas eltavolitasa

    label_columns = [
        col for col in df.columns
        if col.lower() == "label"
    ]

    if not label_columns:
        raise ValueError(
            "Kell 'Label' nevu oszlop az adathalmazban."
        )

    label_column = label_columns[0]
    labels = (df[label_column].astype("string").str.strip().str.upper()) #label oszlop normalizalasa

    valid_labels = labels.notna() & labels.ne("")

    df = df.loc[valid_labels].copy()
    labels = labels.loc[valid_labels]

    y = (labels != "BENIGN").astype("int8") #benign - normal, nem benign - tamadas

    #Jellemzo oszlopok elofeldolgozasa
    X = df.drop(columns=[label_column])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("Nem talalhato szam alapu jellemzo.")

    if y.nunique() != 2:
        raise ValueError(
            "Adathalmaznak tartalmaznia kell normal es tamado forgalmat"
        )

    print(f"Adathalmaz alak: {df.shape}")
    print(f"Jellemzok szama: {X.shape[1]}")
    print(f"Normal mintak: {(y == 0).sum():,}")
    print(f"Tamadas mintak: {(y == 1).sum():,}")

    return X, y


#Tanitas teszteles aranya
def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


#Kulonbozo modellek
def create_models():
    naive_bayes = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB()),
    ])

    decision_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_depth=20,
            min_samples_leaf=2,
        )),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            max_depth=25,
            min_samples_leaf=2,
        )),
    ])

    return {
        "Naive Bayes": naive_bayes,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }


#Tanitas
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n{name} tanitasa...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Modell": name,
            "Helyesseg": accuracy,
            "Pontossag": precision,
            "Recall": recall,
            "F1-Score": f1,
        })

        trained_models[name] = model

        print(f"Helyesseg:  {accuracy:.4f}")
        print(f"Pontossag: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        print("\nKlasszifikacio report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Tamadas"],
            zero_division=0,
        ))

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Tamadas"],
            yticklabels=["Normal", "Tamadas"],
        )
        plt.xlabel("Elvart jellemzo")
        plt.ylabel("Aktualis jellemzo")
        plt.title(f"Konfuzios matrix - {name}")
        plt.tight_layout()

        filename = name.lower().replace(" ", "_")
        plt.savefig(OUTPUT_DIR / f"konfuzios_matrix_{filename}.png", dpi=150)
        plt.close()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Helyesseg", ascending=False
    ).reset_index(drop=True)

    return results_df, trained_models


#Osszehasonlitas
def compare_models(results_df):
    print("\n")
    print("Ideiglenes osszehasonlitas")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "Helyesseg": "{:.4f}".format,
                "Pontossag": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-Score": "{:.4f}".format,
            },
        )
    )

    results_df.to_csv(OUTPUT_DIR / "modell_hasonlitas.csv", index=False)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=results_df,
        x="Modell",
        y="Helyesseg",
        hue="Modell",
        palette="viridis",
        legend=False,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Helyesseg")
    ax.set_xlabel("Modell")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "modell_helyesseg_hasonl.png", dpi=150)
    plt.close()


#Modellek mentese
def save_models(models):
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        output = OUTPUT_DIR / filename
        joblib.dump(model, output)
        print(f"Modell mentve: {filename}")


def main():
    X, y = load_dataset(DATASET_PATH)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print(f"\nTanulasi mintak: {len(X_train):,}")
    print(f"Teszteles mintak:  {len(X_test):,}")

    models = create_models()

    results, trained_models = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    compare_models(results)
    #save_models(trained_models)

    print("\nProgram befejezve.")


if __name__ == "__main__":
    main()
