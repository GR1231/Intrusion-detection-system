import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
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
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET_DIR = Path("Bemeneti-adathalmaz")
OUTPUT_DIR = Path("tanulas_eredmenyek")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_SIZE = 0.20
RANDOM_STATE = 42

#Betoltes
def load_dataset(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Adathalmaz mappa nem talalhato: {folder_path.resolve()}")

    #Osszefuzott adathalmaz helye
    merged_path = folder_path / "Bemeneti-adat-osszes.csv"

    #letezik-e mar az osszefuzott fajl
    if merged_path.exists():

        print("Meglevo osszefuzott csv betoltese...")
        df = pd.read_csv(merged_path, low_memory=False)

    else:

        print("CSV fajlok osszefuzese...")

        csv_files = sorted(
            file for file in folder_path.glob("*.csv")
            if file.name != merged_path.name
        )

        if not csv_files:
            raise FileNotFoundError("Nem talalhato CSV fajl a bemeneti mappaban.")

        dataframes = []

        for file in csv_files:

            df_part = pd.read_csv(file, low_memory=False)

            df_part.columns = df_part.columns.str.strip()
            df_part = df_part.loc[:, ~df_part.columns.duplicated()]

            dataframes.append(df_part)
            reference_columns = set(dataframes[0].columns)

            for i, df_part in enumerate(dataframes[1:], start=2):
                if set(df_part.columns) != reference_columns:
                    print(f"!!!Kulonbozo oszlopok talalhatoak CSV {i}-ben!!!")

        # Osszes CSV osszefuzese
        df = pd.concat(dataframes, ignore_index=True, sort=False)

        del dataframes

        # Oszlopnevek normalizalasa
        df.columns = df.columns.str.strip()

        # Duplikalt oszlopok eltavolitasa
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Osszefuzott fajl mentese
        df.to_csv(merged_path, index=False)

        print(f"Osszefuzott adathalmaz mentve: {merged_path}")

    # Oszlopnevek normalizalasa
    df.columns = df.columns.str.strip()

    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Label oszlop megkeresese
    label_columns = [
        col for col in df.columns
        if col.lower() == "label"
    ]

    if not label_columns:
        raise ValueError("Kell 'Label' nevu oszlop az adathalmazban.")

    label_column = label_columns[0]

    # Label normalizalasa
    labels = (
        df[label_column]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    valid_labels = labels.notna() & labels.ne("")

    df = df.loc[valid_labels].copy()
    labels = labels.loc[valid_labels]

    # BENIGN = 0, ATTACK = 1
    y = (labels != "BENIGN").astype("int8")

    # Jellemzok elofeldolgozasa
    X = df.drop(columns=[label_column])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("Nem talalhato szam alapu jellemzo.")

    if y.nunique() != 2:
        raise ValueError("Adathalmaznak tartalmaznia kell normal es tamado forgalmat.")

    print("\n========== ADATHALMAZ INFORMACIOK ==========")
    print(f"Adathalmaz alakja: {df.shape}")
    print(f"Jellemzok szama: {X.shape[1]}")
    print(f"Normal mintak: {(y == 0).sum():,}")
    print(f"Tamadas mintak: {(y == 1).sum():,}")
    print("============================================")

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
        ("scaler", MinMaxScaler(feature_range=(0, 1))),
        ("classifier", GaussianNB()),
    ])

    decision_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler(feature_range=(0, 1))), #Nem muszaj ennel a modellnel(data scaling [0 1])
        ("classifier", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_depth=20,
            min_samples_leaf=2,
        )),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler(feature_range=(0, 1))), #Nem muszaj ennel a modellnel(data scaling [0 1])
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
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, probabilities)

        results.append({
            "Modell": name,
            "Helyesseg": accuracy,
            "Pontossag": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        trained_models[name] = model

        print(f"Helyesseg:  {accuracy:.4f}")
        print(f"Pontossag: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")

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
    print("Modellek osszehasonlitasa")
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

    #results_df.to_csv(OUTPUT_DIR / "modell_hasonlitas.csv", index=False)

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


def undersample(X_train,y_train):
    undersampler = RandomUnderSampler(sampling_strategy="auto", random_state=RANDOM_STATE)
    X, y = undersampler.fit_resample(X_train, y_train)

    return X, y

#Modellek mentese
def save_models(models):
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        output = OUTPUT_DIR / filename
        joblib.dump(model, output)
        print(f"Modell mentve: {filename}")


def main():
    X, y = load_dataset(DATASET_DIR)

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"\nTanulasi mintak undersampling elott: {len(X_train):,}")
    print(f"Teszteles mintak: {len(X_test):,}")

    X_train, y_train = undersample(X_train,y_train)

    print("\nTanulasi adatok alulmintazas utan:")
    print(f"\nTanulasi mintak: {len(X_train):,}")
    print(f"Teszteles mintak:  {len(X_test):,}")
    print(f"Normal mintak: {(y_train == 0).sum():,}")
    print(f"Tamadas mintak: {(y_train == 1).sum():,}")

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
