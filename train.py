"""Training pipeline for the UCI SMS Spam Collection GRU classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATASET_DOI = "https://doi.org/10.24432/C5CC84"
DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_CITATION = (
    "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84. "
    "License: CC BY 4.0."
)
SEED = 42
CLASSIFICATION_THRESHOLD = 0.5
LABEL_TO_ID = {"ham": 0, "spam": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def load_dataset(data_path: str | Path | None = None, data_dir: str | Path = "data") -> pd.DataFrame:
    """Load UCI's two-column TSV, downloading the official archive if needed."""
    if data_path is not None:
        source_path = Path(data_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo indicado: {source_path}")
        return _read_tsv(source_path)

    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    extracted = directory / "SMSSpamCollection"
    if not extracted.exists():
        archive = directory / "sms-spam-collection.zip"
        try:
            urllib.request.urlretrieve(DATASET_ZIP_URL, archive)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo descargar el dataset de UCI. Descarga SMSSpamCollection "
                f"desde {DATASET_PAGE} y vuelve a ejecutar con --data-path <archivo>."
            ) from exc
        try:
            with zipfile.ZipFile(archive) as zipped:
                matches = [name for name in zipped.namelist() if Path(name).name == "SMSSpamCollection"]
                if not matches:
                    raise ValueError("El ZIP de UCI no contiene SMSSpamCollection.")
                extracted.write_bytes(zipped.read(matches[0]))
        except (zipfile.BadZipFile, KeyError) as exc:
            raise RuntimeError("El archivo descargado de UCI no es un ZIP válido.") from exc
    return _read_tsv(extracted)


def _read_tsv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        usecols=[0, 1],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    if list(frame.columns) != ["label", "message"]:
        raise ValueError("El dataset debe tener las columnas label y message.")
    labels = set(frame["label"].str.strip().unique())
    unexpected = labels - set(LABEL_TO_ID)
    if unexpected:
        raise ValueError(f"Etiquetas no reconocidas en UCI: {sorted(unexpected)}")
    if not labels:
        raise ValueError("El archivo UCI no contiene filas de datos.")
    return frame


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels and trim messages without removing lexical information."""
    required = {"label", "message"}
    if not required.issubset(frame.columns):
        raise ValueError("Se requieren las columnas label y message.")
    clean = frame.loc[:, ["label", "message"]].copy()
    clean["label"] = clean["label"].fillna("").astype(str).str.strip().str.lower()
    clean["message"] = clean["message"].fillna("").astype(str).str.strip()
    clean = clean.loc[clean["message"].ne("")].reset_index(drop=True)
    labels = set(clean["label"].unique())
    unexpected = labels - set(LABEL_TO_ID)
    if unexpected or labels != set(LABEL_TO_ID):
        raise ValueError(f"Se esperaban ham y spam; etiquetas observadas: {sorted(labels)}")
    clean["label_id"] = clean["label"].map(LABEL_TO_ID).astype("int32")
    clean["message_length"] = clean["message"].str.len()
    return clean


def split_dataset(frame: pd.DataFrame, seed: int = SEED):
    """Return stratified 80/10/10 train, validation, and test dataframes."""
    from sklearn.model_selection import train_test_split

    work = frame.copy()
    work["__split_id"] = np.arange(len(work))
    development, test = train_test_split(
        work, test_size=0.10, random_state=seed, stratify=work["label_id"]
    )
    train, validation = train_test_split(
        development,
        test_size=1 / 9,
        random_state=seed,
        stratify=development["label_id"],
    )
    partitions = [part.reset_index(drop=True) for part in (train, validation, test)]
    expected_labels = set(frame["label_id"].unique())
    for name, part in zip(("train", "validation", "test"), partitions):
        if set(part["label_id"].unique()) != expected_labels:
            raise ValueError(f"La partición {name} no conserva todas las clases.")
    row_ids = [set(part["__split_id"]) for part in partitions]
    if any(row_ids[i] & row_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Se detectó contaminación entre particiones.")
    sizes = [len(part) for part in partitions]
    if sum(sizes) != len(frame):
        raise ValueError("El número total de ejemplos cambió durante la partición.")
    ratios = {name: size / len(frame) for name, size in zip(("train", "validation", "test"), sizes)}
    print("Proporciones medidas de partición:", json.dumps(ratios, ensure_ascii=False))
    return tuple(part.drop(columns="__split_id") for part in partitions)


def explore_dataset(frame: pd.DataFrame, output_dir: str | Path = "reports") -> dict[str, Any]:
    """Print measured dataset summaries and save class/length plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = frame["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
    length_summary = frame.groupby("label")["message_length"].describe().round(2)
    summary = {
        "rows": int(len(frame)),
        "class_counts": {label: int(counts[label]) for label in counts.index},
        "message_length_by_class": {
            label: {key: float(value) for key, value in values.items()}
            for label, values in length_summary.to_dict(orient="index").items()
        },
    }
    print("Resumen medido del dataset:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=axes[0])
    axes[0].set(title="Mensajes por clase", xlabel="Etiqueta", ylabel="Cantidad")
    sns.boxplot(data=frame, x="label", y="message_length", hue="label", legend=False, ax=axes[1])
    axes[1].set(title="Longitud de mensaje por clase", xlabel="Etiqueta", ylabel="Caracteres")
    fig.tight_layout()
    fig.savefig(out / "dataset_exploration.png", dpi=160)
    plt.close(fig)
    print("Ejemplos (texto recortado para presentación):")
    for label in ("ham", "spam"):
        sample = frame.loc[frame["label"].eq(label), "message"].head(2)
        for message in sample:
            print(f"[{label}] {message[:120]}")
    return summary


def set_random_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def build_model(
    training_messages: pd.Series,
    max_tokens: int = 20_000,
    sequence_length: int = 120,
    embedding_dim: int = 64,
    gru_units: int = 64,
):
    """Build a raw-string-to-spam-probability model with train-only vocabulary."""
    import tensorflow as tf

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize="lower_and_strip_punctuation",
        name="text_vectorization",
    )
    train_text = training_messages.astype(str).to_numpy().reshape(-1)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_text).batch(256))

    raw_sms = tf.keras.Input(shape=(), dtype=tf.string, name="sms")
    token_ids = vectorizer(raw_sms)
    embedded = tf.keras.layers.Embedding(
        input_dim=max_tokens,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(token_ids)
    encoded = tf.keras.layers.GRU(gru_units, dropout=0.2, use_cudnn=False, name="gru")(embedded)
    probability = tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(encoded)
    model = tf.keras.Model(inputs=raw_sms, outputs=probability, name="sms_spam_gru")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.Precision(name="precision", thresholds=CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=CLASSIFICATION_THRESHOLD),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def fit_model(
    model,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_path: str | Path = "models/sms_spam_gru.keras",
    epochs: int = 20,
    batch_size: int = 32,
):
    """Fit only on train, select the epoch using validation PR-AUC, and save."""
    import tensorflow as tf

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    counts = train["label_id"].value_counts().to_dict()
    total = sum(counts.values())
    class_weight = {
        0: total / (2 * counts[0]),
        1: total / (2 * counts[1]),
    }
    history = model.fit(
        train["message"].astype(str).to_numpy(),
        train["label_id"].to_numpy(dtype=np.float32),
        validation_data=(
            validation["message"].astype(str).to_numpy(),
            validation["label_id"].to_numpy(dtype=np.float32),
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
        class_weight=class_weight,
    )
    model.save(model_path)
    print(f"Modelo guardado en {model_path.resolve()}")
    return history


def evaluate_model(model, test: pd.DataFrame, output_dir: str | Path = "reports") -> dict[str, Any]:
    """Evaluate on the held-out test set once and save measured artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spam_probability = np.asarray(
        model.predict(test["message"].astype(str).to_numpy(), verbose=0)
    ).reshape(-1)
    y_true = test["label_id"].to_numpy(dtype=np.int32)
    y_pred = (spam_probability >= CLASSIFICATION_THRESHOLD).astype(np.int32)
    metrics: dict[str, Any] = {
        "threshold": CLASSIFICATION_THRESHOLD,
        "test_rows": int(len(test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, spam_probability)),
        "pr_auc_average_precision": float(average_precision_score(y_true, spam_probability)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["legítimo (ham)", "spam"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels_ham_spam": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    predictions = test[["label", "message"]].copy()
    predictions["true_label"] = predictions["label"].map({"ham": "legítimo", "spam": "spam"})
    predictions["spam_probability"] = spam_probability
    predictions["predicted_label"] = np.where(y_pred == 1, "spam", "legítimo")
    predictions["predicted_confidence"] = np.where(y_pred == 1, spam_probability, 1 - spam_probability)
    predictions.head(10).to_csv(out / "test_predictions.csv", index=False, encoding="utf-8")

    cm = np.asarray(metrics["confusion_matrix_labels_ham_spam"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["legítimo", "spam"], yticklabels=["legítimo", "spam"])
    ax.set(xlabel="Predicción", ylabel="Etiqueta real", title="Matriz de confusión (prueba)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print("Métricas medidas en prueba:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Primeras predicciones medidas:")
    print(predictions.head(10).to_string(index=False))
    return metrics


def plot_history(history, output_dir: str | Path = "reports") -> Path:
    """Save loss, PR-AUC, and recall curves from the actual fit history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    values = history.history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(
        axes,
        ("loss", "pr_auc", "recall"),
        ("Pérdida", "PR-AUC", "Recall spam"),
    ):
        if metric not in values:
            ax.set_visible(False)
            continue
        ax.plot(values[metric], label="entrenamiento")
        validation_key = f"val_{metric}"
        if validation_key in values:
            ax.plot(values[validation_key], label="validación")
        ax.set(title=title, xlabel="Época", ylabel=title)
        ax.legend()
    fig.tight_layout()
    target = out / "training_curves.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def run_training(
    data_path: str | Path | None = None,
    data_dir: str | Path = "data",
    output_dir: str | Path = ".",
    epochs: int = 20,
    batch_size: int = 32,
):
    """Run the complete reproducible pipeline and return real outputs."""
    import tensorflow as tf

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    set_random_seeds(SEED)
    raw = load_dataset(data_path=data_path, data_dir=data_dir)
    clean = clean_dataset(raw)
    explore_dataset(clean, root / "reports")
    train, validation, test = split_dataset(clean, seed=SEED)
    model = build_model(train["message"])
    model_path = root / "models" / "sms_spam_gru.keras"
    history = fit_model(
        model, train, validation, model_path=model_path, epochs=epochs, batch_size=batch_size
    )
    plot_history(history, root / "reports")
    # Reload from disk to ensure the saved artifact itself is usable for inference.
    reloaded = tf.keras.models.load_model(model_path, compile=False)
    probe_message = validation["message"].iloc[0]
    probe_input = tf.convert_to_tensor([probe_message], dtype=tf.string)
    probe = np.asarray(reloaded(probe_input)).reshape(-1)
    if probe.size != 1 or not np.isfinite(probe[0]):
        raise RuntimeError("El modelo guardado no produjo una probabilidad válida para texto crudo.")
    metrics = evaluate_model(reloaded, test, root / "reports")
    run_info = {
        "dataset_source": DATASET_PAGE,
        "dataset_citation": DATASET_CITATION,
        "seed": SEED,
        "class_mapping": LABEL_TO_ID,
        "threshold": CLASSIFICATION_THRESHOLD,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "epochs_completed": len(history.epoch),
        "model_path": str(model_path),
        "model_input_dtype": str(reloaded.inputs[0].dtype),
    }
    (root / "reports" / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"model_path": model_path, "history": history, "metrics": metrics, "run_info": run_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", help="Ruta local al archivo SMSSpamCollection (TSV).")
    parser.add_argument("--data-dir", default="data", help="Directorio de descarga/lectura del dataset.")
    parser.add_argument("--output-dir", default=".", help="Directorio raíz de artefactos de salida.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_training(args.data_path, args.data_dir, args.output_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()

