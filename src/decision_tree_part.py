from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


@dataclass
class TreeConfig:
    name: str
    criterion: str
    max_depth: int | None
    ccp_alpha: float


def run_decision_tree_experiments(output_dir: Path) -> Dict[str, object]:
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    configs = [
        TreeConfig(name="gini_unpruned", criterion="gini", max_depth=None, ccp_alpha=0.0),
        TreeConfig(name="entropy_unpruned", criterion="entropy", max_depth=None, ccp_alpha=0.0),
        TreeConfig(name="gini_depth4", criterion="gini", max_depth=4, ccp_alpha=0.0),
        TreeConfig(name="entropy_depth6_pruned", criterion="entropy", max_depth=6, ccp_alpha=0.003),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    experiment_rows: List[Dict[str, object]] = []
    fitted_models: Dict[str, DecisionTreeClassifier] = {}

    for cfg in configs:
        model = DecisionTreeClassifier(
            criterion=cfg.criterion,
            max_depth=cfg.max_depth,
            ccp_alpha=cfg.ccp_alpha,
            random_state=42,
        )
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        row = {
            "config": cfg.name,
            "criterion": cfg.criterion,
            "max_depth": cfg.max_depth,
            "ccp_alpha": cfg.ccp_alpha,
            "cv_mean_accuracy": float(np.mean(cv_scores)),
            "cv_std_accuracy": float(np.std(cv_scores)),
            "test_accuracy": float(accuracy_score(y_test, preds)),
            "test_f1_weighted": float(f1_score(y_test, preds, average="weighted")),
            "node_count": int(model.tree_.node_count),
            "tree_depth": int(model.tree_.max_depth),
        }

        experiment_rows.append(row)
        fitted_models[cfg.name] = model

    best = max(experiment_rows, key=lambda r: r["cv_mean_accuracy"])
    best_model = fitted_models[str(best["config"])]
    best_preds = best_model.predict(X_test)

    cm = confusion_matrix(y_test, best_preds)
    report = classification_report(y_test, best_preds, target_names=data.target_names)
    rules = export_text(best_model, feature_names=list(X.columns), max_depth=5)

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "decision_tree_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(experiment_rows[0].keys()))
        writer.writeheader()
        for row in experiment_rows:
            writer.writerow(row)

    with (output_dir / "decision_tree_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "sklearn_breast_cancer",
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "target_names": list(data.target_names),
                "best_config": best,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )

    with (output_dir / "decision_tree_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    with (output_dir / "decision_tree_rules.txt").open("w", encoding="utf-8") as f:
        f.write(rules)

    fig, ax = plt.subplots(figsize=(20, 11))
    plot_tree(
        best_model,
        feature_names=list(X.columns),
        class_names=list(data.target_names),
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Decision Tree (top levels) - best config: {best['config']}")
    plt.tight_layout()
    plt.savefig(output_dir / "decision_tree_visualization.png", dpi=180)
    plt.close(fig)

    return {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "total_configs": len(configs),
        "best_config": best,
    }
