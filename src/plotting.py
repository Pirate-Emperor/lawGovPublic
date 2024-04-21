from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow import keras


class PlottingManager:
    def __init__(self) -> None:
        plt.style.use("seaborn")

    def plot_subplots_curve(
        self,
        training_measure: List[List[float]],
        validation_measure: List[List[float]],
        title: str,
        train_color: str = "orangered",
        validation_color: str = "dodgerblue",
    ) -> None:
        
        plt.figure(figsize=(12, 8))

        for i in range(len(training_measure)):
            plt.subplot(2, 2, i + 1)
            plt.plot(training_measure[i], c=train_color)
            plt.plot(validation_measure[i], c=validation_color)
            plt.title("Fold " + str(i + 1))

        plt.suptitle(title)
        plt.show()

    def plot_heatmap(
        self, measure: List[List[float]], title: str, cmap: str = "coolwarm"
    ) -> None:

        # transpose the array to make it `num_epochs` by `k`
        values_array = np.array(measure).T
        df_cm = pd.DataFrame(
            values_array,
            range(1, values_array.shape[0] + 1),
            ["fold " + str(i + 1) for i in range(4)],
        )

        plt.figure(figsize=(10, 8))
        plt.title(
            title + " Throughout " + str(values_array.shape[1]) + " Folds", pad=20
        )
        sn.heatmap(df_cm, annot=True, cmap=cmap, annot_kws={"size": 10})
        plt.show()

    def plot_average_curves(
        self,
        title: str,
        x: List[float],
        y: List[float],
        x_label: str,
        y_label: str,
        train_color: str = "orangered",
        validation_color: str = "dodgerblue",
    ) -> None:

        plt.title(title, pad=20)
        plt.plot(x, c=train_color, label=x_label)
        plt.plot(y, c=validation_color, label=y_label)
        plt.legend()
        plt.show()

    def plot_roc_curve(
        self,
        all_models: List[keras.models.Sequential],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:

        plt.figure(figsize=(12, 8))
        for i, model in enumerate(all_models):
            y_pred = model.predict(X_test).ravel()
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_curve = auc(fpr, tpr)
            plt.subplot(2, 2, i + 1)
            plt.plot([0, 1], [0, 1], color="dodgerblue", linestyle="--")
            plt.plot(
                fpr,
                tpr,
                color="orangered",
                label=f"Fold {str(i+1)} (area = {auc_curve:.3f})",
            )
            plt.legend(loc="best")
            plt.title(f"Fold {str(i+1)}")

        plt.suptitle("AUC-ROC curves")
        plt.show()

    def plot_classification_report(
        self, model: keras.models.Sequential, X_test: pd.DataFrame, y_test: pd.Series
    ) -> str | dict:

        y_pred = model.predict(X_test).ravel()
        preds = np.where(y_pred > 0.5, 1, 0)
        cls_report = classification_report(y_test, preds)

        return cls_report

    def plot_confusion_matrix(
        self,
        all_models: List[keras.models.Sequential],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:

        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        for i, (model, ax) in enumerate(zip(all_models, axes.flatten())):
            y_pred = model.predict(X_test).ravel()
            preds = np.where(y_pred > 0.5, 1, 0)

            conf_matrix = confusion_matrix(y_test, preds)
            sn.heatmap(conf_matrix, annot=True, ax=ax)
            ax.set_title(f"Fold {i+1}")

        plt.suptitle("Confusion Matrices")
        plt.tight_layout()
        plt.show()
