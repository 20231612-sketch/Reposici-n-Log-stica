"""Port a Python de funciones.R y de los indicadores de regresion_logistica.R."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, log_loss, confusion_matrix, cohen_kappa_score, roc_curve,
)


def correlacionS(m: pd.DataFrame) -> pd.DataFrame:
    """Equivalente a correlacionS(): triangulo superior de la matriz -> (i, j, cor)."""
    cols = list(m.columns)
    filas = [
        {"i": cols[a], "j": cols[b], "cor": m.iloc[a, b]}
        for a in range(len(cols))
        for b in range(a + 1, len(cols))
    ]
    return pd.DataFrame(filas)


def calcula_indicadores(y: np.ndarray, fitted: np.ndarray) -> dict:
    """Equivalente a calcula_indicadores(): SSE y correlacion de Spearman (y vs fitted)."""
    sse = float(np.sum((y - fitted) ** 2))
    rho = float(stats.spearmanr(y, fitted)[0])
    return {"SSE": sse, "correlacion": rho}


def indicadores_modelo(y: np.ndarray, prob: np.ndarray, corte: float = 0.5) -> dict:
    """AUC, Gini, LogLoss, KS y metricas de clasificacion (matriz de confusion)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob)
    auc = roc_auc_score(y, prob)
    ks = stats.ks_2samp(prob[y == 1], prob[y == 0]).statistic  # = max(TPR - FPR)
    pred = (prob > corte).astype(int)  # R: ifelse(prob <= corte, 0, 1)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc = (tp + tn) / n
    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "LogLoss": log_loss(y, np.clip(prob, 1e-15, 1 - 1e-15)),
        "KS": ks,
        "Accuracy": acc,
        "Error": 1 - acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision (PPV)": tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "Balanced Accuracy": (sens + spec) / 2,
        "Kappa": cohen_kappa_score(y, pred),
        "matriz": np.array([[tn, fp], [fn, tp]]),
    }


def curva_roc(y, prob):
    fpr, tpr, thr = roc_curve(y, prob)
    return fpr, tpr, thr
