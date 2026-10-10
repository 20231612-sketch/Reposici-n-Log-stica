# Regresión logística · Modelo 1 (Streamlit)

App de Streamlit que reproduce, en Python, la analítica de `regresion_logistica.R` **hasta el Modelo 1**:

```
y ~ duration + pdays + nr.employed      (glm binomial, link logit)
```

Las tres variables provienen de la reducción de dimensiones: resultaron significativas en el ANOVA.

## Contenido de la app

| Pestaña | Qué muestra | Equivalente en R |
|---|---|---|
| 1 · Datos | Tamaño, balance de `y`, distribución de variables, descriptivos | `str(my_data)`, `table(my_data$y)` |
| 2 · Selección de variables | ANOVA (F y p-valor) por variable numérica y matriz de correlación de Spearman con umbral ajustable | `correlacionS()`, `corre$filtro`, `write.csv(...)` |
| 3 · Modelo 1 | Coeficientes, z, p-valores, odds ratio con IC 95 %, deviance, AIC, SSE y correlación | `summary(modelo1)`, `calcula_indicadores(modelo1)` |
| 4 · Probabilidades | Probabilidad predicha vs `duration` (por `nr.employed` y por `pdays`) | los dos `ggplot` del script |
| 5 · Desempeño | AUC, Gini, LogLoss, KS, matriz de confusión, sensibilidad, especificidad, accuracy, error, ROC y curva KS; punto de corte ajustable | bloque *Indicadores* |
| 6 · Simulador | Probabilidad de y = 1 para un cliente hipotético | `predict(modelo1, type = "response")` |

## Estructura

```
├── app.py                      # aplicación Streamlit
├── funciones.py                # port a Python de funciones.R + indicadores
├── data/archivoDatos_Banco.xlsx
├── requirements.txt
└── .streamlit/config.toml
```

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub.
2. En <https://share.streamlit.io> crea una app nueva apuntando a `app.py` en la rama principal.

## Notas metodológicas

- **Sensibilidad y especificidad.** En el script de R, `MLmetrics::Sensitivity(PRED, y)` y `Specificity(PRED, y)` reciben los argumentos en orden invertido (la función espera `y_true, y_pred`), por lo que devuelven precisión y valor predictivo negativo. La app usa las definiciones estándar, que coinciden con `caret::confusionMatrix(positive = "1")`.
- **KS.** Se calcula como el máximo de |F₁(p) − F₀(p)| entre las distribuciones acumuladas de la probabilidad para y = 1 e y = 0 (escala 0–1).
- **Correlación.** Por defecto se usan solo variables numéricas. El script de R convierte todo a numérico; en este dataset las categóricas son texto, por lo que se ofrece la opción de codificarlas (códigos ordinales arbitrarios, solo referencial).
- **`pdays = 999`** significa «no contactado previamente» (≈96 % de los registros).
