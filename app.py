"""
App Streamlit - Regresion logistica (Modelo 1) sobre datos de Banco.

Modelo 1:  y ~ duration + pdays + nr.employed   (link logit)
Variables elegidas tras la reduccion de dimensiones (ANOVA).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats

from funciones import (
    calcula_indicadores,
    correlacionS,
    curva_roc,
    indicadores_modelo,
)

# --------------------------------------------------------------------------
# Configuracion general
# --------------------------------------------------------------------------
st.set_page_config(page_title="Regresión Logística · Banco", page_icon="📈", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "archivoDatos_Banco.xlsx"
VARS = ["duration", "pdays", "nr.employed"]
TARGET = "y"

# Paleta categorica validada (azul / naranja) y escala secuencial de un solo tono
C_AZUL, C_NARANJA, C_GRIS = "#2a78d6", "#eb6834", "#9a9a95"
SECUENCIAL = "Blues"


def fig_base(fig, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=90, b=10),
        title=dict(y=0.97, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def mostrar(fig):
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo el archivo de datos…")
def leer_excel(origen) -> pd.DataFrame:
    return pd.read_excel(origen)


with st.sidebar:
    st.title("📈 Regresión Logística")
    st.caption("Analítica Predictiva de Datos · Modelo 1 (logit)")
    archivo = st.file_uploader("Cargar otro dataset (.xlsx)", type=["xlsx"])
    st.divider()
    st.markdown("**Fórmula del Modelo 1**")
    st.code("y ~ duration + pdays + nr.employed", language="r")
    corte = st.slider("Punto de corte para clasificar", 0.05, 0.95, 0.50, 0.01,
                      help="El script de R usa 0.5: predice 1 si la probabilidad > corte.")

try:
    df = leer_excel(archivo if archivo is not None else DATA_PATH)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo leer el dataset: {e}")
    st.stop()

faltan = [c for c in VARS + [TARGET] if c not in df.columns]
if faltan:
    st.error(f"Al dataset le faltan columnas requeridas: {faltan}")
    st.stop()
if not set(df[TARGET].dropna().unique()) <= {0, 1}:
    st.error("La variable `y` debe ser binaria (0/1).")
    st.stop()
df = df.dropna(subset=VARS + [TARGET]).reset_index(drop=True)
df[TARGET] = df[TARGET].astype(int)


# --------------------------------------------------------------------------
# Modelo 1  (glm binomial, link logit)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Ajustando el Modelo 1…")
def ajustar_modelo1(datos: pd.DataFrame):
    X = sm.add_constant(datos[VARS].astype(float))
    return sm.GLM(datos[TARGET], X, family=sm.families.Binomial()).fit()


modelo1 = ajustar_modelo1(df[VARS + [TARGET]])
df["prob"] = modelo1.predict(sm.add_constant(df[VARS].astype(float)))
y = df[TARGET].to_numpy()
prob = df["prob"].to_numpy()

# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.title("Regresión logística · Modelo 1")
st.markdown(
    "Aplicación que reproduce la analítica del script `regresion_logistica.R` **hasta el Modelo 1**, "
    "usando como predictoras las tres variables que resultaron significativas en la reducción de "
    "dimensiones (ANOVA): **`duration`**, **`pdays`** y **`nr.employed`**."
)

tab_datos, tab_sel, tab_mod, tab_prob, tab_desemp, tab_sim = st.tabs(
    ["1 · Datos", "2 · Selección de variables", "3 · Modelo 1", "4 · Probabilidades",
     "5 · Desempeño", "6 · Simulador"]
)

# ==========================================================================
# 1 · DATOS
# ==========================================================================
with tab_datos:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones", f"{len(df):,}")
    c2.metric("Variables", df.shape[1] - 1)  # sin la columna 'prob'
    c3.metric("Clientes con y = 1", f"{y.sum():,}")
    c4.metric("Tasa de y = 1", f"{y.mean():.1%}")

    izq, der = st.columns([1, 2])
    with izq:
        conteo = df[TARGET].value_counts().sort_index().rename_axis("y").reset_index(name="n")
        conteo["y"] = conteo["y"].astype(str)
        fig = px.bar(conteo, x="y", y="n", color="y", text="n",
                     color_discrete_map={"0": C_AZUL, "1": C_NARANJA},
                     title="Distribución de la variable respuesta (y)")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_xaxes(type="category")
        fig.update_layout(showlegend=False)
        mostrar(fig_base(fig, 360))
    with der:
        var = st.selectbox("Distribución de una variable predictora según y", VARS)
        serie = df[var]
        recortar = False
        if var == "duration":
            recortar = st.checkbox("Recortar en el percentil 99 (para ver mejor la forma)", value=True)
        d = df[df[var] <= serie.quantile(0.99)] if recortar else df
        if d[var].nunique() <= 12:  # variable casi discreta -> barras
            tabla = (d.groupby([var, TARGET]).size().rename("n").reset_index())
            tabla["%"] = tabla["n"] / tabla.groupby(TARGET)["n"].transform("sum")
            tabla[var] = tabla[var].astype(str)
            tabla[TARGET] = tabla[TARGET].astype(str)
            fig = px.bar(tabla, x=var, y="%", color=TARGET, barmode="group",
                         color_discrete_map={"0": C_AZUL, "1": C_NARANJA},
                         title=f"{var} por y (% dentro de cada grupo)")
            fig.update_yaxes(tickformat=".0%")
        else:
            fig = px.histogram(d, x=var, color=d[TARGET].astype(str), histnorm="probability",
                               barmode="overlay", opacity=0.65, nbins=60,
                               color_discrete_map={"0": C_AZUL, "1": C_NARANJA},
                               labels={"color": "y"},
                               title=f"{var} por y (proporción dentro de cada grupo)")
        mostrar(fig_base(fig, 360))
        if var == "pdays":
            n999 = (df["pdays"] == 999).mean()
            st.caption(f"pdays = 999 significa *no contactado previamente*: {n999:.1%} de los registros.")

    st.subheader("Estadísticos descriptivos de las variables del modelo")
    desc = df[VARS].describe().T
    st.dataframe(desc.style.format("{:,.2f}"), width="stretch")

    with st.expander("Ver el dataset completo"):
        st.dataframe(df.drop(columns="prob"), width="stretch", height=350)

# ==========================================================================
# 2 · SELECCIÓN DE VARIABLES
# ==========================================================================
with tab_sel:
    st.subheader("ANOVA de cada variable numérica frente a y")
    st.markdown(
        "Prueba F de un factor (y = 0 vs y = 1) para cada variable numérica. "
        "Marcadas en azul las tres variables retenidas en la reducción de dimensiones."
    )
    numericas = [c for c in df.select_dtypes("number").columns if c not in (TARGET, "prob")]
    filas = []
    for c in numericas:
        F, p = stats.f_oneway(df.loc[df[TARGET] == 0, c], df.loc[df[TARGET] == 1, c])
        filas.append({"variable": c, "F": F, "p-valor": p, "significativa (α=0.05)": p < 0.05,
                      "retenida en el modelo": c in VARS})
    anova = pd.DataFrame(filas).sort_values("F", ascending=False)

    izq, der = st.columns([3, 2])
    with izq:
        a = anova.sort_values("F")
        fig = go.Figure(go.Bar(
            x=a["F"], y=a["variable"], orientation="h",
            marker_color=[C_AZUL if r else C_GRIS for r in a["retenida en el modelo"]],
            hovertemplate="%{y}<br>F = %{x:,.1f}<extra></extra>",
        ))
        fig.update_layout(title="Estadístico F por variable")
        mostrar(fig_base(fig, 400))
    with der:
        st.dataframe(
            anova.style.format({"F": "{:,.1f}", "p-valor": lambda v: "< 1e-300" if v == 0 else f"{v:.2e}"}),
            width="stretch", hide_index=True, height=400,
        )

    st.divider()
    st.subheader("Correlación de Spearman (antes de modelar)")
    incluir_cat = st.checkbox(
        "Incluir variables categóricas codificadas numéricamente",
        help="El script de R convierte todas las columnas a numérico. Aquí las categóricas se codifican "
             "con códigos ordinales arbitrarios, por lo que su correlación es solo referencial.",
    )
    umbral = st.slider("Umbral de |correlación| para marcar pares redundantes", 0.3, 0.95, 0.60, 0.05)

    base = df.drop(columns="prob").copy()
    if incluir_cat:
        for c in base.select_dtypes(exclude="number").columns:
            base[c] = base[c].astype("category").cat.codes
    else:
        base = base.select_dtypes("number")
    corr = base.corr(method="spearman")

    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                    text_auto=".2f", title="Matriz de correlación de Spearman")
    fig.update_traces(textfont_size=10)
    mostrar(fig_base(fig, 600 if len(corr) > 14 else 520))

    pares = correlacionS(corr)
    pares["filtro"] = (pares["cor"].abs() > umbral).astype(int)
    pares = pares.reindex(pares["cor"].abs().sort_values(ascending=False).index)

    izq, der = st.columns(2)
    with izq:
        st.markdown(f"**Pares con |ρ| > {umbral:.2f}**")
        st.dataframe(pares[pares["filtro"] == 1].style.format({"cor": "{:.3f}"}),
                     width="stretch", hide_index=True)
    with der:
        st.markdown("**Redundancias respecto a las variables del modelo**")
        for v in VARS:
            socios = pares[(pares["filtro"] == 1) & ((pares["i"] == v) | (pares["j"] == v))]
            if socios.empty:
                st.write(f"• `{v}`: sin pares por encima del umbral.")
            else:
                lista = ", ".join(
                    f"`{r.j if r.i == v else r.i}` (ρ={r.cor:.2f})" for r in socios.itertuples()
                )
                st.write(f"• `{v}` está correlacionada con {lista}.")
        st.download_button("⬇️ Descargar correlaciones (CSV)", pares.to_csv(index=False).encode(),
                           "correlacion_variables_logistica.csv", "text/csv")

# ==========================================================================
# 3 · MODELO 1
# ==========================================================================
with tab_mod:
    st.subheader("Modelo 1 · `glm(y ~ duration + pdays + nr.employed, family = binomial())`")

    ci = modelo1.conf_int()
    coef = pd.DataFrame({
        "Estimate": modelo1.params,
        "Std. Error": modelo1.bse,
        "z value": modelo1.tvalues,
        "Pr(>|z|)": modelo1.pvalues,
        "Odds ratio": np.exp(modelo1.params),
        "OR IC95% inf": np.exp(ci[0]),
        "OR IC95% sup": np.exp(ci[1]),
    })
    coef.index = ["(Intercept)" if i == "const" else i for i in coef.index]
    coef.loc["(Intercept)", ["Odds ratio", "OR IC95% inf", "OR IC95% sup"]] = np.nan  # no interpretable
    coef["Signif."] = pd.cut(coef["Pr(>|z|)"], [-1, 0.001, 0.01, 0.05, 0.1, 2],
                             labels=["***", "**", "*", ".", ""])

    st.markdown("**Coeficientes** (equivalente a `summary(modelo1)`)")
    fmt = {"Estimate": "{:.6f}", "Std. Error": "{:.6f}", "z value": "{:.3f}",
           "Odds ratio": "{:.4f}", "OR IC95% inf": "{:.4f}", "OR IC95% sup": "{:.4f}"}
    tabla_coef = pd.DataFrame(index=coef.index)
    for col, f in fmt.items():
        tabla_coef[col] = [("—" if pd.isna(v) else f.format(v)) for v in coef[col]]
    tabla_coef["Pr(>|z|)"] = ["< 2e-16" if v < 2e-16 else f"{v:.2e}" for v in coef["Pr(>|z|)"]]
    tabla_coef["Signif."] = coef["Signif."].astype(str)
    tabla_coef = tabla_coef[["Estimate", "Std. Error", "z value", "Pr(>|z|)", "Signif.",
                             "Odds ratio", "OR IC95% inf", "OR IC95% sup"]]
    st.dataframe(tabla_coef, width="stretch")
    st.caption("Signif. codes:  0 ‘\\*\\*\\*’ 0.001 ‘\\*\\*’ 0.01 ‘\\*’ 0.05 ‘.’ 0.1 ‘ ’ 1  ·  "
               "El odds ratio del intercepto no se interpreta.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Null deviance", f"{modelo1.null_deviance:,.1f}", help=f"{int(modelo1.df_model + modelo1.df_resid)} g.l.")
    c2.metric("Residual deviance", f"{modelo1.deviance:,.1f}", help=f"{int(modelo1.df_resid)} g.l.")
    c3.metric("AIC", f"{modelo1.aic:,.1f}")
    c4.metric("Iteraciones (Fisher scoring)", len(modelo1.fit_history["deviance"]) - 1)

    st.divider()
    izq, der = st.columns([1, 1])
    with izq:
        st.markdown("**Indicadores de `calcula_indicadores(modelo1)`**")
        ind = calcula_indicadores(y, modelo1.fittedvalues.to_numpy())
        st.dataframe(
            pd.DataFrame([{"variable": "regresion", "SSE": ind["SSE"], "correlacion": ind["correlacion"]}])
            .style.format({"SSE": "{:,.2f}", "correlacion": "{:.4f}"}),
            hide_index=True, width="stretch",
        )
        st.caption("SSE = Σ(y − ŷ)²; correlación = Spearman entre y y las probabilidades ajustadas.")

        st.markdown("**Ecuación estimada**")
        b = modelo1.params
        st.latex(
            rf"\ln\!\left(\frac{{p}}{{1-p}}\right) = {b['const']:.4f} "
            rf"{b['duration']:+.5f}\,\text{{duration}} "
            rf"{b['pdays']:+.5f}\,\text{{pdays}} "
            rf"{b['nr.employed']:+.5f}\,\text{{nr.employed}}"
        )
    with der:
        sd = df[VARS].std()
        b_ = modelo1.params.drop("const")
        lo, hi = ci.drop("const")[0], ci.drop("const")[1]
        or_sd = pd.DataFrame({"or": np.exp(b_ * sd), "lo": np.exp(lo * sd), "hi": np.exp(hi * sd)})
        fig = go.Figure(go.Scatter(
            x=or_sd["or"], y=or_sd.index, mode="markers",
            marker=dict(size=10, color=C_AZUL),
            error_x=dict(type="data", symmetric=False, array=or_sd["hi"] - or_sd["or"],
                         arrayminus=or_sd["or"] - or_sd["lo"], color=C_AZUL),
            hovertemplate="%{y}<br>OR = %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=1, line_dash="dash", line_color=C_GRIS)
        fig.update_xaxes(type="log")
        fig.update_layout(title="Odds ratio por +1 desv. estándar (IC 95%)")
        mostrar(fig_base(fig, 300))
        st.caption("Escala logarítmica. Al estandarizar el cambio, los tres efectos son comparables entre sí; "
                   "el odds ratio por unidad está en la tabla de arriba.")

    st.info(
        f"**Lectura.** Cada segundo adicional de `duration` multiplica los odds de y = 1 por "
        f"{np.exp(b['duration']):.4f}; cada unidad de `pdays`, por {np.exp(b['pdays']):.4f}; "
        f"y cada unidad de `nr.employed`, por {np.exp(b['nr.employed']):.4f}, "
        "manteniendo las demás variables constantes."
    )

# ==========================================================================
# 4 · PROBABILIDADES
# ==========================================================================
with tab_prob:
    st.subheader("Probabilidades predichas por el Modelo 1")
    sub_a, sub_b, sub_c = st.tabs(
        ["Observaciones (réplica del script)", "Curvas por nr.employed", "Curvas por pdays"]
    )

    def prob_curva(duracion, pdays, nr):
        eta = (modelo1.params["const"] + modelo1.params["duration"] * duracion
               + modelo1.params["pdays"] * pdays + modelo1.params["nr.employed"] * nr)
        return 1 / (1 + np.exp(-eta))

    niveles_nr = sorted(df["nr.employed"].unique())
    niveles_pd = sorted(df["pdays"].unique())

    with sub_a:
        st.markdown(
            "Probabilidad ajustada de cada cliente frente a `duration`, coloreada por `nr.employed` o `pdays` "
            "(los dos `ggplot` del script)."
        )
        colorear = st.radio("Colorear por", ["nr.employed", "pdays"], horizontal=True)
        muestra = df.sample(min(len(df), 12000), random_state=1)
        if colorear == "nr.employed":
            muestra = muestra.assign(grupo=muestra["nr.employed"].astype(str))
            orden = [str(v) for v in niveles_nr]
            cols = px.colors.sample_colorscale(SECUENCIAL, np.linspace(0.3, 1, len(orden)))
            cmap = dict(zip(orden, cols))
        else:
            muestra = muestra.assign(
                grupo=np.where(muestra["pdays"] == 999, "999 (sin contacto previo)", "< 999 (contactado antes)")
            )
            orden = ["999 (sin contacto previo)", "< 999 (contactado antes)"]
            cmap = {orden[0]: C_AZUL, orden[1]: C_NARANJA}
        fig = px.scatter(muestra, x="duration", y="prob", color="grupo", opacity=0.6,
                         category_orders={"grupo": orden}, color_discrete_map=cmap,
                         labels={"prob": "probabilidad", "grupo": colorear},
                         title=f"Probabilidad predicha vs duration · color: {colorear}")
        fig.update_traces(marker=dict(size=5))
        mostrar(fig_base(fig, 460))
        st.caption(f"Muestra aleatoria de {len(muestra):,} observaciones para mantener la app fluida. "
                   "Con `pdays` se agrupa en 999 vs. contactado previamente (27 niveles distintos en total).")

    with sub_b:
        c1, c2 = st.columns(2)
        pd_fijo = c1.selectbox("pdays fijo", niveles_pd, index=niveles_pd.index(999))
        dmax = c2.slider("Rango de duration (seg.)", 300, int(df["duration"].max()), 1500, 100,
                         key="dmax_b")
        grid = np.linspace(0, dmax, 200)
        fig = go.Figure()
        cols = px.colors.sample_colorscale(SECUENCIAL, np.linspace(0.3, 1, len(niveles_nr)))
        for nr, col in zip(niveles_nr, cols):
            fig.add_scatter(x=grid, y=prob_curva(grid, pd_fijo, nr), mode="lines", name=str(nr),
                            line=dict(color=col, width=2))
        fig.update_layout(title=f"P(y = 1) vs duration · pdays = {pd_fijo}",
                          xaxis_title="duration", yaxis_title="probabilidad", legend_title="nr.employed")
        fig.update_yaxes(range=[0, 1])
        mostrar(fig_base(fig, 460))

    with sub_c:
        c1, c2, c3 = st.columns([1, 2, 1])
        nr_fijo = c1.selectbox("nr.employed fijo", niveles_nr, index=niveles_nr.index(5099.1)
                               if 5099.1 in niveles_nr else 0)
        elegidos = c2.multiselect("Valores de pdays a comparar (máx. 4)", niveles_pd,
                                  default=[v for v in (999, 3, 6) if v in niveles_pd], max_selections=4)
        dmax = c3.slider("Rango de duration (seg.)", 300, int(df["duration"].max()), 1500, 100,
                         key="dmax_c")
        grid = np.linspace(0, dmax, 200)
        fig = go.Figure()
        for pdv, col in zip(elegidos, [C_AZUL, C_NARANJA, "#1baf7a", "#eda100"]):
            fig.add_scatter(x=grid, y=prob_curva(grid, pdv, nr_fijo), mode="lines", name=f"pdays = {pdv}",
                            line=dict(color=col, width=2))
        fig.update_layout(title=f"P(y = 1) vs duration · nr.employed = {nr_fijo}",
                          xaxis_title="duration", yaxis_title="probabilidad")
        fig.update_yaxes(range=[0, 1])
        mostrar(fig_base(fig, 460))

# ==========================================================================
# 5 · DESEMPEÑO
# ==========================================================================
with tab_desemp:
    m = indicadores_modelo(y, prob, corte)
    st.subheader("Indicadores del Modelo 1")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", f"{m['AUC']:.3f}")
    c2.metric("Gini", f"{m['Gini']:.3f}")
    c3.metric("LogLoss", f"{m['LogLoss']:.4f}")
    c4.metric("KS", f"{m['KS']:.4f}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Accuracy", f"{m['Accuracy']:.3f}")
    c6.metric("Error de mala clasificación", f"{m['Error']:.3f}")
    c7.metric("Sensibilidad", f"{m['Sensitivity']:.3f}")
    c8.metric("Especificidad", f"{m['Specificity']:.3f}")
    st.caption(f"Clasificación con punto de corte = {corte:.2f} (se predice 1 si la probabilidad > corte). "
               "Sensibilidad y especificidad usan la definición estándar de `caret::confusionMatrix(positive = \"1\")`.")

    izq, der = st.columns(2)
    with izq:
        fpr, tpr, _ = curva_roc(y, prob)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"Modelo 1 (AUC = {m['AUC']:.3f})",
                        line=dict(color=C_AZUL, width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                        line=dict(color=C_GRIS, dash="dash", width=1))
        fig.add_scatter(x=[1 - m["Specificity"]], y=[m["Sensitivity"]], mode="markers",
                        name=f"Corte {corte:.2f}", marker=dict(color=C_NARANJA, size=11,
                                                              line=dict(color="white", width=2)))
        fig.update_layout(title="Curva ROC", xaxis_title="1 − Especificidad (FPR)",
                          yaxis_title="Sensibilidad (TPR)")
        mostrar(fig_base(fig, 420))
    with der:
        fig = go.Figure()
        for clase, col, nom in ((0, C_AZUL, "y = 0"), (1, C_NARANJA, "y = 1")):
            p = np.sort(prob[y == clase])
            fig.add_scatter(x=p, y=np.arange(1, len(p) + 1) / len(p), mode="lines", name=nom,
                            line=dict(color=col, width=2))
        # posicion del KS
        grid = np.linspace(prob.min(), prob.max(), 500)
        f0 = np.searchsorted(np.sort(prob[y == 0]), grid, side="right") / (y == 0).sum()
        f1 = np.searchsorted(np.sort(prob[y == 1]), grid, side="right") / (y == 1).sum()
        k = int(np.argmax(np.abs(f0 - f1)))
        fig.add_scatter(x=[grid[k], grid[k]], y=[f1[k], f0[k]], mode="lines",
                        name=f"KS = {m['KS']:.3f}", line=dict(color="#52514e", width=2, dash="dot"))
        fig.update_layout(title="Distribuciones acumuladas de la probabilidad (KS)",
                          xaxis_title="probabilidad predicha", yaxis_title="proporción acumulada")
        mostrar(fig_base(fig, 420))

    izq, der = st.columns([1, 1])
    with izq:
        cm = m["matriz"]
        fig = px.imshow(cm, x=["Pred 0", "Pred 1"], y=["Real 0", "Real 1"], text_auto=",",
                        color_continuous_scale="Blues", aspect="auto",
                        title="Matriz de confusión")
        fig.update_traces(textfont_size=18)
        fig.update_coloraxes(showscale=False)
        mostrar(fig_base(fig, 360))
    with der:
        st.markdown("**Métricas de clasificación** (como `confusionMatrix`)")
        extra = pd.DataFrame({
            "Métrica": ["Accuracy", "Kappa", "Sensibilidad (Recall)", "Especificidad",
                        "Precisión (Pos Pred Value)", "Neg Pred Value", "Balanced Accuracy"],
            "Valor": [m["Accuracy"], m["Kappa"], m["Sensitivity"], m["Specificity"],
                      m["Precision (PPV)"], m["NPV"], m["Balanced Accuracy"]],
        })
        st.dataframe(extra.style.format({"Valor": "{:.4f}"}), hide_index=True, width="stretch")
        st.caption(f"Prevalencia de y = 1: {y.mean():.4f}. "
                   "Con clases desbalanceadas, un corte más bajo suele mejorar la sensibilidad.")

    with st.expander("Explorar el efecto del punto de corte"):
        cortes = np.arange(0.05, 0.96, 0.05)
        filas = []
        for c in cortes:
            r = indicadores_modelo(y, prob, c)
            filas.append({"corte": round(c, 2), "Accuracy": r["Accuracy"],
                          "Sensibilidad": r["Sensitivity"], "Especificidad": r["Specificity"]})
        t = pd.DataFrame(filas)
        fig = go.Figure()
        for col, color in (("Accuracy", C_GRIS), ("Sensibilidad", C_NARANJA), ("Especificidad", C_AZUL)):
            fig.add_scatter(x=t["corte"], y=t[col], mode="lines+markers", name=col,
                            line=dict(color=color, width=2), marker=dict(size=7))
        fig.update_layout(xaxis_title="punto de corte", yaxis_title="valor")
        mostrar(fig_base(fig, 360))

# ==========================================================================
# 6 · SIMULADOR
# ==========================================================================
with tab_sim:
    st.subheader("Simulador de probabilidad")
    st.markdown("Ingresa los valores de un cliente para obtener su probabilidad de y = 1 con el Modelo 1.")
    niveles_nr = sorted(df["nr.employed"].unique())
    niveles_pd = sorted(df["pdays"].unique())

    c1, c2, c3 = st.columns(3)
    dur = c1.slider("duration (segundos)", 0, int(df["duration"].max()), int(df["duration"].median()), 10)
    pdv = c2.selectbox("pdays", niveles_pd, index=niveles_pd.index(999),
                       help="999 = no contactado previamente")
    nre = c3.selectbox("nr.employed", niveles_nr, index=niveles_nr.index(5191.0)
                       if 5191.0 in niveles_nr else 0)

    b = modelo1.params
    eta = b["const"] + b["duration"] * dur + b["pdays"] * pdv + b["nr.employed"] * nre
    p_hat = 1 / (1 + np.exp(-eta))

    izq, der = st.columns([1, 2])
    with izq:
        st.metric("P(y = 1)", f"{p_hat:.1%}", delta=f"{(p_hat - y.mean()) * 100:+.1f} pp vs. tasa base",
                  delta_color="off")
        st.metric("Odds", f"{p_hat / (1 - p_hat):.4f}")
        st.metric("Clasificación (corte %.2f)" % corte, "y = 1" if p_hat > corte else "y = 0")
    with der:
        dmax = max(1500, int(dur * 1.15))
        grid = np.linspace(0, dmax, 200)
        eta_g = b["const"] + b["duration"] * grid + b["pdays"] * pdv + b["nr.employed"] * nre
        fig = go.Figure()
        fig.add_scatter(x=grid, y=1 / (1 + np.exp(-eta_g)), mode="lines", name="Curva del modelo",
                        line=dict(color=C_AZUL, width=2))
        fig.add_hline(y=corte, line_dash="dash", line_color=C_GRIS,
                      annotation_text=f"corte {corte:.2f}", annotation_position="bottom right")
        fig.add_scatter(x=[dur], y=[p_hat], mode="markers", name="Cliente simulado",
                        marker=dict(color=C_NARANJA, size=13, line=dict(color="white", width=2)))
        fig.update_layout(title=f"P(y = 1) vs duration · pdays = {pdv}, nr.employed = {nre}",
                          xaxis_title="duration", yaxis_title="probabilidad")
        fig.update_yaxes(range=[0, 1])
        mostrar(fig_base(fig, 360))
    st.caption(f"Log-odds del cliente: {eta:.3f}  →  p = 1 / (1 + e^(−log-odds)) = {p_hat:.4f}.")

st.divider()
st.caption("Proyecto de clase · Analítica Predictiva de Datos · Python/Streamlit a partir de "
           "`regresion_logistica.R` y `funciones.R`.")
