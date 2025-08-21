import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
    df = pd.read_csv(url)
    # normalizar nombres por si varían
    lower = {c.lower(): c for c in df.columns}
    cols = {
        "country": lower.get("country_region", "Country_Region"),
        "province": lower.get("province_state", "Province_State"),
        "confirmed": lower.get("confirmed", "Confirmed"),
        "deaths": lower.get("deaths", "Deaths"),
        "recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
        "active": lower.get("active", "Active") if "active" in lower else None,
    }
    return df, url, cols

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)
st.sidebar.caption(f"Fuente: {source_url}")

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar todas las filas del dataset, luego volver al estado inicial
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar todas las columnas del dataset
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
# ———————————————————————————————————————————————
st.header("c) Gráfica de líneas por país (muertes > 2500)")
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[cols["country"]] + metrics].copy()
base = base.rename(columns={cols["country"]: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)

if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# ———————————————————————————————————————————————
# d) Barras de fallecidos de estados de Estados Unidos
# ———————————————————————————————————————————————
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# ———————————————————————————————————————————————
# e) Gráfica de sectores (simulada con barra si no hay pie nativo)
# ———————————————————————————————————————————————
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    # Como Streamlit no tiene pie nativo, mostramos distribución normalizada como barra
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# ———————————————————————————————————————————————
# f) Histograma del total de fallecidos por país (simulado con bar_chart)
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot de Confirmed, Deaths, Recovered, Active (simulado con box_chart)
# ———————————————————————————————————————————————
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
# Streamlit no tiene boxplot nativo, así que mostramos estadísticas resumen en tabla
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)

# ———————————————————————————————————————————————
# 2. Estadística descriptiva y avanzada
# ———————————————————————————————————————————————
st.header("2) Estadística descriptiva y avanzada")

# 2.1. Métricas clave por país: Confirmados, Fallecidos, CFR y tasas por 100k
df_grouped = df.groupby(country_col).agg({
    C: "sum",
    D: "sum"
}).reset_index()
df_grouped["CFR"] = df_grouped[D] / df_grouped[C]
# Si la población no está disponible, asumimos 1M para tasa
df_grouped["Confirmed_per_100k"] = df_grouped[C] / 1e6 * 100000
df_grouped["Deaths_per_100k"] = df_grouped[D] / 1e6 * 100000
st.subheader("Métricas por país")
st.dataframe(df_grouped.sort_values(D, ascending=False))

# 2.2. Intervalos de confianza para CFR (binomial)
from scipy.stats import binom
def cfr_ci(deaths, confirmed, alpha=0.05):
    if confirmed == 0:
        return (0, 0)
    ci_low, ci_upp = binom.interval(1-alpha, confirmed, deaths/confirmed)
    return ci_low/confirmed, ci_upp/confirmed

df_grouped["CFR_CI"] = df_grouped.apply(lambda row: cfr_ci(row[D], row[C]), axis=1)
st.subheader("Intervalos de confianza de CFR")
st.dataframe(df_grouped[[country_col, "CFR", "CFR_CI"]])

# 2.3. Test de hipótesis de proporciones para comparar CFR entre dos países
st.subheader("Comparación de CFR entre dos países")
pais1 = st.selectbox("País 1", df_grouped[country_col].tolist())
pais2 = st.selectbox("País 2", df_grouped[country_col].tolist(), index=1)

from statsmodels.stats.proportion import proportions_ztest
d1 = int(df_grouped.loc[df_grouped[country_col]==pais1, D])
n1 = int(df_grouped.loc[df_grouped[country_col]==pais1, C])
d2 = int(df_grouped.loc[df_grouped[country_col]==pais2, D])
n2 = int(df_grouped.loc[df_grouped[country_col]==pais2, C])

stat, pval = proportions_ztest([d1, d2], [n1, n2])
st.write(f"Z-test estadístico: {stat:.3f}, p-value: {pval:.3f}")

# 2.4. Detección de outliers usando Z-score
from scipy.stats import zscore
df_grouped["zscore_deaths"] = zscore(df_grouped[D].fillna(0))
outliers = df_grouped[df_grouped["zscore_deaths"].abs() > 3]
st.subheader("Outliers en fallecidos (|Z|>3)")
st.dataframe(outliers[[country_col, D, "zscore_deaths"]])

# 2.5. Gráfico de control (3σ) de muertes diarias
st.subheader("Gráfico de control (3σ) de muertes")
# Si hay columna de fecha
if "Last_Update" in df.columns:
    df["Last_Update"] = pd.to_datetime(df["Last_Update"])
    daily_deaths = df.groupby("Last_Update")[D].sum()
else:
    # fallback si no hay fecha
    daily_deaths = df.groupby(country_col)[D].sum()

mean_deaths = daily_deaths.mean()
std_deaths = daily_deaths.std()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(daily_deaths.index, daily_deaths.values, marker="o", label="Muertes diarias")
ax.axhline(mean_deaths, color="green", linestyle="--", label="Media")
ax.axhline(mean_deaths + 3*std_deaths, color="red", linestyle="--", label="+3σ")
ax.axhline(mean_deaths - 3*std_deaths, color="red", linestyle="--", label="-3σ")
plt.xticks(rotation=45)
ax.legend()
st.pyplot(fig)
