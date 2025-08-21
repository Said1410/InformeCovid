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
# 3) Modelado y proyecciones
# ———————————————————————————————————————————————
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import streamlit as st

st.header("3️⃣ Modelado y proyecciones COVID-19")

# Selección de país
pais_model = st.selectbox("Selecciona un país para proyección", df_grouped["Country"].tolist())

# Filtrar datos diarios del país
if "Last_Update" in df.columns:
    df["Last_Update"] = pd.to_datetime(df["Last_Update"])
    df_pais_daily = df[df[cols["country"]] == pais_model].groupby("Last_Update")[C, D].sum()
else:
    st.warning("No hay columna de fecha para generar series de tiempo.")
    df_pais_daily = df.groupby(cols["country"])[C, D].sum().to_frame().T

# 3.1 Suavizado de 7 días
df_pais_daily_smooth = df_pais_daily.rolling(7, min_periods=1).mean()

st.subheader("Series de tiempo suavizadas (7 días)")
st.line_chart(df_pais_daily_smooth)

# 3.2 Modelo SARIMA y pronóstico a 14 días
st.subheader("Pronóstico a 14 días con SARIMA")
forecast_horizon = 14

# Para pronosticar Confirmados
y_cases = df_pais_daily_smooth[C]
y_deaths = df_pais_daily_smooth[D]

# Configuración SARIMA simple (p,d,q)(P,D,Q,s)
sarima_order = (1,1,1)
seasonal_order = (1,1,1,7)  # semanal

try:
    model_cases = SARIMAX(y_cases, order=sarima_order, seasonal_order=seasonal_order)
    results_cases = model_cases.fit(disp=False)
    forecast_cases = results_cases.get_forecast(steps=forecast_horizon)
    mean_cases = forecast_cases.predicted_mean
    conf_cases = forecast_cases.conf_int()

    model_deaths = SARIMAX(y_deaths, order=sarima_order, seasonal_order=seasonal_order)
    results_deaths = model_deaths.fit(disp=False)
    forecast_deaths = results_deaths.get_forecast(steps=forecast_horizon)
    mean_deaths = forecast_deaths.predicted_mean
    conf_deaths = forecast_deaths.conf_int()
except Exception as e:
    st.error(f"Error al entrenar SARIMA: {e}")
    mean_cases, conf_cases, mean_deaths, conf_deaths = None, None, None, None

# 3.3 Validación (Backtesting) usando MAE y MAPE
if len(y_cases) > forecast_horizon:
    train = y_cases[:-forecast_horizon]
    test = y_cases[-forecast_horizon:]

    model_bt = SARIMAX(train, order=sarima_order, seasonal_order=seasonal_order)
    results_bt = model_bt.fit(disp=False)
    pred_bt = results_bt.get_forecast(steps=forecast_horizon).predicted_mean

    mae = mean_absolute_error(test, pred_bt)
    mape = mean_absolute_percentage_error(test, pred_bt)
    st.write(f"Backtesting - Confirmados: MAE={mae:.2f}, MAPE={mape*100:.2f}%")

# 3.4 Gráfica de forecast con bandas de confianza
if mean_cases is not None:
    plt.figure(figsize=(12,5))
    plt.plot(y_cases.index, y_cases, label="Casos observados")
    plt.plot(mean_cases.index, mean_cases, label="Forecast casos", color="orange")
    plt.fill_between(conf_cases.index, conf_cases.iloc[:,0], conf_cases.iloc[:,1], color="orange", alpha=0.2)

    plt.plot(y_deaths.index, y_deaths, label="Muertes observadas")
    plt.plot(mean_deaths.index, mean_deaths, label="Forecast muertes", color="red")
    plt.fill_between(conf_deaths.index, conf_deaths.iloc[:,0], conf_deaths.iloc[:,1], color="red", alpha=0.2)

    plt.title(f"Forecast a 14 días - {pais_model}")
    plt.xlabel("Fecha")
    plt.ylabel("Casos / Muertes")
    plt.legend()
    st.pyplot(plt)

