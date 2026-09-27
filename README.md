# ESG Workshop — código Python para finanzas sostenibles y riesgo climático

Ejemplos reproducibles en Python que acompañan como **material de apoyo** los diplomados y talleres de finanzas sostenibles, ESG y riesgo climático que imparte Liber Jaime Merlos. El repositorio contiene solo código y datos abiertos o anonimizados; el material de los cursos (presentaciones, lecturas, grabaciones) no forma parte de él.

## Contenido

| Notebook | Tema | Qué aprenderás |
|---|---|---|
| [01 · Pronóstico de temperatura](notebooks/01_temperature_forecasting_cdmx.ipynb) | Riesgo climático físico | Tendencia de calentamiento (°C/década), estacionalidad (FFT, ACF/PACF), ADF + KPSS, selección ARIMA vs SARIMA por AIC, evaluación fuera de muestra contra un benchmark, grados-día (HDD/CDD) para derivados climáticos |
| [02 · Datos ESG y *screening*](notebooks/02_esg_data_and_screening.ipynb) | Integración ESG | Universo S&P 500 desde fuente abierta, métodos históricos para descargar scores ESG de Yahoo Finance (y cómo verificar si una fuente sigue viva), exclusión, *best-in-class* e inclinación por score |
| [03 · Optimización con restricción ESG](notebooks/03_esg_portfolio_optimization.ipynb) | Construcción de portafolios | Mínima varianza con score ESG mínimo, frontera eficiente riesgo–ESG, costo en diversificación de exigir un score más alto |

Los notebooks están ejecutados: se pueden leer directamente en GitHub con resultados y gráficas.

## Estructura

```
ESG-Workshop-Python/
├── notebooks/            # los tres ejercicios, ejecutados
├── src/esg_workshop/     # funciones reutilizables
│   ├── temperature.py    # carga de datos, tendencia, pruebas, SARIMA, métricas, grados-día
│   ├── esg_data.py       # universo, scores ESG (Yahoo, histórico), screening
│   ├── portfolio.py      # mínima varianza con restricción ESG, frontera riesgo–ESG
│   └── plotting.py       # estilo de gráficas
├── data/                 # datos abiertos / anonimizados (ver data/README.md)
├── tests/                # pruebas rápidas con pytest
└── REFERENCES.md         # fuentes, créditos y lecturas
```

## Cómo ejecutarlo

```bash
git clone https://github.com/libjai/ESG-Workshop-Python.git
cd ESG-Workshop-Python
python -m pip install -r requirements.txt
python -m pip install -e .        # opcional: permite `import esg_workshop` desde cualquier carpeta
jupyter lab notebooks/
```

Probado con Python 3.11, pandas 1.5, statsmodels 0.13 y versiones posteriores. Los notebooks 01 y 03 funcionan sin internet; el 02 descarga la lista del S&P 500 de Wikipedia.

Uso de las funciones fuera de los notebooks:

```python
from esg_workshop import temperature as tp

T = tp.load_monthly_temperature()["avg_temp_c"]
tp.warming_trend(T, start=1951)          # tendencia en °C por década con IC 95%
train, test = tp.train_test_split_last(T["1992":], test_months=24)
fc = tp.forecast(train, 24, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
tp.forecast_metrics(test, fc.forecast)   # RMSE, MAE, sesgo
```

## Nota sobre datos ESG de Yahoo Finance

Yahoo Finance publicó los *ESG Risk Ratings* de Sustainalytics entre 2018 y finales de 2025, y era la fuente gratuita más usada en ejemplos académicos. Hoy esos endpoints devuelven HTTP 404 (verificado el 27-sep-2026). El notebook 02 conserva el código como referencia del método, incluye una función que verifica la disponibilidad (`yahoo_esg_available`) y usa scores ilustrativos para los ejercicios.

## Aviso

Material educativo. Nada en este repositorio constituye recomendación de inversión. Los scores ESG de la muestra son ilustrativos y los emisores están anonimizados.

## Licencia

Código bajo licencia [MIT](LICENSE). Los datos de temperatura son de Berkeley Earth, con su propia licencia (ver [data/README.md](data/README.md)).
