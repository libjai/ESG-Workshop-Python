# Datos

| Archivo | Contenido | Fuente y licencia |
|---|---|---|
| `cdmx_monthly_temperature.csv` | Temperatura promedio mensual (°C) e incertidumbre (IC 95%) de la celda "Mexico" (20.09 N, 98.96 W), 1835-01 a 2013-09, 2,145 meses sin faltantes | Berkeley Earth, *GlobalLandTemperaturesByMajorCity*. Extracto sin modificar los valores. Licencia CC BY-NC 4.0 (uso no comercial, con atribución); uso comercial requiere permiso de Berkeley Earth. https://berkeleyearth.org/data/ |
| `sample_universe.csv` | 10 emisoras mexicanas **anonimizadas** (Stock1–Stock10): sector, industria, métricas fundamentales y un score ESG **ilustrativo** 0–10 (mayor = mejor) | Elaboración propia con fines educativos |
| `sample_prices.csv` | Precios de cierre diarios de las 10 emisoras, 2012-04-20 a 2013-04-23 (263 días) | Elaboración propia con fines educativos |

Para otra ciudad, descargar el archivo completo de Berkeley Earth y usar
`esg_workshop.temperature.load_berkeley_city("GlobalLandTemperaturesByMajorCity.csv", "Monterrey")`.
