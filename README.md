# Monitoreo meteorológico para operaciones agrícolas

**Equipo 5 · Unidad 2 · Preparación de los datos**

## Caso de estudio y objetivo

Una cooperativa agrícola necesita identificar periodos de lluvia, heladas y temperaturas extremas para planificar sus actividades.

Este proyecto prepara datos meteorológicos del KNMI para desarrollar posteriormente un dashboard con consultas por estación y periodo.

## Dataset

- **Periodo:** 2016–2025.
- **Estaciones:** 30.
- **Registros:** 109,497.
- **Variables principales:** temperatura, precipitación, humedad, viento y evapotranspiración.
- **Granularidad:** una observación por estación y fecha disponible.

## Contenido del repositorio

- `dataset_crudo.csv`: datos consolidados con unidades y códigos originales.
- `dataset_agricola_diario_2016_2025.csv`: datos preprocesados e indicadores derivados.
- `descargar_knmi_diario.py`: script de extracción y preparación.
- `knmi_*.txt`: archivos originales del KNMI.

## Preparación de datos

El script utiliza **Python, pandas y requests** para descargar y consolidar los datos, validar fechas y duplicados, convertir unidades y generar indicadores de heladas y calidad.

Los faltantes se conservan como nulos y los valores extremos no se eliminan automáticamente, porque pueden representar fenómenos meteorológicos reales. Los resultados se almacenan en archivos CSV.

## Estado del proyecto

Este repositorio corresponde al avance de preparación de datos. El Data Warehouse está diseñado; su implementación y el desarrollo del dashboard son las siguientes etapas.

## Fuente

[Datos meteorológicos diarios del KNMI](https://www.knmi.nl/nederland-nu/klimatologie/daggegevens/)
