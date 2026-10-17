# Monitoreo de condiciones meteorológicas para operaciones agrícolas

**Equipo 5 · Extracción de Conocimientos de Bases de Datos**  
**Fase 1: obtención y revisión inicial del dataset**

## Caso de estudio

Una cooperativa agrícola necesita identificar periodos de lluvia, heladas y temperaturas extremas para apoyar la planificación de sus actividades. El proyecto utiliza observaciones meteorológicas diarias por estación del Instituto Real de Meteorología de los Países Bajos (KNMI).

## Objetivo

Preparar una base de datos meteorológicos que permita analizar las condiciones registradas por fecha y estación. En etapas posteriores se realizará la transformación de los datos y su carga en un data warehouse con esquema estrella.

## Contenido de esta fase

El archivo `dataset_original_consolidado.csv` reúne los registros descargados en una tabla. Conserva las unidades y los códigos de las variables de origen. No se rellenaron las celdas vacías con datos aleatorios.

| Característica | Resultado |
|---|---|
| Periodo global | 1 de enero de 2016 al 31 de diciembre de 2025 |
| Estaciones | 30 |
| Registros | 109,497 |
| Columnas | 11 |
| Granularidad | Una observación diaria por estación |
| Formato | CSV |
| Clave de cada registro | `STN` + `YYYYMMDD` |

Estas cifras corresponden al CSV revisado para esta entrega. El periodo global no implica que todas las estaciones tengan la misma cobertura.
