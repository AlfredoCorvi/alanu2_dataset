# Dataset escolar: original vs. limpio

Práctica de limpieza de datos con calificaciones de un grupo escolar.

## Cómo reproducirlo (Python 3.10+, con `pandas` y `numpy`)
```
pip install pandas numpy
python scripts/generar_datos_crudos.py   # crea data/raw/calificaciones_raw.csv
python scripts/limpiar_dataset.py        # crea data/clean/calificaciones_limpio.csv
```
Al usar semilla fija, siempre se obtienen exactamente los mismos archivos.

## Diccionario de datos
| Columna | Tipo | Descripción |
|---|---|---|
| matricula | nominal | Identificador del alumno (A001 a A030) |
| nombre | texto | Nombre del alumno |
| sexo | nominal | Masculino / Femenino / No especificado |
| carrera | nominal | Sistemas, Administración, Contaduría |
| materia | nominal | Matemáticas, Programación, Bases de Datos, Inglés |
| periodo | temporal | 2024-1 o 2024-2 |
| parcial1, parcial2 | continuo | Calificaciones de 0 a 100 |
| asistencia_pct | continuo | Porcentaje de asistencia, de 0 a 100 |
| promedio_final (solo limpio) | continuo | Media de parcial1 y parcial2 |
| aprobado (solo limpio) | binario | 1 si promedio_final >= 70, si no 0 |

## Problemas del dataset original y cómo se corrigieron
| # | Problema en el original | Cantidad | Corrección |
|---|---|---|---|
| 1 | Filas duplicadas | 3 | Eliminar duplicados |
| 2 | Nombres en MAYÚSCULAS, minúsculas o con espacios de más | 21 | Recortar espacios y unificar mayúsculas (`Ana López`) |
| 3 | Sexo escrito de varias formas (`M`, `H`, `masculino`, `Mujer`, vacío…) | 7 variantes | Unificar a Masculino / Femenino / No especificado |
| 4 | Carrera escrita de varias formas (`SISTEMAS`, `Ing. Sistemas`, `admin`…) | 8 variantes | Unificar a 3 carreras |
| 5 | Materia con mayúsculas o espacios distintos (`PROGRAMACIÓN`) | 10 variantes | Unificar a 4 materias |
| 6 | Calificaciones con coma decimal (`85,5`) | 5 | Convertir a número (`85.5`) |
| 7 | Asistencia con símbolo % (`90%`) | 8 | Quitar el símbolo y convertir a número |
| 8 | Valores fuera de rango (calificación o asistencia > 100 o negativa) | 3 | Marcar como nulo y tratar como faltante |
| 9 | Valores faltantes (vacío o `NA`) en parciales y asistencia | 19 (ya incluye los del punto 8) | Reemplazar por la mediana de la materia |

Después de la limpieza se calcularon `promedio_final` y `aprobado`.
