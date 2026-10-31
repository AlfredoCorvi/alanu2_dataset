"""KNMI diario, 2016-2025. Python 3.9+. Instalar: py -m pip install pandas requests

Fuente y contrato: https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
Esta via de descarga tiene cierre anunciado para finales de 2026.
Conserva originales, faltantes y trazas de precipitacion. No imputa datos.
"""
from io import StringIO
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

INICIO = 2016
FIN = 2025
ESTACIONES = [215, 235, 240, 249, 251, 257, 260, 267, 269, 270,
              273, 275, 277, 278, 279, 280, 283, 286, 290, 310,
              319, 344, 348, 350, 356, 370, 375, 377, 380, 391]
VARIABLES = ['TG', 'TN', 'TX', 'T10N', 'RH', 'DR', 'UG', 'FG', 'EV24']
URL = 'https://www.daggegevens.knmi.nl/klimatologie/daggegevens'
BASE = Path(__file__).resolve().parent / 'dataset_diario_knmi_2016_2025'


def interpretar(texto, anio):
    cabecera = next((line.lstrip('# ').strip() for line in texto.splitlines()
                     if re.match(r'^\s*#\s*STN\s*,\s*YYYYMMDD', line)), None)
    if not cabecera:
        raise ValueError('La respuesta no contiene la cabecera KNMI esperada.')
    columnas = [c.strip() for c in cabecera.split(',')]
    df = pd.read_csv(StringIO(texto), comment='#', names=columnas,
                     skipinitialspace=True, na_values=['', ' '])
    if df.empty or not set(['STN', 'YYYYMMDD'] + VARIABLES).issubset(df.columns):
        raise ValueError('Respuesta vacia o variables incompletas.')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='raise')
    fechas = pd.to_datetime(df['YYYYMMDD'].astype('Int64').astype(str),
                            format='%Y%m%d', errors='raise')
    if not fechas.dt.year.eq(anio).all() or not df.STN.isin(ESTACIONES).all():
        raise ValueError('La respuesta contiene fechas o estaciones inesperadas.')
    if df.duplicated(['STN', 'YYYYMMDD']).any():
        raise ValueError('Hay claves estacion-fecha duplicadas; revisar originales.')
    return df


def main():
    originales = BASE / 'originales'
    originales.mkdir(parents=True, exist_ok=True)
    sesion = requests.Session()
    retry = Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['POST'], respect_retry_after_header=True)
    sesion.mount('https://', HTTPAdapter(max_retries=retry))
    partes, manifest = [], []
    for anio in range(INICIO, FIN + 1):
        parametros = {'start': f'{anio}0101', 'end': f'{anio}1231',
                      'stns': ':'.join(map(str, ESTACIONES)), 'vars': ':'.join(VARIABLES)}
        firma = hashlib.sha256(json.dumps(parametros, sort_keys=True).encode()).hexdigest()[:12]
        archivo = originales / f'knmi_{anio}_{firma}.txt'
        if archivo.exists():
            print(f'{anio}: reutilizando descarga guardada...', flush=True)
            texto = archivo.read_text(encoding='utf-8')
            df = interpretar(texto, anio)
        else:
            print(f'{anio}: descargando datos diarios...', flush=True)
            respuesta = sesion.post(URL, data=parametros, timeout=(30, 240))
            respuesta.raise_for_status()
            texto = respuesta.text
            df = interpretar(texto, anio)
            temporal = archivo.with_suffix('.tmp')
            temporal.write_text(texto, encoding='utf-8')
            temporal.replace(archivo)
            time.sleep(1)
        partes.append(df)
        manifest.append({'anio': anio, 'parametros': parametros,
                         'archivo': archivo.name, 'filas': len(df),
                         'sha256': hashlib.sha256(archivo.read_bytes()).hexdigest(),
                         'archivo_guardado_utc': datetime.fromtimestamp(
                             archivo.stat().st_mtime, timezone.utc).isoformat()})
        print(f'  {len(df):,} filas; {df.STN.nunique()} estaciones.', flush=True)

    crudo = pd.concat(partes, ignore_index=True).sort_values(['STN', 'YYYYMMDD'])
    if crudo.duplicated(['STN', 'YYYYMMDD']).any():
        raise ValueError('Duplicados en el consolidado; no se exportara.')
    crudo.to_csv(BASE / 'dataset_original_consolidado.csv', index=False, encoding='utf-8-sig')

    salida = pd.DataFrame({'estacion_id': crudo.STN.astype('Int64'),
                           'fecha': pd.to_datetime(crudo.YYYYMMDD.astype('Int64').astype(str),
                                                   format='%Y%m%d').dt.strftime('%Y-%m-%d')})
    conversiones = {
        'TG': ('temperatura_media_c', 10), 'TN': ('temperatura_minima_c', 10),
        'TX': ('temperatura_maxima_c', 10), 'T10N': ('temperatura_minima_10cm_c', 10),
        'DR': ('duracion_precipitacion_h', 10), 'UG': ('humedad_media_pct', 1),
        'FG': ('viento_medio_m_s', 10), 'EV24': ('evapotranspiracion_potencial_mm', 10)}
    for origen, (destino, divisor) in conversiones.items():
        salida[destino] = crudo[origen] / divisor
    # -1 significa traza (<0.05 mm), NO precipitacion negativa ni dato faltante.
    salida['precipitacion_traza'] = crudo.RH.eq(-1).astype('boolean').mask(crudo.RH.isna())
    salida['precipitacion_mm'] = crudo.RH.mask(crudo.RH.eq(-1), 0) / 10
    # Cero es una convencion numerica SOLO para trazas; el indicador las distingue.
    salida['helada_aire'] = crudo.TN.lt(0).astype('boolean').mask(crudo.TN.isna())
    salida['temperaturas_inconsistentes'] = (
        (crudo.TN > crudo.TX) | (crudo.TG < crudo.TN) | (crudo.TG > crudo.TX)
    ).astype('boolean').mask(crudo[['TN', 'TG', 'TX']].isna().any(axis=1))
    salida['numero_variables_faltantes'] = crudo[VARIABLES].isna().sum(axis=1)
    salida.to_csv(BASE / 'dataset_agricola_diario_2016_2025.csv',
                   index=False, encoding='utf-8-sig')

    faltantes = pd.DataFrame({'variable': VARIABLES,
                              'faltantes': crudo[VARIABLES].isna().sum().values})
    faltantes['porcentaje'] = (100 * faltantes.faltantes / len(crudo)).round(2)
    faltantes.to_csv(BASE / 'reporte_faltantes.csv', index=False, encoding='utf-8-sig')
    dias = len(pd.date_range(f'{INICIO}-01-01', f'{FIN}-12-31'))
    cobertura = (salida.groupby('estacion_id').agg(
        filas=('fecha', 'size'), primera_fecha=('fecha', 'min'), ultima_fecha=('fecha', 'max'))
        .reindex(ESTACIONES))
    cobertura['filas'] = cobertura['filas'].fillna(0).astype(int)
    cobertura['dias_sin_fila_en_periodo'] = dias - cobertura['filas']
    cobertura.to_csv(BASE / 'cobertura_estaciones.csv', encoding='utf-8-sig')
    (BASE / 'procedencia.json').write_text(json.dumps(
        {'fuente': URL, 'generado_utc': datetime.now(timezone.utc).isoformat(),
         'descargas': manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
    (BASE / 'LEEME.txt').write_text(
        'Observaciones diarias KNMI. Una fila por estacion y fecha disponible.\n'
        'No se inventaron ni imputaron valores. Vacios significan datos ausentes.\n'
        'Los originales TXT conservan definiciones, unidades y metadatos de estaciones.\n'
        'Las unidades del CSV agricola estan convertidas; el CSV original no esta convertido.\n'
        'RH=-1 es traza (<0.05 mm): se representa como 0 mm y precipitacion_traza=True.\n'
        'Helada_aire significa minima diaria del aire <0 C; no mide dano al cultivo.\n'
        'No se impone un umbral universal de calor extremo: depende del cultivo.\n'
        'Cobertura muestra dias sin fila, ademas del reporte de celdas faltantes.\n'
        'Las estaciones pueden no cubrir todo el periodo (por ejemplo Arcen, 391).\n'
        'KNMI controla y puede corregir datos; no todas las series son homogeneizadas.\n'
        'No inferir tendencias de cambio climatico sin revisar la homogeneizacion.\n'
        'Documentacion: https://www.knmi.nl/nederland-nu/klimatologie/daggegevens/\n'
        'Via de descarga con cierre anunciado a finales de 2026.\n', encoding='utf-8')
    print(f'\nTERMINADO: {len(salida):,} filas, {salida.estacion_id.nunique()} estaciones.')
    print(f'Carpeta: {BASE}')
    print('Archivo principal: dataset_agricola_diario_2016_2025.csv')
    print('Los datos faltantes se conservaron; revisa los reportes.')


if __name__ == '__main__':
    try:
        main()
    except (requests.RequestException, ValueError, OSError) as exc:
        print(f'\nERROR: {exc}')
        print('No se completo esta ejecucion. Las descargas validas se conservan para reintentar.')
        raise SystemExit(1)
