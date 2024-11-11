# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:17:25 2024

@author: USER
"""

import pandas as pd
from datetime import datetime
import numpy as np
import sys

# Parámetros configurables
path = 'C:/DataForex/'
symbol = 'EURUSD'
hist_file = f'{path}Data_hist_eurusd_m1_201101-202406.txt'
lmax_file = f'{path}Data_LMAX_08072024.txt'
mt4_file = f'{path}Data_DWNX_08072024.txt'  # Coloca el nombre de tu archivo data de Metatrader 4
output_csv = f'{path}{symbol}_data_mergefilled.csv'
output_txt = f'{path}{symbol}_data_mergefilled.txt'
start_analysis_date = '2011-01-01'

start_time = datetime.now()

# Función para cargar datos con manejo de errores
def load_data(file_path, columns, date_format, skip_rows=0):
    try:
        df = pd.read_csv(file_path, names=columns, header=None, skiprows=skip_rows)
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=date_format, errors='coerce')
        df.drop(columns=['Date', 'Time'], inplace=True)
        df.set_index('datetime', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except Exception as e:
        print(f"Error al cargar {file_path}: {e}")
        return pd.DataFrame()

# Cargar datos
columnas_hist = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
columnas_lmax = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
columnas_mt4 = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

data_hist = load_data(hist_file, columnas_hist, '%Y.%m.%d %H:%M', skip_rows=0)
data_lmax = load_data(lmax_file, columnas_lmax, '%d/%m/%Y %H:%M:%S', skip_rows=1)
data_mt4 = load_data(mt4_file, columnas_mt4, '%Y.%m.%d %H:%M', skip_rows=0)

# Verificar si los datos se cargaron correctamente
if data_hist.empty or data_lmax.empty or data_mt4.empty:
    print("Error: No se pudieron cargar los datos correctamente.")
    sys.exit()

# Filtrar los datos a partir de la fecha de inicio del análisis
data_hist = data_hist[data_hist.index >= start_analysis_date]
data_lmax = data_lmax[data_lmax.index >= start_analysis_date]
data_mt4 = data_mt4[data_mt4.index >= start_analysis_date]

# Eliminar índices duplicados
data_hist = data_hist[~data_hist.index.duplicated(keep='first')]
data_lmax = data_lmax[~data_lmax.index.duplicated(keep='first')]
data_mt4 = data_mt4[~data_mt4.index.duplicated(keep='first')]

# Verificar que las fechas no sean NaT
if data_hist.index.min() is pd.NaT or data_hist.index.max() is pd.NaT:
    print("Error: Fechas no válidas en data_hist")
    print(data_hist[data_hist.index.isna()])  # Mostrar filas con fechas no válidas
    sys.exit()
if data_lmax.index.min() is pd.NaT or data_lmax.index.max() is pd.NaT:
    print("Error: Fechas no válidas en data_lmax")
    print(data_lmax[data_lmax.index.isna()])  # Mostrar filas con fechas no válidas
    sys.exit()
if data_mt4.index.min() is pd.NaT or data_mt4.index.max() is pd.NaT:
    print("Error: Fechas no válidas en data_mt4")
    print(data_mt4[data_mt4.index.isna()])  # Mostrar filas con fechas no válidas
    sys.exit()

# Identificar el rango completo de fechas
start_date = min(data_hist.index.min(), data_lmax.index.min(), data_mt4.index.min())
end_date = max(data_hist.index.max(), data_lmax.index.max(), data_mt4.index.max())
all_dates = pd.date_range(start=start_date, end=end_date, freq='T')

# Crear un DataFrame con todas las fechas posibles
all_dates_df = pd.DataFrame(index=all_dates)

# Eliminar sábados, domingos, viernes a partir de las 22:00 y lunes desde las 00:00 hasta la 01:00
all_dates_df = all_dates_df[~(
    (all_dates_df.index.weekday == 5) |  # Sábados
    (all_dates_df.index.weekday == 6) |  # Domingos
    ((all_dates_df.index.weekday == 4) & (all_dates_df.index.hour >= 22)) |  # Viernes a partir de las 22:00
    ((all_dates_df.index.weekday == 0) & (all_dates_df.index.hour < 1))  # Lunes desde las 00:00 hasta la 01:00
)]

# Unir los datos con all_dates_df como índice común
data_hist_full = all_dates_df.join(data_hist, how='left', rsuffix='_hist')
data_lmax_full = all_dates_df.join(data_lmax, how='left', rsuffix='_lmax')
data_mt4_full = all_dates_df.join(data_mt4, how='left', rsuffix='_mt4')

# Asegurarse de que las series Volume tengan el mismo índice
data_hist_full['Volume_hist'] = data_hist_full['Volume']
data_lmax_full['Volume_lmax'] = data_lmax_full['Volume']
data_mt4_full['Volume_mt4'] = data_mt4_full['Volume']

# Rellenar los huecos en data_hist_full usando data_lmax_full y luego data_mt4_full
data_filled = data_hist_full.combine_first(data_lmax_full).combine_first(data_mt4_full)

# Manejar el volumen por separado sin fusionarlo
data_filled['Volume'] = np.where(data_hist_full['Volume_hist'].notna(), data_hist_full['Volume_hist'],
                                 np.where(data_lmax_full['Volume_lmax'].notna(), data_lmax_full['Volume_lmax'], data_mt4_full['Volume_mt4']))

# Rellenar NaNs en la columna Volume y convertirla a int64
data_filled['Volume'] = data_filled['Volume'].fillna(0).astype('int64')

# Eliminar las columnas de volumen innecesarias
data_filled.drop(columns=['Volume_hist', 'Volume_lmax', 'Volume_mt4'], inplace=True)

# Verificar NaN y saltos temporales
print("NaN en el DataFrame:", data_filled.isnull().values.any())
print("Saltos temporales:", not data_filled.index.is_monotonic_increasing or data_filled.index.to_series().diff().max() > pd.Timedelta(minutes=1))

# Formatear el índice datetime y convertirlo a columna
data_filled.reset_index(inplace=True)
data_filled.rename(columns={'index': 'datetime'}, inplace=True)
data_filled['datetime'] = data_filled['datetime'].dt.strftime('%Y.%m.%d,%H:%M')

# Reordenar las columnas
data_filled = data_filled[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Función para guardar los datos en formato específico
def save_data(df, file_path):
    with open(file_path, 'w') as f:
        for _, row in df.iterrows():
            date, time = row['datetime'].split(',')
            line = f"{date},{time},{row['Open']:.5f},{row['High']:.5f},{row['Low']:.5f},{row['Close']:.5f},{row['Volume']}\n"
            f.write(line)

# Guardar los resultados en CSV y TXT (ambos separados por comas)
save_data(data_filled, output_csv)
save_data(data_filled, output_txt)

print(f"Datos guardados en '{output_csv}' y '{output_txt}'.")

# Medir el tiempo de finalización
end_time_exec = datetime.now()
print(f"Duración: {end_time_exec - start_time}")
