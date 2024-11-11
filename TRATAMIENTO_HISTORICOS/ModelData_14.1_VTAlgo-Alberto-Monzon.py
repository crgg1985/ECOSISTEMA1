"""
Created on Sun Jun  2 18:57 2024

@author: Fernando Monzón
Modificación realizada sobre el codigo propuesto de @author: VTAlgo & Alberto Granero 
"""
"""
Este script realiza las siguientes tareas:
    
1.- Detecta los encabezados de 13 formatos de distintos proveedores de datos
2.- Carga los datos desde un archivo CSV y maneja posibles errores en las fechas.
3.- Ajusta los datos de acuerdo con la zona horaria especificada.
4.- Calcula las diferencias de tiempo entre registros y detecta gaps.
5.- Filtra los gaps válidos, excluyendo los periodos de cierre del mercado.
6.- Ajusta los días y horas para que los datos sean más fáciles de interpretar.
7.- Exporta los resultados a archivos Excel.
8.- Genera un heatmap de los gaps y lo guarda en un archivo HTML.
9.- Genera una carpeta independiente con el estudio realizado para comparar ejecuciones
9.- Solicita al usuario una diferencia temporal y genera un informe de los gaps ubicados en ese rango 
"""

import os
from datetime import datetime, time, timedelta
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tabulate import tabulate

# Definir la zona horaria en formato UTC+X
timezone_offset = 0  # Ejemplo: UTC+0 (GMT)
symbol = 'EURUSD'  # Define el símbolo aquí
proveedor = "LMAX_MC"
tipo_gap = "Gap_Temp"

start_time = datetime.now()  # Guarda el tiempo de inicio

# Preparación de rutas de trabajo
#file_work = 'EURUSD1_DWNX.csv'  # Nombre del archivo a cargar
#file_work = 'EURUSD_M1_MT5.csv'  
file_work = 'EUR_USD 1 Minute.txt'
#file_work = 'tradestation.txt'

file_main = 'C:/DataForex/'  # Ruta principal de trabajo
file_path = f"{file_main}/{file_work}"  # Ruta completa del archivo a cargar

def detect_format(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        second_line = file.readline().strip()

    # Detectar Multicharts (7 campos con cabeceras, delimitados por comas)
    if "," in first_line and len(first_line.split(',')) == 7 and "Date" in first_line:
        return 'Multicharts'

    # Detectar Metatrader 4 bar (7 campos sin cabeceras, delimitados por comas)
    if "," in second_line and len(second_line.split(',')) == 7 and not any(c.isalpha() for c in second_line):
        return 'Metatrader 4 bar'

    # Detectar Metatrader 5 (9 campos con cabeceras, delimitados por tabulaciones)
    if "\t" in first_line and len(first_line.split('\t')) == 9 and "DATE" in first_line:
        return 'Metatrader 5'

    # Detectar Tradestation (8 campos con cabeceras, delimitados por comas)
    if "," in first_line and len(first_line.split(',')) == 8 and "Date" in first_line:
        return 'Tradestation'

    # Detectar Metatrader 4 - tick (5 campos sin cabeceras, delimitados por comas)
    if "," in second_line and len(second_line.split(',')) == 5 and second_line.split(',')[-1] == '0':
        return 'Metatrader 4 - tick'

    # Detectar Ninja Trader - tick (3 campos sin cabeceras, delimitados por comas)
    if "," in second_line and len(second_line.split(',')) == 3:
        return 'Ninja Trader - tick'

    # Detectar Ninja Trader - bar (6 campos sin cabeceras, delimitados por comas)
    if "," in second_line and len(second_line.split(',')) == 6 and len(second_line.split(',')[0]) == 15:
        return 'Ninja Trader - bar'

    # Detectar Amibroker tick (7 campos sin cabeceras, delimitados por comas)
    if "," in second_line and len(second_line.split(',')) == 7 and second_line.split(',')[2] == second_line.split(',')[3] == second_line.split(',')[4] == second_line.split(',')[5]:
        return 'Amibroker tick'

    # Detectar Amibroker bar (7 campos sin cabeceras, delimitados por comas)
    if "," in second_line and len(second_line.split(',')) == 7:
        return 'Amibroker bar'

    # Detectar JForex tick (5 campos con cabeceras, delimitados por comas)
    if "," in first_line and len(first_line.split(',')) == 5 and "Time (UTC)" in first_line:
        return 'JForex tick'

    # Detectar Forextester (8 campos con cabeceras, delimitados por comas)
    if "," in first_line and len(first_line.split(',')) == 8 and "<TICKER>" in first_line:
        return 'Forextester'

    # Detectar Dukascopy tick (5 campos con cabeceras, delimitados por comas)
    if "," in first_line and len(first_line.split(',')) == 5 and "Local time" in first_line:
        return 'Dukascopy tick'

    # Detectar Dukascopy bar (6 campos con cabeceras, delimitados por comas)
    if "," in first_line and len(first_line.split(',')) == 6 and "Local time" in first_line:
        return 'Dukascopy bar'
    
    raise ValueError("Formato de archivo no reconocido")

def get_data(file_path, timezone_offset):
    """
    Carga los datos desde un archivo CSV, maneja posibles errores en las fechas,
    y ajusta los datos de acuerdo con la zona horaria especificada.

    Args:
        file_path (str): Ruta del archivo CSV a cargar.
        timezone_offset (int): Desplazamiento de la zona horaria en horas.

    Returns:
        pd.DataFrame: DataFrame con los datos ajustados.
    """
    platform = detect_format(file_path)
    if platform == 'Multicharts':
        data = pl.read_csv(file_path, 
                        has_header=True,
                        new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        format_conversion_dates = "%d/%m/%Y"
        format_conversion_times = "%H:%M:%S"
    elif platform == 'Metatrader 4 bar':
        data = pl.read_csv(file_path, 
                        has_header=False,
                        new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        format_conversion_dates = "%Y.%m.%d"
        format_conversion_times = "%H:%M"
    elif platform == 'Metatrader 5':
        data = pl.read_csv(file_path, 
                        separator='\t',
                        has_header=True,
                        new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VOL', 'SPREAD'])
        data = data.drop(['VOL', 'SPREAD'])  # Eliminar columnas no necesarias
        format_conversion_dates = "%Y.%m.%d"
        format_conversion_times = "%H:%M:%S"
    elif platform == 'Tradestation':
        data = pl.read_csv(file_path, 
                        has_header=True,
                        new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Up', 'Down'])
        data = data.rename({'Up': 'Volume'}).drop(['Down'])  # Renombrar columna y eliminar 'Down'
        format_conversion_dates = "%m/%d/%Y"
        format_conversion_times = "%H:%M"
    elif platform == 'Metatrader 4 - tick':
        data = pl.read_csv(file_path, 
                        has_header=False,
                        new_columns=['DateTime', 'Bid', 'Ask', 'Volume', 'Zero'])
        data = data.drop(['Zero'])  # Eliminar columna no necesaria
        format_conversion_dates = "%Y.%m.%d %H:%M:%S.%f"
    elif platform == 'Ninja Trader - tick':
        data = pl.read_csv(file_path, 
                        has_header=False,
                        new_columns=['DateTime', 'Bid', 'Volume'])
        format_conversion_dates = "%Y.%m.%d %H:%M:%S.%f"
    elif platform == 'Ninja Trader - bar':
        data = pl.read_csv(file_path, 
                        has_header=False,
                        new_columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        format_conversion_dates = "%Y%m%d %H%M%S"
    elif platform == 'Amibroker tick':
        data = pl.read_csv(file_path, 
                        has_header=False,
                        new_columns=['Date', 'Time', 'Bid1', 'Bid2', 'Bid3', 'Bid4', 'Volume'])
        data = data.drop(['Bid2', 'Bid3', 'Bid4'])  # Eliminar columnas duplicadas no necesarias
        format_conversion_dates = "%Y%m%d"
        format_conversion_times = "%H%M%S"
    elif platform == 'Amibroker bar':
        data = pl.read_csv(file_path, 
                        has_header=False,
                        new_columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        format_conversion_dates = "%Y%m%d"
        format_conversion_times = "%H%M%S"
    elif platform == 'JForex tick':
        data = pd.read_csv(file_path, 
                        delimiter=',',
                        decimal=',',
                        names=['DateTime', 'Ask', 'Bid', 'AskVolume', 'BidVolume'],
                        skiprows=1)
        # Convertir comas a puntos decimales
        for column in ['Ask', 'Bid', 'AskVolume', 'BidVolume']:
            data[column] = data[column].str.replace(',', '.').astype(float)
        format_conversion_dates = "%Y.%m.%d %H:%M:%S.%f"
    elif platform == 'Forextester':
        data = pl.read_csv(file_path, 
                        has_header=True,
                        new_columns=['Ticker', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        format_conversion_dates = "%Y%m%d"
        format_conversion_times = "%H%M%S"
    elif platform == 'Dukascopy tick':
        data = pd.read_csv(file_path, 
                        delimiter=',',
                        names=['DateTime', 'Ask', 'Bid', 'AskVolume', 'BidVolume'],
                        skiprows=1)
        format_conversion_dates = "%d.%m.%Y %H:%M:%S.%f"
    elif platform == 'Dukascopy bar':
        data = pd.read_csv(file_path, 
                        delimiter=',',
                        names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                        skiprows=1)
        format_conversion_dates = "%d.%m.%Y %H:%M:%S.%f"
    else:
        raise ValueError(f"Formato de archivo no reconocido: {platform}")

    # Convertir las columnas de fechas y tiempos al formato adecuado
    if 'Date' in data.columns:
        data = data.with_columns(
            pl.col("Date").str.strptime(pl.Date, format_conversion_dates, strict=False).alias("Date"),
            pl.col("Time").str.strptime(pl.Time, format_conversion_times, strict=False).alias("Time")
        )
        data = data.with_columns(pl.concat_str(["Date", pl.lit(" "), "Time"]).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S",strict=False).alias("Datetime"))
        data = data.drop(["Date", "Time"])
    else:
        data = data.with_columns(pl.col("DateTime").str.strptime(pl.Datetime, format_conversion_dates, strict=False).alias("Datetime"))
        data = data.drop(["DateTime"])

    # Ajustar los datos a la zona horaria especificada
    offset = int(timedelta(hours=timezone_offset).total_seconds())
    data = data.with_columns(pl.col("Datetime").dt.offset_by(f"-{offset}s").dt.convert_time_zone("UTC"))
    data = data.to_pandas().set_index('Datetime')

    #execution_time = datetime.now() - start_time
    #print(f"El archivo a integrar es: {platform}. Dataframe ingestado y preprocesado en {execution_time}")
    
    return data, platform

# Cargar los datos desde el archivo especificado
df, platform = get_data(file_path, timezone_offset)
if df is None:
    raise ValueError("Error al cargar los datos. Por favor, verifica el archivo y vuelve a intentarlo.")

# Preparar el nombre de la carpeta de salida con la plataforma incluida
exec_path = f"{file_main}/{start_time.strftime('%y%m%d%H%M')}_{platform}"

# Crear la carpeta de trabajo si no existe
os.makedirs(exec_path, exist_ok=True)

# Definición de nombres de archivos de salida usando el nuevo exec_path
heatmap_file = f"{exec_path}/{symbol}_{proveedor}_{tipo_gap}_heatmap_gaps.html"
scatter_file = f"{exec_path}/{symbol}_{proveedor}_{tipo_gap}_scatter_gaps.html"
freq_file = f"{exec_path}/{symbol}_{proveedor}_{tipo_gap}_time_diff_frecuencies.xlsx"
output_filename = f"{exec_path}/{symbol}_{proveedor}_{tipo_gap}_output.txt"

# Añadir columnas adicionales al DataFrame
df['Day_of_Week'] = df.index.day_name()
market_symbol = 'EUR_USD'
df['Weekday'] = df.index.weekday
df['Time'] = df.index.time

# Definir las horas de apertura y cierre del mercado en UTC
local_open_time = time(17, 0)  # 17:00 hora local
local_close_time = time(17, 0)  # 17:00 hora local
offset = timedelta(hours=timezone_offset)
utc_open_time = (datetime.combine(datetime.today(), local_open_time) - offset).time()
utc_close_time = (datetime.combine(datetime.today(), local_close_time) - offset).time()

# Identificar las horas de mercado abierto
df['Market_Open'] = np.where(
    ((df['Weekday'] == 6) & (df['Time'] >= utc_open_time)) |
    ((df['Weekday'] == 4) & (df['Time'] < utc_close_time)) |
    (df['Weekday'].between(0, 3)),
    True, False
)

# Calcular las diferencias de tiempo entre registros y detectar gaps
df['Time_Diff'] = df.index.to_series().diff().dt.total_seconds() / 60
df['Is_Gap'] = (df['Time_Diff'] > 1)

# Marcar periodos de cierre del mercado
market_closure_start = (df['Weekday'] == 4) & (df['Time'] >= utc_close_time)
market_closure_end = (df['Weekday'] == 6) & (df['Time'] < utc_open_time)
df['Market_Closure'] = market_closure_start | market_closure_end.shift(fill_value=False)
df['Valid_Gap'] = df['Is_Gap'] & ~df['Market_Closure']

# Excluir registros con Time_Diff de 1 minuto
df = df[df['Time_Diff'] != 1]

# Ajustar los días y horas para facilitar la interpretación
df['Adjusted_Day'] = np.where(
    df['Time'] < utc_open_time,
    df.index.day_name(),
    (df.index - pd.Timedelta(days=1)).day_name()
)
df['Adjusted_Hour'] = df.index.hour

# Identificar el domingo desde las 17:00 hasta las 23:59 como un día separado
df['Adjusted_Day'] = np.where(
    (df['Weekday'] == 6) & (df['Time'] >= utc_open_time),
    'Sunday',
    df['Adjusted_Day']
)

# Filtrar gaps válidos excluyendo fines de semana
df_filtered = df[(df['Weekday'] != 5) & ~((df['Weekday'] == 6) & (df['Time'] < utc_open_time))]

# Obtener estadísticas descriptivas sobre los gaps temporales
descriptive_stats = df_filtered['Time_Diff'].describe()
print(descriptive_stats)

# Agrupar y contar los gaps para visualización
gaps_grouped = df_filtered[df_filtered['Valid_Gap']].groupby(['Adjusted_Day', 'Adjusted_Hour', 'Time_Diff']).size()
gaps_grouped = gaps_grouped.reset_index(name='Count')

# Crear y configurar la tabla pivot para el mapa de calor
pivot_table = gaps_grouped.pivot_table(index='Adjusted_Day', columns='Adjusted_Hour', values='Count', fill_value=0)
ordered_days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
pivot_table = pivot_table.reindex(ordered_days)

# Crear el mapa de calor con Plotly
heatmap_fig = px.imshow(pivot_table, 
                        labels=dict(x="Hour", y="Day of Week", color="Count"),
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        color_continuous_scale="YlOrBr",
                        title=f"{os.path.splitext(file_work)[0]}_API_{proveedor}_Heatmap of Hourly Gap Frequencies")


# Guardar el heatmap como un archivo HTML
heatmap_fig.write_html(heatmap_file)

# Identificar outliers (por ejemplo, Time_Diff > 100 minutos)
outliers = df_filtered[df_filtered['Time_Diff'] > 100]
print("Outliers encontrados:")
print(outliers[['Time_Diff']])

# Crear el diagrama de dispersión con Plotly
scatter_fig = go.Figure()

# Agregar todos los gaps
scatter_fig.add_trace(go.Scatter(x=df_filtered.index, y=df['Time_Diff'], 
                                 mode='markers', 
                                 name='Gaps',
                                 marker=dict(color='blue', opacity=0.5)))

# Agregar los outliers
scatter_fig.add_trace(go.Scatter(x=outliers.index, y=outliers['Time_Diff'], 
                                 mode='markers', 
                                 name='Outliers',
                                 marker=dict(color='red', size=10, symbol='x'),
                                 hovertext=outliers.index.strftime('%Y-%m-%d %H:%M'),
                                 hoverinfo='text'))

scatter_fig.update_layout(title='Scatter Plot of Time Gaps (Time_Diff) in EUR/USD Data',
                          xaxis_title='Datetime',
                          yaxis_title='Time_Diff (minutes)',
                          legend_title='Legend',
                          yaxis=dict(dtick=100, range=[0, df_filtered['Time_Diff'].max()]))  # Ajuste del eje Y con incrementos de 100 en 100

# Guardar el scatter plot como un archivo HTML
scatter_fig.write_html(scatter_file)

# Exportar la tabla de frecuencias a un archivo Excel
output_file_path_freq = f"{exec_path}/{symbol}_{proveedor}_{tipo_gap}_time_diff_frequencies.xlsx"
time_diff_counts = df['Time_Diff'].value_counts().sort_index()
time_diff_counts.to_excel(output_file_path_freq)
print(f"La tabla de frecuencias de Time_Diff ha sido exportada a: {output_file_path_freq}")

# Crear y configurar la tabla pivot para los detalles en el excel
pivot_table2 = gaps_grouped.pivot_table(index=['Adjusted_Day', 'Adjusted_Hour'], columns='Time_Diff', values='Count', fill_value=0)
pivot_table2 = pivot_table2.reindex(ordered_days, level=0)

# Exportar la cantidad de frecuencias de Time_Diff por hora y día a un archivo Excel
output_file_path_time_diff_by_hour_day = f"{exec_path}/{symbol}_{proveedor}_{tipo_gap}_time_diff_by_hour_day.xlsx"
pivot_table2.to_excel(output_file_path_time_diff_by_hour_day)
print(f"La tabla de frecuencias de Time_Diff por hora y día ha sido exportada a: {output_file_path_time_diff_by_hour_day}")

# Función para mostrar los gaps según el rango de Time_Diff ingresado
def show_gaps_by_timediff_range(time_diff_min, time_diff_max, symbol):
    """
    Filtra y muestra los gaps según el rango de Time_Diff especificado.

    Args:
        time_diff_min (int): Valor mínimo de Time_Diff a filtrar.
        time_diff_max (int): Valor máximo de Time_Diff a filtrar.
        symbol (str): Símbolo del mercado.

    Returns:
        bool: True si se encontraron gaps, False en caso contrario.
    """
    filtered_gaps = df_filtered[(df_filtered['Time_Diff'] >= time_diff_min) & (df_filtered['Time_Diff'] <= time_diff_max)]
    if not filtered_gaps.empty:
        gap_details = filtered_gaps[['Time_Diff']].copy()
        gap_details['Date'] = gap_details.index.date
        gap_details['Time'] = gap_details.index.time
        gap_details['Day_of_Week'] = gap_details.index.day_name()
        
        # Obtener la fecha y hora de inicio y fin del gap
        gap_details['Initial_DateTime'] = gap_details.index - pd.to_timedelta(gap_details['Time_Diff'], unit='m')
        gap_details['Initial_Date'] = gap_details['Initial_DateTime'].dt.date
        gap_details['Initial_Time'] = gap_details['Initial_DateTime'].dt.time
        gap_details['Initial_Day_of_Week'] = gap_details['Initial_DateTime'].dt.day_name()
        
        # Usar tabulate para formatear la salida
        table = tabulate(gap_details[['Initial_Date', 'Initial_Time', 'Initial_Day_of_Week', 'Date', 'Time', 'Day_of_Week', 'Time_Diff']],
                         headers=["Date Initial", "Time Initial", "Day Initial", "Date End", "Time End", "Day End", "Time_Diff"],
                         tablefmt="plain")
        
        with open(output_filename, 'w') as f:
            f.write(table)
                
        print(f"Gaps guardados en {output_filename}")
        return True
    else:
        print(f"No se encontraron gaps con Time_Diff entre {time_diff_min} y {time_diff_max} minutos.")
        return False
"""
# Solicitar al usuario ingresar el rango de Time_Diff
while True:
    try:
        user_time_diff_min = int(input("Ingrese el valor mínimo de Time_Diff: "))
        user_time_diff_max = int(input("Ingrese el valor máximo de Time_Diff: "))
        if user_time_diff_min > user_time_diff_max:
            print("El valor mínimo no puede ser mayor que el valor máximo. Por favor, inténtelo de nuevo.")
            continue
        if show_gaps_by_timediff_range(user_time_diff_min, user_time_diff_max, symbol):
            break
    except ValueError:
        print("Por favor, ingrese valores válidos para Time_Diff.")
"""

execution_time = datetime.now() - start_time
print(f"Tiempo de ejecución del script: {execution_time}")