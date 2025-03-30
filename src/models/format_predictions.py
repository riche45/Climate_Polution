import json
import os
from datetime import datetime

def convert_predictions_format():
    """
    Convierte las predicciones del formato optimizado al formato esperado por Nuwe
    """
    print("Convirtiendo formato de predicciones...")
    
    # Cargar el archivo optimizado
    with open('predictions/predictions_task_2_optimized.json', 'r') as f:
        optimized_predictions = json.load(f)
    
    # Crear estructura para el nuevo formato
    nuwe_format = {"target": {}}
    
    # Procesar cada estación y contaminante
    for station_id, station_data in optimized_predictions.items():
        nuwe_format["target"][station_id] = {}
        
        for pollutant, predictions in station_data.items():
            for pred in predictions:
                date = pred["date"]
                value = pred["value"]
                nuwe_format["target"][station_id][date] = value
    
    # Guardar el archivo con el formato esperado
    with open('predictions/predictions_task_2_nuwe_format.json', 'w') as f:
        json.dump(nuwe_format, f, indent=2)
    
    print("Conversión completada. Archivo guardado en 'predictions/predictions_task_2_nuwe_format.json'")

if __name__ == "__main__":
    convert_predictions_format() 