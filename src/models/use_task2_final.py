import json
import shutil
from pathlib import Path

def use_final_predictions():
    """
    Copia el archivo de predicciones final a la ubicación correcta para la evaluación de Nuwe.
    """
    # Definir rutas de archivos
    final_predictions_file = 'predictions/predictions_task_2_final.json'
    target_file = 'predictions/predictions_task_2.json'
    
    # Verificar que el archivo final existe
    if not Path(final_predictions_file).exists():
        print(f"Error: No se encontró el archivo {final_predictions_file}")
        return
    
    # Cargar el archivo final
    with open(final_predictions_file, 'r') as f:
        final_predictions = json.load(f)
    
    # Verificar la estructura del archivo
    if "target" not in final_predictions or not final_predictions["target"]:
        print("Error: El archivo no tiene la estructura correcta con la clave 'target'")
        return
    
    # Guardar en la ubicación de destino
    with open(target_file, 'w') as f:
        json.dump(final_predictions, f, indent=2)
    
    print(f"Se ha copiado el contenido de {final_predictions_file} a {target_file}")
    
    # Verificar estaciones y cantidad de predicciones
    station_counts = {
        "206": 744,  # Julio (31 días * 24 horas)
        "211": 744,  # Agosto (31 días * 24 horas)
        "217": 720,  # Septiembre (30 días * 24 horas)
        "219": 744,  # Octubre (31 días * 24 horas)
        "225": 720,  # Noviembre (30 días * 24 horas)
        "228": 744   # Diciembre (31 días * 24 horas)
    }
    
    for station, expected_count in station_counts.items():
        if station in final_predictions["target"]:
            actual_count = len(final_predictions["target"][station])
            print(f"Estación {station}: {actual_count} predicciones (esperadas {expected_count})")
            
            if actual_count != expected_count:
                print(f"  ¡ADVERTENCIA! La estación {station} no tiene el número correcto de predicciones.")
        else:
            print(f"¡ADVERTENCIA! Falta la estación {station} en el archivo de predicciones.")

if __name__ == "__main__":
    use_final_predictions() 