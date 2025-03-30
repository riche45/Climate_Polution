import json
import shutil
from pathlib import Path
import os

def use_fixed_task2_predictions():
    """
    Copia el archivo de predicciones fixed (que obtuvo el 37% de puntos)
    al archivo principal de predicciones que se utiliza para la evaluación.
    """
    # Definir rutas de archivos
    fixed_predictions_file = 'predictions/predictions_task_2_fixed.json'
    current_predictions_file = 'predictions/predictions_task_2.json'
    backup_file = 'predictions/predictions_task_2_backup.json'
    
    # Verificar que los archivos existen
    if not Path(fixed_predictions_file).exists():
        print(f"Error: No se encontró el archivo {fixed_predictions_file}")
        return
    
    if Path(current_predictions_file).exists():
        # Crear copia de seguridad del archivo actual
        shutil.copy2(current_predictions_file, backup_file)
        print(f"Se ha creado una copia de seguridad en {backup_file}")
    
    # Cargar ambos archivos para comparar
    try:
        with open(fixed_predictions_file, 'r') as f:
            fixed_predictions = json.load(f)
        
        if Path(current_predictions_file).exists():
            with open(current_predictions_file, 'r') as f:
                current_predictions = json.load(f)
            
            # Comparar si son idénticos
            if fixed_predictions == current_predictions:
                print("Los archivos de predicciones son idénticos.")
            else:
                print("Los archivos de predicciones son diferentes.")
                
                # Mostrar estadísticas
                if "target" in fixed_predictions and "target" in current_predictions:
                    print("\nEstaciones en predictions_task_2_fixed.json:")
                    for station, values in fixed_predictions["target"].items():
                        print(f"Estación {station}: {len(values)} predicciones")
                    
                    print("\nEstaciones en predictions_task_2.json:")
                    for station, values in current_predictions["target"].items():
                        print(f"Estación {station}: {len(values)} predicciones")
    except Exception as e:
        print(f"Error al comparar archivos: {e}")
    
    # Crear una copia limpia del archivo fixed para la evaluación
    try:
        shutil.copy2(fixed_predictions_file, current_predictions_file)
        print(f"\nSe ha copiado {fixed_predictions_file} a {current_predictions_file}")
    except Exception as e:
        print(f"Error al copiar archivo: {e}")
    
    print("\nProceso completado. Ahora puede hacer commit y push para la evaluación.")

if __name__ == "__main__":
    use_fixed_task2_predictions() 