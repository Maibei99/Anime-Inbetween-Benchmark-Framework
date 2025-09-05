#!/usr/bin/env python3
"""
Prepara y estandariza el dataset ATD-12K para el benchmark de ToonCrafter.

Este script realiza las siguientes operaciones:
1.  Localiza los tripletes de fotogramas en los directorios de entrenamiento y prueba.
2.  Divide el conjunto total de datos en particiones de entrenamiento y validación.
3.  Normaliza las imágenes a una resolución fija (512x320).
4.  Renombra los fotogramas de manera canónica (00000000.jpg, 00000001.jpg, etc.).
5.  Genera un archivo de prompts (prompts.txt) para la inferencia condicionada.
6.  Crea un mapa de asociación (triplet_map.json) que vincula los nuevos nombres
    de tripletes con sus carpetas originales para trazabilidad.
"""
import os
import shutil
import json
import random
import argparse
from pathlib import Path
from PIL import Image

# --- Constantes de Configuración ---
# Usar constantes mejora la legibilidad y facilita los cambios.
IMG_WIDTH = 512
IMG_HEIGHT = 320
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
IMG_QUALITY = 95
FRAME_MAPPING = {
    0: "00000000.jpg",  # Primer frame (input)
    1: "00000001.jpg",  # Segundo frame (ground truth)
    2: "00000002.jpg"   # Último frame (input)
}

def process_triplets(triplet_paths, output_dir, source_config, counters, triplet_map, prompts):
    """
    Procesa una lista de rutas de tripletes, las normaliza y guarda en el directorio de salida.

    Args:
        triplet_paths (list): Lista de rutas a las carpetas de tripletes originales.
        output_dir (str): Directorio donde se guardarán los tripletes procesados.
        source_config (dict): Diccionario con rutas a archivos de anotaciones y series.
        counters (dict): Diccionario para contar tripletes procesados y fallidos.
        triplet_map (dict): Diccionario para almacenar la asociación de nombres.
        prompts (list): Lista para almacenar los prompts generados.
    """
    for triplet_path in triplet_paths:
        original_folder_name = os.path.basename(triplet_path)
        try:
            # Encuentra todos los archivos de imagen en la carpeta del triplete
            frame_files = sorted([
                f for f in Path(triplet_path).iterdir()
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])

            if len(frame_files) < 2:
                print(f"⚠️  Triplete incompleto (menos de 2 frames): {original_folder_name}")
                counters['failed'] += 1
                continue

            # Asigna un nuevo nombre canónico al triplete
            new_folder_name = f"triplet_{counters['processed']:06d}"
            triplet_output_path = Path(output_dir) / new_folder_name
            triplet_output_path.mkdir(exist_ok=True)

            # Guarda la asociación para trazabilidad antes de procesar
            triplet_map[new_folder_name] = original_folder_name

            # Procesa el primer, segundo (ground truth) y último frame
            frames_to_process = [frame_files[0], frame_files[1], frame_files[-1]]

            for i, source_frame_path in enumerate(frames_to_process):
                try:
                    img = Image.open(source_frame_path).convert('RGB')
                    img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                    output_frame_name = FRAME_MAPPING[i]
                    img_resized.save(triplet_output_path / output_frame_name, 'JPEG', quality=IMG_QUALITY)
                except Exception as e:
                    print(f"❌ Error procesando el frame {source_frame_path}: {e}")
                    raise  # Propaga el error para que el bloque except exterior lo maneje

            # Si el bucle de frames se completa, genera el prompt
            # (El código de generación de prompts se mantiene igual)
            serie_name = original_folder_name
            for key, value in source_config['series_mapping'].items():
                if key in original_folder_name:
                    serie_name = value
                    break
            
            behavior = "animation"
            annotation_path = source_config['test_annotation'] / original_folder_name / f"{original_folder_name}.json"
            if annotation_path.exists():
                try:
                    with annotation_path.open('r') as f:
                        behavior = json.load(f).get('behavior', 'animation')
                except (json.JSONDecodeError, KeyError):
                    pass
            
            style = "disney style" if "Disney" in original_folder_name else "anime style"
            prompt = f"{style} interpolation, behavior {behavior}, the serie is {serie_name}"
            prompts.append(f"{new_folder_name}: {prompt}")
            
            counters['processed'] += 1
            if counters['processed'] % 100 == 0:
                print(f"✓ Procesados {counters['processed']} tripletes...")

        except Exception as e:
            print(f"❌ Error grave procesando el triplete {original_folder_name}: {e}")
            counters['failed'] += 1
            # Limpieza: si el triplete falló, elimina su carpeta de salida y su entrada del mapa
            if 'triplet_output_path' in locals() and triplet_output_path.exists():
                shutil.rmtree(triplet_output_path)
            if 'new_folder_name' in locals() and new_folder_name in triplet_map:
                del triplet_map[new_folder_name]


def main(args):
    """
    Función principal que orquesta la preparación del dataset.
    """
    # --- 1. Validar rutas de entrada y crear directorios de salida ---
    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)

    train_source = source_path / 'train_10k'
    test_source = source_path / 'test_2k_original'
    test_annotation = source_path / 'test_2k_annotations'
    series_json_path = source_path / "series.json"

    for path in [train_source, test_source, test_annotation, series_json_path]:
        if not path.exists():
            print(f"❌ Error: La ruta de entrada no fue encontrada: {path}")
            return

    output_train_path = output_path / 'train'
    output_val_path = output_path / 'validation'
    output_train_path.mkdir(parents=True, exist_ok=True)
    output_val_path.mkdir(parents=True, exist_ok=True)

    # --- 2. Recolectar y mezclar todos los tripletes ---
    all_triplets = [p for p in train_source.iterdir() if p.is_dir()]
    all_triplets.extend([p for p in test_source.iterdir() if p.is_dir()])
    
    print(f"✓ Total de tripletes encontrados: {len(all_triplets)}")
    random.shuffle(all_triplets)

    # --- 3. Dividir en conjuntos de entrenamiento y validación ---
    val_count = int(len(all_triplets) * args.val_split)
    val_triplets = all_triplets[:val_count]
    train_triplets = all_triplets[val_count:]

    print(f"✓ Usando {args.val_split*100:.1f}% para validación.")
    print(f"✓ Tripletes para entrenamiento: {len(train_triplets)}")
    print(f"✓ Tripletes para validación: {len(val_triplets)}")

    # --- 4. Procesar ambos conjuntos de datos ---
    counters = {'processed': 0, 'failed': 0}
    triplet_map = {}
    prompts = []
    
    source_config = {
        'test_annotation': test_annotation,
        'series_mapping': json.loads(series_json_path.read_text())
    }

    print("\n🔄 Procesando tripletes de entrenamiento...")
    process_triplets(train_triplets, output_train_path, source_config, counters, triplet_map, prompts)
    
    print("\n🔄 Procesando tripletes de validación...")
    process_triplets(val_triplets, output_val_path, source_config, counters, triplet_map, prompts)
    
    # --- 5. Guardar archivos de metadatos ---
    # Guardar prompts
    (output_path / 'prompts.txt').write_text('\n'.join(prompts), encoding='utf-8')
    
    # Guardar mapa de asociación
    with (output_path / 'triplet_map.json').open('w', encoding='utf-8') as f:
        json.dump(triplet_map, f, indent=2)

    # Guardar metadatos generales de la ejecución
    metadata = {
        "dataset_name": f"{source_path.name}_prepared_for_tooncrafter",
        "total_triplets_processed": counters['processed'],
        "total_triplets_failed": counters['failed'],
        "train_triplet_count": len(train_triplets),
        "validation_triplet_count": len(val_triplets),
        "validation_split_ratio": args.val_split,
        "image_size": [IMG_WIDTH, IMG_HEIGHT],
    }
    with (output_path / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*50)
    print("✅ PROCESO COMPLETADO")
    print(f"   📁 Dataset procesado guardado en: {output_path}")
    print(f"   ✓ Tripletes exitosos: {counters['processed']}")
    print(f"   ❌ Tripletes fallidos: {counters['failed']}")
    print(f"   🗺️  Mapa de asociación y prompts guardados.")
    print("="*50)


if __name__ == '__main__':
    # --- Configuración de Argumentos de Línea de Comandos ---
    parser = argparse.ArgumentParser(
        description="Prepara el dataset ATD-12K para el benchmark de ToonCrafter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--source_dir', 
        type=str, 
        required=True,
        help="Ruta al directorio raíz del dataset ATD-12K (debe contener 'train_10k', 'test_2k_original', etc.)."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help="Ruta al directorio donde se guardará el dataset procesado."
    )
    parser.add_argument(
        '--val_split', 
        type=float, 
        default=0.1,
        help="Proporción del dataset a utilizar para la validación (ej. 0.1 para 10%)."
    )
    
    # Parsea los argumentos y ejecuta la función principal
    args = parser.parse_args()
    main(args)