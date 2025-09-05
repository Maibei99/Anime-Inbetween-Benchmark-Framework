#!/usr/bin/env python3
"""
Utilidad para extraer todos los fotogramas de un archivo de video.

Este script es una herramienta de apoyo para depuración y análisis visual.
Toma un video como entrada y guarda cada uno de sus fotogramas como
imágenes individuales en un directorio de salida especificado.
"""
import argparse
import cv2
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path, image_format: str):
    """
    Extrae todos los fotogramas de un video y los guarda en un directorio.

    Args:
        video_path (Path): Ruta al archivo de video de entrada.
        output_dir (Path): Directorio donde se guardarán los fotogramas.
        image_format (str): Formato de imagen para guardar ('jpg' o 'png').
    """
    # 1. Validar que el video exista
    if not video_path.is_file():
        print(f"❌ Error: No se encontró el video en la ruta: {video_path}")
        return

    # 2. Crear el directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Los fotogramas se guardarán en: {output_dir}")

    # 3. Abrir el video para lectura
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el archivo de video: {video_path}")
        return

    # 4. Procesar el video fotograma a fotograma
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break  # Salir del bucle si no hay más fotogramas

        # Construir el nombre del archivo de salida con padding para ordenamiento
        frame_filename = f"frame_{frame_count:06d}.{image_format}"
        output_path = output_dir / frame_filename

        # Guardar el fotograma
        cv2.imwrite(str(output_path), frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"🖼️  Procesados {frame_count} fotogramas...")

    # 5. Liberar recursos y mostrar resumen
    cap.release()
    print("=" * 40)
    print("✅ ¡Proceso de extracción completado!")
    print(f"   Total de fotogramas extraídos: {frame_count}")
    print("=" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extrae todos los fotogramas de un video y los guarda como imágenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Ruta al archivo de video de entrada."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Ruta a la carpeta de salida para guardar los fotogramas."
    )
    parser.add_argument(
        "--format",
        type=str,
        default='jpg',
        choices=['jpg', 'png'],
        help="Formato de las imágenes guardadas."
    )

    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir, args.format)