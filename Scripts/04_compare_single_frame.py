#!/usr/bin/env python3
"""
Compara un único fotograma generado contra su ground truth.

Este script calcula un conjunto de métricas de calidad de imagen (PSNR, SSIM,
MS-SSIM, LPIPS) y guarda los resultados en un archivo CSV de una sola fila,
diseñado para ser llamado por el script orquestador del benchmark.
"""
import argparse
import csv
import torch
import warnings
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure
)
import lpips

# Suprimir warnings de torchmetrics y otros para una salida más limpia
warnings.filterwarnings("ignore")

def load_image_as_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Carga una imagen desde una ruta, la convierte a un tensor y la mueve al dispositivo.

    Args:
        image_path (Path): Ruta al archivo de imagen.
        device (torch.device): Dispositivo (ej. 'cpu' o 'cuda') al que se moverá el tensor.

    Returns:
        torch.Tensor: Tensor de la imagen con forma (1, 3, H, W) y rango [0, 1].
    """
    try:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        tensor = transform(img).unsqueeze(0)  # Añadir dimensión de batch
        return tensor.to(device)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró la imagen en {image_path}")
        raise
    except Exception as e:
        print(f"❌ Error cargando la imagen {image_path}: {e}")
        raise

def calculate_metrics(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, device: torch.device) -> dict:
    """
    Calcula un conjunto de métricas de calidad de imagen.

    Args:
        pred_tensor (torch.Tensor): Tensor de la imagen predicha.
        gt_tensor (torch.Tensor): Tensor de la imagen ground truth.
        device (torch.device): Dispositivo para realizar los cálculos.

    Returns:
        dict: Un diccionario con los nombres y valores de las métricas.
    """
    metrics = {}
    try:
        # Reuso de objetos de métrica es una optimización clave [cite: 38]
        # Aquí los instanciamos para una única comparación.
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # Se usa la red 'alex' para LPIPS por su buen balance calidad/coste [cite: 40]
        lpips_metric = lpips.LPIPS(net='alex', verbose=False).to(device)

        # La inferencia y métricas se ejecutan sin gradientes para optimizar [cite: 39]
        with torch.no_grad():
            metrics['PSNR'] = psnr_metric(pred_tensor, gt_tensor).item()
            metrics['SSIM'] = ssim_metric(pred_tensor, gt_tensor).item()
            metrics['MS_SSIM'] = ms_ssim_metric(pred_tensor, gt_tensor).item()
            
            # LPIPS requiere que los tensores estén en el rango [-1, 1]
            pred_lpips = pred_tensor * 2.0 - 1.0
            gt_lpips = gt_tensor * 2.0 - 1.0
            metrics['LPIPS'] = lpips_metric(pred_lpips, gt_lpips).item()

    except Exception as e:
        print(f"❌ Error durante el cálculo de métricas: {e}")
        return {key: float('nan') for key in ['PSNR', 'SSIM', 'MS_SSIM', 'LPIPS']}
    
    return metrics

def save_metrics_to_csv(metrics: dict, output_path: Path):
    """
    Guarda las métricas en un archivo CSV con un header y una única fila de datos.

    Args:
        metrics (dict): Diccionario con los resultados de las métricas.
        output_path (Path): Ruta al archivo CSV de salida.
    """
    header = ['PSNR', 'SSIM', 'MS_SSIM', 'LPIPS']
    try:
        with output_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow([f"{metrics.get(h, float('nan')):.6f}" for h in header])
    except Exception as e:
        print(f"❌ Error guardando el archivo CSV en {output_path}: {e}")

def main(args):
    """Función principal del script de comparación."""
    # Se fuerza la CPU para estabilidad y resultados deterministas 
    device = torch.device(args.device)
    print(f"🔬 Comparando en dispositivo: {device}")
    
    try:
        # Cargar imágenes
        pred_tensor = load_image_as_tensor(args.predicted_image, device)
        gt_tensor = load_image_as_tensor(args.ground_truth_image, device)
    except Exception:
        # Los errores ya se imprimen en la función de carga
        return

    # Validar y ajustar dimensiones si es necesario
    if pred_tensor.shape != gt_tensor.shape:
        print(f"⚠️  Las dimensiones no coinciden. Pred: {pred_tensor.shape}, GT: {gt_tensor.shape}.")
        print(f"   Ajustando la predicción al tamaño del ground truth.")
        pred_tensor = torch.nn.functional.interpolate(
            pred_tensor, size=gt_tensor.shape[2:], mode='bilinear', align_corners=False
        )
    
    # Calcular y guardar métricas
    metrics = calculate_metrics(pred_tensor, gt_tensor, device)
    save_metrics_to_csv(metrics, args.output_csv)

    print("✅ Comparación completada. Resultados guardados.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compara un frame generado con un ground truth y calcula métricas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("predicted_image", type=Path, help="Ruta al frame generado por el modelo.")
    parser.add_argument("ground_truth_image", type=Path, help="Ruta al frame real (ground truth).")
    parser.add_argument("output_csv", type=Path, help="Ruta al archivo CSV donde se guardará el resultado.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo para los cálculos ('cpu' o 'cuda'). Se recomienda 'cpu' para determinismo."
    )
    
    args = parser.parse_args()
    main(args)