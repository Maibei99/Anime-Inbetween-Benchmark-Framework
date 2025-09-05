#!/usr/bin/env python3
"""
Analiza el archivo CSV de métricas generado por el benchmark.

Calcula estadísticas descriptivas (media, desviación estándar, min, max) para
cada métrica y realiza una validación de sanidad sobre los resultados.
"""
import argparse
import pandas as pd

def analyze_metrics(csv_path: str):
    """
    Carga un archivo CSV de métricas, calcula estadísticas y las muestra.

    Args:
        csv_path (str): Ruta al archivo CSV con los resultados.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo de resultados en: {csv_path}")
        return

    if df.empty:
        print("⚠️ Advertencia: El archivo de resultados está vacío. No hay nada que analizar.")
        return

    print("=" * 80)
    print("RESUMEN DE MÉTRICAS DEL BENCHMARK (ANTI-DATA LEAK)")
    print("=" * 80)
    print(f"Total de Triplets Válidos: {len(df)}")
    print("-" * 80)

    metrics_to_analyze = ['PSNR', 'SSIM', 'MS_SSIM', 'LPIPS']
    
    print(f"{'Métrica':<10} | {'Media':<12} | {'Std Dev':<12} | {'Min':<12} | {'Max':<12}")
    print("-" * 80)

    for metric in metrics_to_analyze:
        if metric in df.columns:
            values = df[metric].dropna()
            if not values.empty:
                mean = values.mean()
                std = values.std()
                min_val = values.min()
                max_val = values.max()
                print(f"{metric:<10} | {mean:<12.4f} | {std:<12.4f} | {min_val:<12.4f} | {max_val:<12.4f}")
    
    print("-" * 80)
    print("\nVALIDACIÓN DE SANIDAD (Rangos esperados para interpolación real):")
    
    # Sanity check para PSNR
    if 'PSNR' in df.columns and not df['PSNR'].empty:
        psnr_mean = df['PSNR'].mean()
        if psnr_mean > 35:
            print(f"  - ⚠️ PSNR promedio ({psnr_mean:.2f} dB) es muy alto. Podría indicar un problema.")
        else:
            print(f"  - ✅ PSNR promedio ({psnr_mean:.2f} dB) está en un rango esperado (20-35 dB).")

    # Sanity check para LPIPS (menor es mejor)
    if 'LPIPS' in df.columns and not df['LPIPS'].empty:
        lpips_mean = df['LPIPS'].mean()
        if lpips_mean < 0.05:
            print(f"  - ⚠️ LPIPS promedio ({lpips_mean:.4f}) es muy bajo. Podría indicar imágenes muy similares.")
        else:
            print(f"  - ✅ LPIPS promedio ({lpips_mean:.4f}) es realista.")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analiza los resultados de un archivo CSV de métricas de benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Ruta al archivo CSV final generado por el script de benchmark."
    )
    args = parser.parse_args()
    analyze_metrics(args.input_csv)