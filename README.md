# Framework de Benchmark para Interpolación de Fotogramas en Anime

Este repositorio contiene un framework de software robusto y reproducible para la evaluación sistemática de modelos de interpolación de fotogramas, con un enfoque principal en el modelo **ToonCrafter**.

---

## 🎯 Características Principales

- **Arquitectura Anti-Data Leak**: Separación física y estricta de los datos de entrada (frames 0 y 2) y los datos de evaluación (frame 1, _ground truth_) para garantizar una evaluación justa.
- **Reproducibilidad Garantizada**: Uso de **submódulos de Git** para versionar los modelos externos y scripts parametrizados para asegurar que los experimentos se puedan replicar con exactitud.
- **Pipeline Automatizado**: Scripts numerados que guían al usuario a través del proceso completo: preparación del dataset, ejecución del benchmark y análisis de resultados.
- **Métricas Unificadas**: Sistema de evaluación consistente que utiliza métricas estándar de la industria como PSNR, SSIM, MS-SSim y LPIPS.

---

## ⚙️ Cómo Empezar (Getting Started)

Sigue estos pasos para configurar el entorno y ejecutar el benchmark.

### 1\. Requisitos Previos

\*\* Git

- Python 3.9+ (se recomienda el uso de un entorno virtual como `venv` o `conda`)
- FFmpeg (necesario para la extracción de fotogramas)\*

### 2\. Clonar el Repositorio

Clona este repositorio utilizando el flag `--recurse-submodules` para descargar también el código de los modelos externos (`ToonCrafter` y `TPS-InBetween`).

```bash
git clone --recurse-submodules https://github.com/Maibei99/Inbetween.git
cd Inbetween
```

Si ya clonaste el repositorio sin los submódulos, puedes inicializarlos con:

```bash
git submodule update --init --recursive
```

### 3\. Instalar Dependencias

Instala las librerías de Python necesarias.

```bash
pip install -r requirements.txt
```

### 4\. Descargar el Dataset (Acción Manual)

Este benchmark utiliza el dataset **ATD-12K**. Debido a su tamaño, debe ser descargado manualmente.

1.  **Descarga el archivo** desde su fuente oficial en Google Cloud:

    - `https://drive.google.com/file/d/1XBDuiEgdd6c0S4OXLF4QvgSn_XNPwc-g/view`
    - `https://entuedu-my.sharepoint.com/personal/siyao002_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsiyao002_e_ntu_edu_sg%2FDocuments%2Fatd_12k%2Ezip&parent=%2Fpersonal%2Fsiyao002_e_ntu_edu_sg%2FDocuments&ga=1`

2.  **Crea la carpeta `data/raw`** y descomprime el contenido allí.

    ```bash
    mkdir -p data/raw
    # Mueve el .zip descargado a data/raw/ y descomprímelo
    ```

3.  **Verifica la estructura final**. La carpeta `data/raw/` debe contener lo siguiente:

    ```
    data/
    └── raw/
        ├── train_10k/
        ├── test_2k_original/
        ├── test_2k_annotations/
    ```

---

## 🚀 Uso del Framework

El pipeline está dividido en scripts numerados que deben ejecutarse en orden.

### Paso 1: Preparar el Dataset

Este script convierte el dataset crudo a un formato estandarizado (imágenes de 512x320), creando las particiones de entrenamiento/validación y los archivos de metadatos necesarios.

```bash
python scripts/01_prepare_dataset.py \
    --source_dir "data/raw" \
    --output_dir "data/processed" \
    --val_split 0.1
```

### Paso 2: Ejecutar el Benchmark Completo

Este es el script principal que orquesta la inferencia y la evaluación. Tomará los datos de `data/processed/validation`, ejecutará ToonCrafter para cada triplete y calculará las métricas comparando las predicciones con el _ground truth_ aislado.

```bash
bash scripts/02_run_full_benchmark.sh \
    --dataset_dir "data/processed" \
    --output_dir "analysis" \
    --ckpt_path "models/tooncrafter/checkpoints/tooncrafter_512_interp_v1/model.ckpt" \
    --config_path "models/tooncrafter/configs/inference_512_v1.0.yaml" \
    --seed 123
```

Este comando creará una carpeta de resultados con marca de tiempo dentro de `analysis/`, que contendrá el archivo `final_metrics.csv` con todos los resultados numéricos.

---

## 📁 Estructura del Proyecto

```
├── analysis/               # Resultados, métricas y notebooks de análisis
├── data/                   # Datos del benchmark (crudos y procesados)
├── models/                 # Modelos, configuraciones y submódulos externos
│   ├── tooncrafter/        # Checkpoints y configs personalizadas
│   └── external/           # Submódulos de Git (ToonCrafter, TPS-Inbetween)
├── scripts/                # Pipeline de ejecución principal
├── .gitmodules             # Archivo de configuración de submódulos
└── README.md               # Este archivo
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un _issue_ para discutir cambios importantes antes de realizar un _pull request_.

---

## 📜 Licencia

Este proyecto se distribuye bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 📚 Citaciones y Agradecimientos

Este trabajo se basa en los siguientes modelos y datasets de investigación:

- **ToonCrafter**: https://github.com/Doubiiu/ToonCrafter.git
- **TPS-InBetween**: https://github.com/Tian-one/tps-inbetween.git
- **ATD-12K Dataset**: https://github.com/lisiyao21/AnimeInterp.git
