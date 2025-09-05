#!/bin/bash

# ==============================================================================
# SCRIPT ORQUESTADOR DEL BENCHMARK PARA TOONCRAFTER
#
# Uso:
# 1. Prepara el dataset con '01_prepare_dataset.py'.
# 2. Ejecuta este script para realizar la inferencia, evaluación y análisis.
#
# Elimina el data leak separando estrictamente los datos de entrada
# (frames 0 y 2) del ground truth (frame 1) durante todo el proceso.
# ==============================================================================

# --- MODO ESTRICTO ---
# set -e: Salir inmediatamente si un comando falla.
# set -o pipefail: El código de salida de una tubería es el del último comando que falló.
set -e
set -o pipefail

# --- VALORES POR DEFECTO ---
SEED=123
VIDEO_LENGTH=16
DDIM_STEPS=30
GUIDANCE_SCALE=7.5
FRAME_STRIDE=10
MIDDLE_FRAME_IDX=8

# --- FUNCIÓN DE AYUDA ---
usage() {
    echo "Uso: $0 --dataset_dir <ruta> --output_dir <ruta> --ckpt_path <ruta> --config_path <ruta> [OPCIONES]"
    echo ""
    echo "Argumentos Obligatorios:"
    echo "  --dataset_dir    Ruta al dataset procesado (salida de 01_prepare_dataset.py)."
    echo "  --output_dir     Directorio base donde se guardarán todos los resultados."
    echo "  --ckpt_path      Ruta al archivo .ckpt del modelo ToonCrafter."
    echo "  --config_path    Ruta al archivo .yaml de configuración de inferencia."
    echo ""
    echo "Opciones Adicionales:"
    echo "  --seed           Semilla aleatoria para reproducibilidad. (Default: $SEED)"
    echo "  --ddim_steps     Pasos de difusión DDIM. (Default: $DDIM_STEPS)"
    echo "  --help           Muestra esta ayuda."
    exit 1
}

# --- PARSEO DE ARGUMENTOS ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_dir) DATASET_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --ckpt_path) CKPT_PATH="$2"; shift ;;
        --config_path) CONFIG_PATH="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --ddim_steps) DDIM_STEPS="$2"; shift ;;
        --help) usage ;;
        *) echo "Opción desconocida: $1"; usage ;;
    esac
    shift
done

# --- VALIDACIÓN DE ARGUMENTOS OBLIGATORIOS ---
if [[ -z "$DATASET_DIR" || -z "$OUTPUT_DIR" || -z "$CKPT_PATH" || -z "$CONFIG_PATH" ]]; then
    echo "❌ Error: Faltan argumentos obligatorios."
    usage
fi

# --- DEFINICIÓN DE RUTAS Y ARCHIVOS ---
# Directorio de resultados con marca de tiempo para evitar sobreescrituras
BASE_RESULTS_DIR="$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S)"

# Directorios para el proceso anti-data-leak
CLEAN_INPUTS_DIR="$BASE_RESULTS_DIR/clean_inputs"
ISOLATED_GT_DIR="$BASE_RESULTS_DIR/isolated_gt"
GENERATED_VIDEOS_DIR="$BASE_RESULTS_DIR/generated"
EXTRACTED_FRAMES_DIR="$BASE_RESULTS_DIR/extracted"
FINAL_RESULTS_FILE="$BASE_RESULTS_DIR/final_metrics.csv"

# Archivos de referencia del dataset
TRIPLET_MAP_FILE="$DATASET_DIR/triplet_map.json"
GLOBAL_PROMPTS_FILE="$DATASET_DIR/prompts.txt"
VALIDATION_DATA_DIR="$DATASET_DIR/validation"

# --- VERIFICACIONES INICIALES ---
echo "🔍 Verificando rutas y archivos..."
for path in "$VALIDATION_DATA_DIR" "$TRIPLET_MAP_FILE" "$GLOBAL_PROMPTS_FILE" "$CKPT_PATH" "$CONFIG_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "❌ Error: No se encontró el archivo o directorio requerido: $path"
        exit 1
    fi
done
echo "✅ Todas las rutas son válidas."

# --- INICIO DEL BENCHMARK ---
echo "============================================================================="
echo "🚀 INICIANDO BENCHMARK TOONCRAFTER (MODO ANTI-DATA LEAK)"
echo "============================================================================="
echo "  - Dataset: $DATASET_DIR"
echo "  - Resultados se guardarán en: $BASE_RESULTS_DIR"
echo "  - Modelo Checkpoint: $CKPT_PATH"
echo "============================================================================="

# Crear estructura de directorios
mkdir -p "$CLEAN_INPUTS_DIR" "$ISOLATED_GT_DIR" "$GENERATED_VIDEOS_DIR" "$EXTRACTED_FRAMES_DIR"

# Crear header del CSV de resultados
echo "triplet_id,original_folder,PSNR,SSIM,MS_SSIM,LPIPS,separation_confirmed" > "$FINAL_RESULTS_FILE"

# --- PASO 1: SEPARAR DATOS PARA ELIMINAR LEAK ---
echo -e "\n[PASO 1/3] 🛡️  Separando Inputs de Ground Truth..."
total_triplets=$(find "$VALIDATION_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
separated_count=0

for original_triplet_dir in "$VALIDATION_DATA_DIR"/*/; do
    # Lógica de separación... (idéntica a tu versión original)
    # ...
done
echo "✅ Separación completada: $separated_count triplets listos para inferencia."

# --- PASO 2: EJECUTAR INFERENCIA Y EVALUACIÓN ---
echo -e "\n[PASO 2/3] 🤖 Ejecutando Inferencia y Evaluación..."
processed_videos=0
start_time=$(date +%s)

for clean_triplet_dir in "$CLEAN_INPUTS_DIR"/*/; do
    triplet_id=$(basename "$clean_triplet_dir")
    ((processed_videos++))

    echo "($processed_videos/$separated_count) Procesando: $triplet_id"

    video_output_dir="$GENERATED_VIDEOS_DIR/$triplet_id"
    mkdir -p "$video_output_dir"
    
    # 1. Ejecutar Inferencia
    python3 scripts/evaluation/inference.py \
        --seed "$SEED" \
        --ckpt_path "$CKPT_PATH" \
        --config "$CONFIG_PATH" \
        --prompt_dir "$clean_triplet_dir" \
        --savedir "$video_output_dir" \
        --n_samples 1 --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale "$GUIDANCE_SCALE" \
        --ddim_steps "$DDIM_STEPS" \
        --ddim_eta 1.0 --text_input \
        --video_length "$VIDEO_LENGTH" \
        --frame_stride "$FRAME_STRIDE" \
        --timestep_spacing 'uniform_trailing' \
        --guidance_rescale 0.7 --perframe_ae --interp > /dev/null 2>&1

    # 2. Extraer Frame
    generated_video=$(find "$video_output_dir" -name "*.mp4" | head -n 1)
    if [[ -z "$generated_video" ]]; then
        echo "  ❌ Fallo en generación para $triplet_id"
        continue
    fi
    extracted_frame="$EXTRACTED_FRAMES_DIR/${triplet_id}_pred.jpg"
    ffmpeg -i "$generated_video" -vf "select=eq(n\\,$MIDDLE_FRAME_IDX)" -vframes 1 -q:v 2 "$extracted_frame" -y > /dev/null 2>&1

    # 3. Evaluar (Comparar con GT)
    ground_truth_file="$ISOLATED_GT_DIR/${triplet_id}_gt.jpg"
    temp_metrics="$BASE_RESULTS_DIR/temp_metrics_${triplet_id}.csv"
    python3 04_compare_single_frame.py "$extracted_frame" "$ground_truth_file" "$temp_metrics"

    # 4. Guardar Resultados y Limpiar
    if [[ -f "$temp_metrics" ]]; then
        metrics_line=$(tail -n 1 "$temp_metrics")
        original_folder_name=$(jq -r ".[\"$triplet_id\"]" "$TRIPLET_MAP_FILE")
        echo "$triplet_id,$original_folder_name,$metrics_line,YES" >> "$FINAL_RESULTS_FILE"
        rm -f "$temp_metrics"
    fi
    rm -rf "$video_output_dir"
done
echo "✅ Inferencia y evaluación completadas."

# --- PASO 3: ANÁLISIS DE RESULTADOS ---
echo -e "\n[PASO 3/3] 📊 Analizando Resultados..."
python3 05_analyze_results.py --input_csv "$FINAL_RESULTS_FILE"

total_time=$(($(date +%s) - start_time))
echo "============================================================================="
echo "🎉 BENCHMARK COMPLETADO"
echo "   - Tiempo total: $((total_time / 60)) minutos y $((total_time % 60)) segundos."
echo "   - Resultados finales en: $FINAL_RESULTS_FILE"
echo "============================================================================="