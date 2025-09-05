import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.io as tv_io
import numpy as np
from omegaconf import OmegaConf
import random
import gc
import time
import psutil
import subprocess
import tempfile
import json

# ==================== CONFIGURACIÓN INICIAL ====================
# Agregar la ruta del proyecto al sys.path para importaciones locales
project_root_path = os.getcwd() 
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

# Importaciones del proyecto ToonCrafter
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler
from scripts.evaluation.inference import load_model_checkpoint, get_latent_z_with_hidden_states as get_vae_latents_and_hs_from_pixels
from lvdm.models.autoencoder import AutoencoderKL_Dualref

# Importaciones de métricas
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID

# ==================== CONFIGURACIÓN GLOBAL ====================
# Dimensiones objetivo para los frames
TARGET_HEIGHT = 320
TARGET_WIDTH = 512

# Configuración del dispositivo y precisión
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == 'cuda'  # Usar half precision solo en GPU

# Rutas para VMAF (ajustar según tu instalación)
VMAF_EXECUTABLE_PATH = "/mnt/c/Users/BM10/Desktop/Mai/vmaf/python/vmaf/script/run_vmaf.py"
VMAF_MODEL_PATH = "/mnt/c/Users/BM10/Desktop/Mai/vmaf/models/vmaf_v0.6.1.json"

# Inicialización de modelos de métricas perceptuales
lpips_model = lpips.LPIPS(net='alex', verbose=False).eval().to(DEVICE)
ssim_metric = SSIM(data_range=1.0).to(DEVICE)

# ==================== FUNCIONES DE CARGA DEL MODELO ====================
def load_tooncrafter_model_and_sampler(model_config_path, checkpoint_path, device, use_fp16=USE_FP16):
    """
    Carga el modelo ToonCrafter y su sampler DDIM.
    
    Args:
        model_config_path: Ruta al archivo de configuración YAML
        checkpoint_path: Ruta al checkpoint del modelo
        device: Dispositivo de computación (cuda/cpu)
        use_fp16: Si usar precisión FP16
    
    Returns:
        model: Modelo ToonCrafter cargado
        sampler: Sampler DDIM inicializado
    """
    # Cargar configuración
    config = OmegaConf.load(model_config_path)
    model_conf = config.model
    
    # Desactivar checkpointing para inferencia (ahorra memoria)
    if 'unet_config' in model_conf['params'] and 'params' in model_conf['params']['unet_config']:
        if 'use_checkpoint' in model_conf['params']['unet_config']['params']:
            model_conf['params']['unet_config']['params']['use_checkpoint'] = False

    # Instanciar y cargar el modelo
    model = instantiate_from_config(model_conf)
    model = load_model_checkpoint(model, checkpoint_path)
    
    # Convertir a FP16 si está en GPU
    if use_fp16 and device.type == 'cuda':
        model = model.half()
    
    # Mover a dispositivo y modo evaluación
    model = model.to(device)
    model.eval()
    
    # Crear sampler DDIM
    sampler = DDIMSampler(model)
    
    return model, sampler

# ==================== FUNCIONES DE PREPROCESAMIENTO ====================
def preprocess_frame_for_tooncrafter(frame_path, target_height, target_width, device, dtype=None):
    """
    Preprocesa un frame para el modelo ToonCrafter.
    
    Args:
        frame_path: Ruta a la imagen
        target_height: Altura objetivo
        target_width: Ancho objetivo
        device: Dispositivo de computación
        dtype: Tipo de datos del tensor (opcional)
    
    Returns:
        Tensor preprocesado de forma [1, 3, H, W] o None si falla
    """
    if dtype is None:
        dtype = torch.float16 if USE_FP16 and device.type == 'cuda' else torch.float32
    
    try:
        img = Image.open(frame_path).convert("RGB")
    except:
        return None

    # Pipeline de transformaciones
    min_dim = min(target_height, target_width)
    transform_pipeline = transforms.Compose([
        transforms.Resize(min_dim),  # Redimensionar manteniendo aspecto
        transforms.CenterCrop((target_height, target_width)),  # Recortar al centro
        transforms.ToTensor(),  # Convertir a tensor [0, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalizar a [-1, 1]
    ])
    
    img_tensor = transform_pipeline(img)
    return img_tensor.unsqueeze(0).to(device, dtype=dtype)

# ==================== FUNCIONES DE CÁLCULO DE MÉTRICAS ====================
@torch.no_grad()
def calculate_image_metrics(predicted, ground_truth, device=DEVICE):
    """
    Calcula métricas de calidad de imagen entre predicción y ground truth.
    
    Args:
        predicted: Tensor predicho
        ground_truth: Tensor ground truth
        device: Dispositivo para cálculos
    
    Returns:
        Diccionario con métricas: MSE, MAE, PSNR, SSIM, LPIPS
    """
    # Convertir a float32 y mover al dispositivo
    pred = predicted.detach().float().to(device)
    gt = ground_truth.detach().float().to(device)
    
    # Remover dimensión temporal si existe
    if pred.dim() == 5 and pred.size(2) == 1:
        pred = pred.squeeze(2)
    
    # Convertir de [-1, 1] a [0, 1] para métricas
    pred_01 = (pred.clamp(-1, 1) + 1.0) / 2.0
    gt_01 = (gt.clamp(-1, 1) + 1.0) / 2.0
    
    metrics = {}
    
    # MSE y MAE
    mse = torch.mean((pred_01 - gt_01) ** 2).item()
    metrics["MSE"] = mse
    metrics["MAE"] = torch.mean(torch.abs(pred_01 - gt_01)).item()
    
    # PSNR
    if mse == 0:
        metrics["PSNR"] = float('inf')
    else:
        metrics["PSNR"] = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM
    metrics["SSIM"] = ssim_metric(pred_01, gt_01).item()
    
    # LPIPS (requiere rango [-1, 1])
    lpips_val = lpips_model(pred.clamp(-1,1), gt.clamp(-1,1))
    metrics["LPIPS"] = lpips_val.mean().item()
    
    return metrics

@torch.no_grad()
def calculate_vmaf(pred_video, gt_video, vmaf_model_path, vmaf_executable, fps=8):
    """
    Calcula el score VMAF comparando dos videos.
    
    Args:
        pred_video: Video predicho [T, C, H, W] en uint8
        gt_video: Video ground truth [T, C, H, W] en uint8
        vmaf_model_path: Ruta al modelo VMAF
        vmaf_executable: Ruta al ejecutable VMAF
        fps: Frames por segundo para los videos
    
    Returns:
        Score VMAF o NaN si falla
    """
    if not os.path.exists(vmaf_model_path):
        return float('nan')
    
    # Crear directorio temporal para videos
    temp_dir = tempfile.mkdtemp()
    pred_path = os.path.join(temp_dir, "pred.mp4")
    gt_path = os.path.join(temp_dir, "gt.mp4")
    output_json = os.path.join(temp_dir, "vmaf.json")
    
    try:
        # Escribir videos temporales (convertir de TCHW a THWC para torchvision)
        tv_io.write_video(pred_path, pred_video.permute(0,2,3,1), fps=fps, 
                         video_codec='libx264', options={'-crf': '18'})
        tv_io.write_video(gt_path, gt_video.permute(0,2,3,1), fps=fps, 
                         video_codec='libx264', options={'-crf': '18'})
        
        # Ejecutar VMAF
        cmd = [
            vmaf_executable,
            gt_path,      # Referencia
            pred_path,    # Distorsionado
            "--model", f"path={vmaf_model_path}",
            "--json", output_json,
            "--feature", "psnr,float_ssim"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            return float('nan')
        
        # Parsear resultado JSON
        with open(output_json, 'r') as f:
            data = json.load(f)
        
        # Extraer score VMAF (diferentes formatos posibles)
        vmaf_score = data.get("pooled_metrics", {}).get("vmaf", {}).get("mean", float('nan'))
        if np.isnan(vmaf_score):
            vmaf_score = data.get("aggregate", {}).get("VMAF_score", float('nan'))
        
        return vmaf_score
    
    except:
        return float('nan')
    finally:
        # Limpiar archivos temporales
        for path in [pred_path, gt_path, output_json]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

# ==================== FUNCIONES DE INFERENCIA ====================
@torch.no_grad()
def prepare_inference_conditions(model, frame0_pixel, frame2_pixel, prompt_text, 
                                 video_length_target, fs_value, uncond_scale, device):
    """
    Prepara las condiciones para la inferencia del modelo.
    
    Args:
        model: Modelo ToonCrafter
        frame0_pixel: Primer frame del triplete
        frame2_pixel: Último frame del triplete
        prompt_text: Texto de condicionamiento
        video_length_target: Longitud del video a generar (3 para interpolación)
        fs_value: Valor de frame skip
        uncond_scale: Escala para guidance incondicional
        device: Dispositivo de computación
    
    Returns:
        Tupla con condiciones, shape del ruido, tensor fs y kwargs de decodificación
    """
    batch_size = frame0_pixel.size(0)
    model_dtype = next(model.parameters()).dtype
    
    # Asegurar que los frames estén en el dispositivo y dtype correctos
    frame0_pixel = frame0_pixel.to(device, dtype=model_dtype)
    frame2_pixel = frame2_pixel.to(device, dtype=model_dtype)
    
    # Codificar texto
    text_emb = model.get_learned_conditioning([prompt_text] * batch_size)
    
    # Codificar imagen para condicionamiento cruzado
    img_emb = model.embedder(frame0_pixel)
    img_emb_proj = model.image_proj_model(img_emb)
    
    # Concatenar embeddings de texto e imagen
    cond_crossattn = [torch.cat([text_emb, img_emb_proj], dim=1)]
    
    # Preparar frames para codificación VAE
    videos_for_vae = torch.stack([frame0_pixel.squeeze(0), frame2_pixel.squeeze(0)], dim=1).unsqueeze(0)
    
    # Obtener latentes y hidden states del VAE
    z_cond, hs_decode = get_vae_latents_and_hs_from_pixels(model, videos_for_vae)
    
    # Extraer latentes individuales
    latent_f0 = z_cond[:, :, 0, :, :]
    latent_f2 = z_cond[:, :, 1, :, :]
    
    # Configurar shape del ruido
    latent_c = model.model.diffusion_model.out_channels
    latent_h = TARGET_HEIGHT // 8  # Factor de reducción del VAE
    latent_w = TARGET_WIDTH // 8
    noise_shape = [batch_size, latent_c, video_length_target, latent_h, latent_w]
    
    # Crear condición de concatenación con frames de inicio y fin
    img_cat_cond = torch.zeros(noise_shape, device=device, dtype=model_dtype)
    img_cat_cond[:, :, 0, :, :] = latent_f0
    img_cat_cond[:, :, -1, :, :] = latent_f2
    
    # Construir diccionario de condiciones
    cond = {"c_crossattn": cond_crossattn, "c_concat": [img_cat_cond]}
    
    # Preparar condición incondicional si se usa guidance
    uc = None
    if uncond_scale != 1.0:
        uc_text = model.get_learned_conditioning([""] * batch_size)
        uc_img = model.embedder(torch.zeros_like(frame0_pixel))
        uc_img_proj = model.image_proj_model(uc_img)
        uc_crossattn = [torch.cat([uc_text, uc_img_proj], dim=1)]
        uc = {"c_crossattn": uc_crossattn, "c_concat": [img_cat_cond]}
    
    # Preparar tensor de frame skip
    fs_tensor = torch.tensor([fs_value] * batch_size, dtype=torch.long, device=device)
    
    # Preparar kwargs para decodificación (necesario para VAE DualRef)
    decode_kwargs = {}
    if isinstance(model.first_stage_model, AutoencoderKL_Dualref):
        decode_kwargs['ref_context'] = hs_decode
    
    return cond, uc, noise_shape, fs_tensor, decode_kwargs

@torch.no_grad()
def decode_latents_per_frame(model, sampled_latents, decode_kwargs):
    """
    Decodifica latentes frame por frame para optimizar memoria.
    
    Args:
        model: Modelo ToonCrafter
        sampled_latents: Latentes generados [B, C, T, H, W]
        decode_kwargs: Argumentos adicionales para el decodificador
    
    Returns:
        Secuencia de frames decodificados
    """
    b, c, t, h, w = sampled_latents.shape
    model_dtype = next(model.parameters()).dtype
    decoded_frames = []
    
    # Decodificar cada frame individualmente
    for i in range(t):
        frame_latent = sampled_latents[:, :, i:i+1, :, :].to(model.device, dtype=model_dtype)
        
        try:
            # Intentar con autocast FP16
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                decoded = model.decode_first_stage(frame_latent, **decode_kwargs)
            decoded_frames.append(decoded.float().cpu())
        except torch.cuda.OutOfMemoryError:
            # Si falla por memoria, limpiar caché y reintentar
            if model.device.type == 'cuda':
                torch.cuda.empty_cache()
            decoded = model.decode_first_stage(frame_latent.float(), **decode_kwargs)
            decoded_frames.append(decoded.cpu())
        
        # Limpiar memoria entre frames
        if i < t - 1 and model.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenar frames decodificados
    return torch.cat(decoded_frames, dim=2).to(model.device)

@torch.no_grad()
def perform_tooncrafter_inference(model, sampler, cond, uc, noise_shape, fs_tensor,
                                  ddim_steps, ddim_eta, guidance_scale, guidance_rescale, 
                                  seed, decode_kwargs):
    """
    Realiza la inferencia completa del modelo ToonCrafter.
    
    Args:
        model: Modelo ToonCrafter
        sampler: Sampler DDIM
        cond: Condiciones para la generación
        uc: Condiciones incondicionales
        noise_shape: Shape del tensor de ruido
        fs_tensor: Tensor de frame skip
        ddim_steps: Número de pasos DDIM
        ddim_eta: Parámetro eta para DDIM
        guidance_scale: Escala de classifier-free guidance
        guidance_rescale: Rescalado de guidance
        seed: Semilla para reproducibilidad
        decode_kwargs: Argumentos para el decodificador
    
    Returns:
        Secuencia de video generada
    """
    # Fijar semilla para reproducibilidad
    torch.manual_seed(seed)
    
    # Generar latentes con DDIM
    with torch.cuda.amp.autocast(enabled=USE_FP16):
        sampled_latents, _ = sampler.sample(
            S=ddim_steps,
            batch_size=noise_shape[0],
            shape=noise_shape[1:],
            conditioning=cond,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            fs=fs_tensor,
            timestep_spacing='uniform_trailing',
            guidance_rescale=guidance_rescale,
            verbose=False,
            schedule_verbose=False
        )
    
    # Limpiar caché antes de decodificar
    if model.device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Decodificar latentes a píxeles
    generated = decode_latents_per_frame(model, sampled_latents, decode_kwargs)
    
    return generated.float()

# ==================== FUNCIÓN PRINCIPAL DE EVALUACIÓN ====================
def run_atd12k_evaluation_loop(triplets, model, sampler, params):
    """
    Ejecuta el loop de evaluación sobre los tripletes de ATD-12k.
    
    Args:
        triplets: Lista de tuplas (f0_path, f1_gt_path, f2_path)
        model: Modelo ToonCrafter
        sampler: Sampler DDIM
        params: Diccionario con parámetros de inferencia
    
    Returns:
        results: Lista de resultados por triplete
        summary: Diccionario con resumen de métricas
    """
    results = []
    
    # Listas para acumular frames para FID
    collected_pred_f1_for_fid = []
    collected_gt_f1_for_fid = []
    
    # Variables para monitoreo de rendimiento
    total_time = 0
    process = psutil.Process(os.getpid())
    initial_ram = process.memory_info().rss
    peak_ram = initial_ram
    cpu_measurements = []
    
    # Procesar cada triplete
    for idx, (f0_path, f1_gt_path, f2_path) in enumerate(triplets):
        print(f"\nProcessing triplet {idx+1}/{len(triplets)}")
        
        # Reset estadísticas de memoria GPU
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        cpu_start = process.cpu_percent(interval=None)
        
        try:
            # Cargar y preprocesar frames
            model_dtype = next(model.parameters()).dtype
            f0 = preprocess_frame_for_tooncrafter(f0_path, TARGET_HEIGHT, TARGET_WIDTH, 
                                                 model.device, dtype=model_dtype)
            f1_gt = preprocess_frame_for_tooncrafter(f1_gt_path, TARGET_HEIGHT, TARGET_WIDTH, 
                                                    model.device, dtype=torch.float32)
            f2 = preprocess_frame_for_tooncrafter(f2_path, TARGET_HEIGHT, TARGET_WIDTH, 
                                                 model.device, dtype=model_dtype)
            
            # Verificar que todos los frames se cargaron correctamente
            if any(x is None for x in [f0, f1_gt, f2]):
                continue
            
            # Preparar condiciones para inferencia
            cond, uc, noise_shape, fs_tensor, decode_kwargs = prepare_inference_conditions(
                model, f0, f2, "an anime scene", 3,
                params.get("fs", 10), params.get("guidance_scale", 7.5), model.device
            )
            
            # Realizar inferencia
            generated = perform_tooncrafter_inference(
                model, sampler, cond, uc, noise_shape, fs_tensor,
                params.get("ddim_steps", 50), params.get("eta", 1.0),
                params.get("guidance_scale", 7.5), params.get("guidance_rescale", 0.7),
                params.get("seed", 42) + idx, decode_kwargs
            )
            
            # Extraer frame intermedio generado
            f1_pred = generated[:, :, 1, :, :]
            
            # Calcular métricas de imagen
            metrics = calculate_image_metrics(f1_pred, f1_gt)
            
            # Preparar frames para VMAF y FID
            f0_uint8 = ((f0.cpu().float().clamp(-1,1)+1)/2 * 255).type(torch.uint8).squeeze(0)
            f1_pred_uint8 = ((f1_pred.cpu().float().clamp(-1,1)+1)/2 * 255).type(torch.uint8).squeeze(0)
            f1_gt_uint8 = ((f1_gt.cpu().float().clamp(-1,1)+1)/2 * 255).type(torch.uint8).squeeze(0)
            f2_uint8 = ((f2.cpu().float().clamp(-1,1)+1)/2 * 255).type(torch.uint8).squeeze(0)
            
            # Crear tripletes para VMAF
            pred_triplet = torch.stack([f0_uint8, f1_pred_uint8, f2_uint8], dim=0)
            gt_triplet = torch.stack([f0_uint8, f1_gt_uint8, f2_uint8], dim=0)
            
            # Calcular VMAF
            vmaf_score = calculate_vmaf(pred_triplet, gt_triplet, VMAF_MODEL_PATH, VMAF_EXECUTABLE_PATH)
            metrics["VMAF"] = vmaf_score
            
            # Acumular frames para FID
            collected_pred_f1_for_fid.append(f1_pred_uint8)
            collected_gt_f1_for_fid.append(f1_gt_uint8)
            
            # Guardar resultados
            results.append({
                "triplet_idx": idx,
                "metrics": metrics,
                "paths": (f0_path, f1_gt_path, f2_path)
            })
            
        except Exception as e:
            print(f"Error processing triplet: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Limpiar memoria
            if model.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # Registrar métricas de rendimiento
        elapsed = time.time() - start_time
        total_time += elapsed
        
        cpu_end = process.cpu_percent(interval=None)
        if cpu_end > 0:
            cpu_measurements.append(cpu_end)
        
        current_ram = process.memory_info().rss
        peak_ram = max(peak_ram, current_ram)
    
    # Calcular resumen de métricas
    summary = {}
    
    if results:
        summary["num_triplets"] = len(results)
        summary["avg_time_per_triplet"] = total_time / len(results)
        
        # Métricas de memoria
        if model.device.type == 'cuda':
            summary["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        
        summary["initial_ram_mb"] = initial_ram / (1024**2)
        summary["peak_ram_mb"] = peak_ram / (1024**2)
        summary["avg_cpu_percent"] = np.mean(cpu_measurements) if cpu_measurements else 0
        
        # Promediar métricas por frame
        for metric in ["PSNR", "SSIM", "LPIPS", "MSE", "MAE", "VMAF"]:
            values = [r["metrics"][metric] for r in results 
                     if not np.isnan(r["metrics"][metric]) and r["metrics"][metric] != float('inf')]
            if values:
                summary[f"avg_{metric}"] = np.mean(values)
        
        # Calcular FID si hay suficientes muestras
        if collected_pred_f1_for_fid and collected_gt_f1_for_fid:
            print("\nCalculating FID...")
            fid_metric = FID(feature=2048).to(DEVICE)
            gt_batch = torch.stack(collected_gt_f1_for_fid).to(DEVICE)
            pred_batch = torch.stack(collected_pred_f1_for_fid).to(DEVICE)
            fid_metric.update(gt_batch, real=True)
            fid_metric.update(pred_batch, real=False)
            summary["FID"] = fid_metric.compute().item()
    
    return results, summary

# ==================== FUNCIÓN DE PARSEO DEL DATASET ====================
def parse_atd12k_dataset(base_dir, max_triplets=None):
    """
    Parsea el dataset ATD-12k para obtener tripletes de frames.
    
    Args:
        base_dir: Directorio base del dataset
        max_triplets: Número máximo de tripletes a procesar (None para todos)
    
    Returns:
        Lista de tuplas (f0_path, f1_gt_path, f2_path)
    """
    triplets = []
    
    if not os.path.exists(base_dir):
        return triplets
    
    # Iterar sobre subdirectorios (cada uno es una secuencia)
    for sequence_folder in sorted(os.listdir(base_dir)):
        seq_path = os.path.join(base_dir, sequence_folder)
        if not os.path.isdir(seq_path):
            continue
        
        # Obtener frames ordenados
        frames = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(('.png', '.jpg'))])
        
        # Verificar que hay exactamente 3 frames
        if len(frames) == 3:
            f0 = os.path.join(seq_path, frames[0])
            f1_gt = os.path.join(seq_path, frames[1])
            f2 = os.path.join(seq_path, frames[2])
            
            # Verificar que todos los archivos existen
            if all(os.path.exists(f) for f in [f0, f1_gt, f2]):
                triplets.append((f0, f1_gt, f2))
                if max_triplets and len(triplets) >= max_triplets:
                    return triplets
    
    print(f"Found {len(triplets)} triplets in {base_dir}")
    return triplets

# ==================== SCRIPT PRINCIPAL ====================
if __name__ == "__main__":
    print(f"Starting ToonCrafter benchmark on {DEVICE}")
    
    # Rutas del modelo
    config_path = os.path.join(project_root_path, "configs/inference_512_v1.0.yaml")
    checkpoint_path = os.path.join(project_root_path, "checkpoints/tooncrafter_512_interp_v1/model.ckpt")
    
    # Cargar modelo
    try:
        model, sampler = load_tooncrafter_model_and_sampler(config_path, checkpoint_path, DEVICE)
        
        # Configurar decodificación frame por frame si está disponible
        if hasattr(model, 'perframe_ae') and model.perframe_ae:
            model.en_and_decode_n_samples_a_time = 1
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Ruta del dataset ATD-12k
    atd12k_dir = '/mnt/c/Users/BM10/Desktop/Mai/ToonCrafter/datasets/ADT-12k/test_2k_original'
    
    # Parsear dataset
    triplets = parse_atd12k_dataset(atd12k_dir, max_triplets=None)
    
    # Crear datos dummy si no se encuentra el dataset
    if not triplets:
        print("No triplets found, using dummy data")
        dummy_dir = "temp_dummy"
        os.makedirs(dummy_dir, exist_ok=True)
        
        # Crear imágenes dummy de colores
        for color, name in [("red", "f0"), ("green", "f1"), ("blue", "f2")]:
            path = os.path.join(dummy_dir, f"{name}.png")
            Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), color).save(path)
        
        triplets = [(
            os.path.join(dummy_dir, "f0.png"),
            os.path.join(dummy_dir, "f1.png"),
            os.path.join(dummy_dir, "f2.png")
        )]
    
    # Parámetros de inferencia
    params = {
        "ddim_steps": 50,         # Número de pasos de denoising
        "guidance_scale": 7.5,    # Escala de classifier-free guidance
        "eta": 1.0,              # Parámetro eta para DDIM
        "seed": 42,              # Semilla base
        "fs": 10,                # Frame skip
        "guidance_rescale": 0.7  # Rescalado de guidance
    }
    
    # Ejecutar benchmark
    print(f"\nRunning benchmark on {len(triplets)} triplets")
    results, summary = run_atd12k_evaluation_loop(triplets, model, sampler, params)
    
    # Mostrar resumen
    print("\n=== Benchmark Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    # Limpiar directorio dummy si existe
    if 'dummy_dir' in locals() and os.path.exists(dummy_dir):
        import shutil
        shutil.rmtree(dummy_dir)