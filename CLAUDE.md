# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of the original StableAvatar project for infinite-length audio-driven avatar video generation. The focus is on optimizing deployment, inference speed, and output quality on single GPU remote pods.

## Current Environment

- **Python**: 3.12.11 (in venv at `/workspace/StableAvatar/venv`)
- **PyTorch**: 2.7.0+cu128
- **CUDA**: 12.8
- **GPU**: NVIDIA H100 PCIe (80GB VRAM)
- **Key Packages**:
  - torch: 2.7.0+cu128
  - torchvision: 0.22.0+cu128
  - torchaudio: 2.7.0+cu128
  - diffusers: 0.30.1
  - transformers: 4.51.3
  - accelerate: 1.10.0

**Important**: Always activate the venv before running commands:
```bash
source /workspace/StableAvatar/venv/bin/activate
```

## Deployment Environment

- **Platform**: Remote pod deployment
- **GPU Configuration**: Single GPU only (no multi-GPU considerations needed)
- **Primary Goal**: Optimize inference speed and output quality

## Core Architecture for Inference

### Key Models
- **WanTransformer3DFantasyModel**: Main diffusion transformer (1.3B/14B variants) at `wan/models/wan_fantasy_transformer3d_1B.py`
- **FantasyTalkingVocalCondition**: Audio processing module at `wan/models/vocal_projector_fantasy_1B.py`
- **AutoencoderKLWan**: VAE for video encoding/decoding at `wan/models/wan_vae.py`
- **WanI2VTalkingInferenceLongPipeline**: Main inference pipeline at `wan/pipeline/wan_inference_long_pipeline.py`

### Supported Resolutions
- 512x512 (square format)
- 480x832 (vertical format)
- 832x480 (horizontal format)

## Quick Deployment Commands

### Environment Setup
```bash
# Activate virtual environment
source /workspace/StableAvatar/venv/bin/activate

# Download model weights if not present
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
```

### Running Inference
```bash
# Activate venv first
source /workspace/StableAvatar/venv/bin/activate

# Basic inference with default settings
bash inference.sh

# Launch Gradio web interface
python app.py

# Process audio from video
python audio_extractor.py --video_path="input.mp4" --saved_audio_path="audio.wav"

# Separate vocals for better lip sync
python vocal_seperator.py --audio_separator_model_file="checkpoints/Kim_Vocal_2.onnx" --audio_file_path="audio.wav" --saved_vocal_path="vocal.wav"
```

## Speed Optimization Strategies (H100 Specific)

### Memory Management Modes
With H100's 80GB VRAM, we can use the most performant settings:
- **Recommended**: `model_full_load` - Utilize full GPU memory for maximum speed
- `model_cpu_offload`: Only if processing extremely long videos
- `sequential_cpu_offload`: Not recommended with H100
- `model_cpu_offload_and_qfloat8`: Can experiment for even faster inference

### H100 Optimized Parameters
```bash
# In inference.sh, set:
--GPU_memory_mode="model_full_load"
--sample_steps=35  # Balance of speed and quality
--overlap_window_length=8  # Good smoothness without too much overhead
--clip_sample_n_frames=49  # Can handle larger batches with 80GB VRAM
```

### Key Performance Parameters
- `--sample_steps`: 30-35 for speed, 40-50 for quality
- `--overlap_window_length`: 5-8 for speed, 10-15 for smoothness
- `--clip_sample_n_frames`: Can increase to 49+ with H100
- Flash Attention: Should install for H100 optimization

## Quality Optimization Strategies

### Audio Synchronization
- `--sample_audio_guide_scale`: Increase (4-6) for better lip sync
- Use vocal separation (`vocal_seperator.py`) to remove background music
- Process clean audio without noise for best results

### Visual Quality
- `--sample_steps`: 40-50 for highest quality
- `--overlap_window_length`: 10-15 for smoothest transitions
- `--sample_text_guide_scale`: 3-6 for prompt adherence
- Choose appropriate checkpoint:
  - `transformer3d-square.pt`: Better for square videos
  - `transformer3d-rec-vec.pt`: Better for rectangular videos

### Prompt Engineering
Format: `[First frame description]-[Human behavior]-[Background (optional)]`
- Be specific about desired appearance and actions
- Include motion descriptions for dynamic videos

## Critical Files for Optimization

### Inference Pipeline
- `inference.py`: Main inference script with configuration options
- `wan/pipeline/wan_inference_long_pipeline.py`: Pipeline logic, sliding window implementation
- `wan/models/wan_fantasy_transformer3d_1B.py`: Model architecture, attention mechanisms
- `wan/models/vocal_projector_fantasy_1B.py`: Audio processing and synchronization

### Performance Tuning
- `wan/utils/fp8_optimization.py`: FP8 quantization utilities
- `wan/models/cache_utils.py`: TeaCache implementation for speed
- Flash Attention integration points in transformer models

## H100-Specific Optimizations

### Leverage H100 Features
- Use FP8 computation when available
- Enable flash_attn for optimized attention
- Increase batch sizes due to large VRAM
- Use `model_full_load` mode by default

### Recommended Settings for H100
```python
# Optimal configuration for H100
config = {
    "GPU_memory_mode": "model_full_load",
    "sample_steps": 40,  # Can afford higher quality
    "overlap_window_length": 10,
    "clip_sample_n_frames": 49,
    "sample_text_guide_scale": 4.5,
    "sample_audio_guide_scale": 5.0,
}
```

## Common Issues and Solutions

### Performance Not Optimal on H100
- Ensure flash_attn is installed
- Check CUDA version compatibility (12.8)
- Use FP16/BF16 computation modes
- Verify venv is activated

### Slow Inference Despite H100
- Check if model is actually on GPU
- Verify `model_full_load` mode
- Increase `clip_sample_n_frames`
- Ensure no CPU offloading

### Poor Lip Sync
- Increase `--sample_audio_guide_scale` to 5-6
- Use vocal separation
- Ensure audio sample rate is correct

### Video Quality Issues
- With H100, use `--sample_steps=45-50`
- Increase `--overlap_window_length` to 12-15
- Use appropriate checkpoint for aspect ratio