import argparse
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm as _tqdm

from modules.commons import str2bool

# Set up device and torch configurations
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

# Global variables to store model instances
vc_wrapper_v2 = None


def load_v2_models(args):
    """Load V2 models using the wrapper from app.py"""
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(
        ar_checkpoint_path=args.ar_checkpoint_path,
        cfm_checkpoint_path=args.cfm_checkpoint_path,
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(
        max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device
    )

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper


def batch_convert_voice_v2(
    source: str | np.ndarray,
    target: str | np.ndarray,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    intelligebility_cfg_rate: float = 0.7,
    similarity_cfg_rate: float = 0.7,
    top_p: float = 0.7,
    temperature: float = 0.7,
    repetition_penalty: float = 1.5,
    convert_style: bool = False,
    anonymization_only: bool = False,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
    stream_output: bool = True,
):
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)

    if isinstance(source, str):
        source_wave = librosa.load(source, sr=vc_wrapper_v2.sr)[0]
    else:
        source_wave = source
    if isinstance(target, str):
        target_wave = librosa.load(target, sr=vc_wrapper_v2.sr)[0]
    else:
        target_wave = target

    # Limit target audio to 25 seconds
    target_wave = target_wave[
        : vc_wrapper_v2.sr * (vc_wrapper_v2.dit_max_context_len - 5)
    ]

    source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).float().to(device)
    target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).float().to(device)

    # Resample to 16kHz for feature extraction
    source_wave_16k = librosa.resample(
        source_wave, orig_sr=vc_wrapper_v2.sr, target_sr=16000
    )
    target_wave_16k = librosa.resample(
        target_wave, orig_sr=vc_wrapper_v2.sr, target_sr=16000
    )
    source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(device)
    target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(device)

    # Compute mel spectrograms
    source_mel = vc_wrapper_v2.mel_fn(source_wave_tensor)
    target_mel = vc_wrapper_v2.mel_fn(target_wave_tensor)
    source_mel_len = source_mel.size(2)
    target_mel_len = target_mel.size(2)

    # Set up chunk processing parameters
    max_context_window = (
        vc_wrapper_v2.sr // vc_wrapper_v2.hop_size * vc_wrapper_v2.dit_max_context_len
    )
    overlap_wave_len = vc_wrapper_v2.overlap_frame_len * vc_wrapper_v2.hop_size

    with torch.autocast(device_type=device.type, dtype=dtype):
        # Compute content features
        source_content_indices = vc_wrapper_v2._process_content_features(
            source_wave_16k_tensor, is_narrow=False
        )
        target_content_indices = vc_wrapper_v2._process_content_features(
            target_wave_16k_tensor, is_narrow=False
        )
        # Compute style features
        target_style = vc_wrapper_v2.compute_style(target_wave_16k_tensor)
        (
            prompt_condition,
            _,
        ) = vc_wrapper_v2.cfm_length_regulator(
            target_content_indices, ylens=torch.LongTensor([target_mel_len]).to(device)
        )

    # prepare for streaming
    generated_wave_chunks = []
    processed_frames = 0
    previous_chunk = None
    cond, _ = vc_wrapper_v2.cfm_length_regulator(
        source_content_indices, ylens=torch.LongTensor([source_mel_len]).to(device)
    )

    # Process in chunks for streaming
    max_source_window = max_context_window - target_mel.size(2)

    # Generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        original_len = cat_condition.size(1)
        # pad cat_condition to compile_len
        if vc_wrapper_v2.dit_compiled:
            cat_condition = torch.nn.functional.pad(
                cat_condition,
                (
                    0,
                    0,
                    0,
                    vc_wrapper_v2.compile_len - cat_condition.size(1),
                ),
                value=0,
            )
        with torch.autocast(
            device_type=device.type, dtype=torch.float32
        ):  # force CFM to use float32
            # Voice Conversion
            vc_mel = vc_wrapper_v2.cfm.inference(
                cat_condition,
                torch.LongTensor([original_len]).to(device),
                target_mel,
                target_style,
                diffusion_steps,
                inference_cfg_rate=[intelligebility_cfg_rate, similarity_cfg_rate],
                random_voice=anonymization_only,
            )
        vc_mel = vc_mel[:, :, target_mel_len:original_len]
        vc_wave = vc_wrapper_v2.vocoder(vc_mel).squeeze()[None]

        processed_frames, previous_chunk, should_break, mp3_bytes, full_audio = (
            vc_wrapper_v2._stream_wave_chunks(
                vc_wave,
                processed_frames,
                vc_mel,
                overlap_wave_len,
                generated_wave_chunks,
                previous_chunk,
                is_last_chunk,
                True,
            )
        )

        if mp3_bytes is not None:
            yield mp3_bytes, full_audio
        if should_break:
            break


def convert_voice_v2(source_audio_path, target_audio_path, args):
    """Convert voice using V2 model"""
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)

    # Use the generator function but collect all outputs
    generator = batch_convert_voice_v2(
        source=source_audio_path,
        target=target_audio_path,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        convert_style=args.convert_style,
        anonymization_only=args.anonymization_only,
        device=device,
        dtype=dtype,
        stream_output=True,
    )

    # Collect all outputs from the generator
    for output in generator:
        _, full_audio = output
    return full_audio


def main(args):
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()
    target_wave = librosa.load(args.target, sr=vc_wrapper_v2.sr)[0]
    if Path(args.source).is_dir():
        for source_file in _tqdm(list(Path(args.source).glob("*.*"))):
            if source_file.suffix.lower() not in [".wav", ".mp3", ".flac"]:
                print(f"Skipping unsupported file format: {source_file}")
                continue
            converted_audio = convert_voice_v2(str(source_file), target_wave, args)
            if converted_audio is None:
                print("Error: Failed to convert voice")
                return

        # Save the converted audio
        source_name = os.path.basename(args.source).split(".")[0]
        target_name = os.path.basename(args.target).split(".")[0]

        # Create a descriptive filename
        filename = f"seed_vc_v2_{source_name}_{target_name}_{args.length_adjust}_{args.diffusion_steps}_{args.similarity_cfg_rate}.wav"

        output_path = os.path.join(args.output, filename)
        save_sr, converted_audio = converted_audio
        sf.write(output_path, converted_audio, 16000)
    elif Path(args.source).suffix in [".jsonl"]:
        df = pd.read_json(args.source, lines=True)
        for idx, row in _tqdm(df.iterrows(), total=len(df)):
            source_file = row.get(
                "source",
                row.get("audio_filepath", row.get("filepath", row.get("audio", None))),
            )
            if source_file is None:
                print(f"Error: Failed to find source file for row {idx}")
                continue

            # Save the converted audio
            source_name = os.path.basename(source_file).split(".")[0]
            target_name = os.path.basename(args.target).split(".")[0]

            # Create a descriptive filename
            filename = f"seed_vc_v2_{source_name}_{target_name}_{args.length_adjust}_{args.diffusion_steps}_{args.similarity_cfg_rate}.wav"

            if Path(args.output, filename).exists():
                continue

            converted_audio = convert_voice_v2(str(source_file), args.target, args)
            if converted_audio is None:
                print(f"Error: Failed to convert voice for row {idx}")
                continue

            output_path = os.path.join(args.output, filename)
            save_sr, converted_audio = converted_audio
            sf.write(output_path, converted_audio, 16000)
    else:
        converted_audio = convert_voice_v2(args.source, args.target, args)
        if converted_audio is None:
            print("Error: Failed to convert voice")
            return

        # Save the converted audio
        source_name = os.path.basename(args.source).split(".")[0]
        target_name = os.path.basename(args.target).split(".")[0]

        # Create a descriptive filename
        filename = f"seed_vc_v2_{source_name}_{target_name}_{args.length_adjust}_{args.diffusion_steps}_{args.similarity_cfg_rate}.wav"

        output_path = os.path.join(args.output, filename)
        save_sr, converted_audio = converted_audio
        sf.write(output_path, converted_audio, 16000)

    end_time = time.time()

    print(f"Voice conversion completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Conversion Inference Script")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source audio file or directory containing audio files",
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Path to target/reference audio file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for converted audio",
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=30, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--length-adjust",
        type=float,
        default=1.0,
        help="Length adjustment factor (<1.0 for speed-up, >1.0 for slow-down)",
    )
    parser.add_argument(
        "--compile",
        type=bool,
        default=False,
        help="Whether to compile the model for faster inference",
    )

    # V2 specific arguments
    parser.add_argument(
        "--intelligibility-cfg-rate",
        type=float,
        default=0.7,
        help="Intelligibility CFG rate for V2 model",
    )
    parser.add_argument(
        "--similarity-cfg-rate",
        type=float,
        default=0.9,
        help="Similarity CFG rate for V2 model",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling parameter for V2 model"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature sampling parameter for V2 model",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for V2 model",
    )
    parser.add_argument(
        "--convert-style",
        type=str2bool,
        default=False,
        help="Convert style/emotion/accent for V2 model",
    )
    parser.add_argument(
        "--anonymization-only",
        type=str2bool,
        default=False,
        help="Anonymization only mode for V2 model",
    )

    # V2 custom checkpoints
    parser.add_argument(
        "--ar-checkpoint-path",
        type=str,
        default=None,
        help="Path to custom checkpoint file",
    )
    parser.add_argument(
        "--cfm-checkpoint-path",
        type=str,
        default=None,
        help="Path to custom checkpoint file",
    )

    args = parser.parse_args()
    main(args)
