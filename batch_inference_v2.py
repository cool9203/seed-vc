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
    sources: list[str] | list[np.ndarray],
    targets: list[str] | list[np.ndarray],
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
) -> list[torch.Tensor]:
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)
    source_waves = list()
    target_waves = list()

    assert len(sources) == len(targets)

    for _source_wave, _target_wave in zip(sources, targets):
        if isinstance(_source_wave, (str, Path)):
            source_waves.append(librosa.load(_source_wave, sr=vc_wrapper_v2.sr)[0])
        else:
            source_waves.append(_source_wave)
        if isinstance(_target_wave, (str, Path)):
            target_waves.append(librosa.load(_target_wave, sr=vc_wrapper_v2.sr)[0])
        else:
            target_waves.append(_target_wave)

    # Limit target audio to 25 seconds
    target_waves = np.array(target_waves)[
        :, : vc_wrapper_v2.sr * (vc_wrapper_v2.dit_max_context_len - 5)
    ]

    source_waves_pad = [torch.tensor(source_wave) for source_wave in source_waves]
    source_wave_tensor = (
        torch.nn.utils.rnn.pad_sequence(
            source_waves_pad, batch_first=True, padding_value=0.0
        )
        .float()
        .to(device)
    )
    target_wave_tensor = torch.tensor(target_waves).float().to(device)

    # Resample to 16kHz for feature extraction
    source_wave_16k = [
        librosa.resample(source_wave, orig_sr=vc_wrapper_v2.sr, target_sr=16000)
        for source_wave in source_waves
    ]
    target_wave_16k = [
        librosa.resample(target_wave, orig_sr=vc_wrapper_v2.sr, target_sr=16000)
        for target_wave in target_waves
    ]

    source_wave_16k_pad = [torch.tensor(source_wave) for source_wave in source_wave_16k]
    source_wave_16k_tensor = (
        torch.nn.utils.rnn.pad_sequence(
            source_wave_16k_pad, batch_first=True, padding_value=0.0
        )
        .float()
        .to(device)
    )
    target_wave_16k_tensor = torch.tensor(np.array(target_wave_16k)).to(device)

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
        source_content_indices = torch.tensor(
            np.array(
                [
                    vc_wrapper_v2._process_content_features(
                        _source_wave_16k_tensor.unsqueeze(0), is_narrow=False
                    ).numpy()[0]
                    for _source_wave_16k_tensor in source_wave_16k_tensor
                ]
            )
        )
        target_content_indices = torch.tensor(
            np.array(
                [
                    vc_wrapper_v2._process_content_features(
                        _target_wave_16k_tensor.unsqueeze(0), is_narrow=False
                    ).numpy()[0]
                    for _target_wave_16k_tensor in target_wave_16k_tensor
                ]
            )
        )
        # Compute style features
        target_style = torch.tensor(
            np.array(
                [
                    vc_wrapper_v2.compute_style(
                        _target_wave_16k_tensor.unsqueeze(0)
                    ).numpy()[0]
                    for _target_wave_16k_tensor in target_wave_16k_tensor
                ]
            )
        )
        (
            prompt_condition,
            _,
        ) = vc_wrapper_v2.cfm_length_regulator(
            target_content_indices,
            ylens=torch.LongTensor([target_mel_len] * len(source_waves)).to(device),
        )

    # prepare for streaming
    generated_wave_chunks = [[] for _ in range(len(source_waves))]
    processed_frames = 0
    previous_chunk = [None for _ in range(len(source_waves))]
    full_audios = [None for _ in range(len(source_waves))]
    cond, _ = vc_wrapper_v2.cfm_length_regulator(
        source_content_indices,
        ylens=torch.LongTensor([source_mel_len] * len(source_waves)).to(device),
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
                torch.LongTensor([original_len] * len(source_waves)).to(device),
                target_mel,
                target_style,
                diffusion_steps,
                inference_cfg_rate=[intelligebility_cfg_rate, similarity_cfg_rate],
                random_voice=anonymization_only,
            )
        vc_mel = vc_mel[:, :, target_mel_len:original_len]
        vc_wave = vc_wrapper_v2.vocoder(vc_mel).squeeze()[None]

        for i in range(len(source_waves)):
            (
                processed_frames,
                previous_chunk[i],
                should_break,
                mp3_bytes,
                full_audios[i],
            ) = vc_wrapper_v2._stream_wave_chunks(
                vc_wave[i],
                processed_frames,
                vc_mel[i],
                overlap_wave_len,
                generated_wave_chunks[i],
                previous_chunk[i],
                is_last_chunk,
                True,
            )

    print(full_audios)
    print(full_audios.size() if hasattr(full_audios, "size") else full_audios.shape)

    return full_audios  # type: ignore


def main(args):
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()
    target_wave = librosa.load(args.target, sr=vc_wrapper_v2.sr)[0]
    source_files: list[str] = list()
    if Path(args.source).is_dir():
        for source_file in _tqdm(
            list(Path(args.source).glob("*.*")), desc="Loading source files"
        ):
            if source_file.suffix.lower() not in [".wav", ".mp3", ".flac"]:
                print(f"Skipping unsupported file format: {source_file}")
                continue
            source_files.append(source_file)

    elif Path(args.source).suffix in [".jsonl"]:
        df = pd.read_json(args.source, lines=True)
        for idx, row in _tqdm(
            df.iterrows(), total=len(df), desc="Loading source files"
        ):
            source_file = row.get(
                "source",
                row.get("audio_filepath", row.get("filepath", row.get("audio", None))),
            )
            if source_file is None:
                print(f"Error: Failed to find source file for row {idx}")
                continue
            source_files.append(source_file)

    else:
        source_files.append(args.source)

    source_files = [
        source_file
        for source_file in _tqdm(source_files, desc="Filtering source files")
        if not Path(
            args.output,
            (
                "seed_vc_v2"
                + f"_{Path(source_file).stem}"
                + f"_{Path(args.target).stem}"
                + f"_{args.length_adjust}"
                + f"_{args.diffusion_steps}"
                + f"_{args.similarity_cfg_rate}.wav"
            ),
        ).exists()
    ]

    for batch in _tqdm(
        np.array_split(source_files, len(source_files) // args.batch_size + 1),
        desc="Converting voice",
    ):
        converted_audio = batch_convert_voice_v2(
            sources=batch.tolist(),
            targets=[target_wave] * len(batch),
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
        if converted_audio is None:
            raise Exception(f"Error: Failed to convert voice for {batch}")

        for idx, source in enumerate(batch):
            # Create a descriptive filename
            output_path = Path(
                args.output,
                (
                    "seed_vc_v2"
                    + f"_{Path(source).stem}"
                    + f"_{Path(args.target).stem}"
                    + f"_{args.length_adjust}"
                    + f"_{args.diffusion_steps}"
                    + f"_{args.similarity_cfg_rate}.wav"
                ),
            )
            sf.write(output_path, converted_audio[idx], 16000)

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
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
