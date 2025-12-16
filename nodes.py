from __future__ import annotations

import os
import uuid
import wave
import threading
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

import time
import base64
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import comfy.utils
import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def _split_main_and_thinking(text: str) -> Tuple[str, str]:
    """
    MAIN:
      - remove all <think>...</think> blocks
      - ensure MAIN does NOT start with a newline (strip only leading \r/\n)
    THINKING:
      - concatenate contents of all <think>...</think> blocks (without the tags)
      - return "" if no blocks
    """
    if not text:
        return "", ""

    matches = _THINK_RE.findall(text)
    if not matches:
        return text.lstrip("\r\n"), ""

    thinking = "\n\n".join(m.strip("\r\n") for m in matches).strip("\r\n")
    main = _THINK_RE.sub("", text).lstrip("\r\n")
    return main, thinking

# ----------------------------
# Model discovery
# ----------------------------

_ALLOWED_EXTS = (".gguf", ".safetensors")
_SUBFOLDERS = ("clip", "text_encoders", "LLM")

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}  # (abs_path, device) -> loaded backend model bundle
_MODEL_INDEX: Dict[str, str] = {}  # display_name -> abs_path


def _norm_display(s: str) -> str:
    return s.replace("\\", "/")


def _get_search_roots() -> List[str]:
    roots: List[str] = []
    # Use ComfyUI's registered model paths when possible (supports extra_model_paths)
    try:
        roots.extend(folder_paths.get_folder_paths("clip"))
    except Exception:
        pass
    # "text_encoders" may not exist in all ComfyUI versions/configs; fall back to models_dir/text_encoders
    try:
        roots.extend(folder_paths.get_folder_paths("text_encoders"))
    except Exception:
        roots.append(os.path.join(folder_paths.models_dir, "text_encoders"))
    # LLM folder (not standard in all installs)
    roots.append(os.path.join(folder_paths.models_dir, "LLM"))

    # Ensure the canonical "models/<subfolder>" paths are included
    for sf in _SUBFOLDERS:
        p = os.path.join(folder_paths.models_dir, sf)
        if p not in roots:
            roots.append(p)

    # De-dup while preserving order
    seen = set()
    out = []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _scan_models() -> Dict[str, str]:
    model_index: Dict[str, str] = {}
    roots = _get_search_roots()

    for root in roots:
        if not os.path.isdir(root):
            continue

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith(_ALLOWED_EXTS):
                    continue

                abs_path = os.path.join(dirpath, fn)

                # Prefer a stable-ish display name relative to models_dir when possible
                display = None
                try:
                    rel_to_models = os.path.relpath(abs_path, folder_paths.models_dir)
                    if not rel_to_models.startswith(".."):
                        display = _norm_display(rel_to_models)
                except Exception:
                    pass

                if display is None:
                    rel_to_root = os.path.relpath(abs_path, root)
                    root_tag = os.path.basename(root.rstrip("/\\"))
                    display = _norm_display(os.path.join(root_tag, rel_to_root))

                # Ensure uniqueness
                base = display
                i = 2
                while display in model_index and model_index[display] != abs_path:
                    display = f"{base} ({i})"
                    i += 1

                model_index[display] = abs_path

    return dict(sorted(model_index.items(), key=lambda kv: kv[0].lower()))


def _refresh_model_index() -> None:
    global _MODEL_INDEX
    _MODEL_INDEX = _scan_models()


def _find_hf_root_from_file(model_file: str, max_up: int = 6) -> Optional[str]:
    """
    For a .safetensors file, walk up parents to find a folder with config.json (HF-style).
    """
    d = os.path.dirname(model_file)
    for _ in range(max_up):
        if os.path.isfile(os.path.join(d, "config.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


# ----------------------------
# Progress reporting
# ----------------------------

@dataclass
class TokenProgress:
    total: int
    start_t: float
    last_print_t: float = 0.0
    last_print_tokens: int = 0


class ProgressReporter:
    def __init__(self, total_tokens: int, label: str = "LLM"):
        self.pbar = comfy.utils.ProgressBar(total_tokens)
        self.state = TokenProgress(total=total_tokens, start_t=time.time())
        self.label = label

        # Initialize UI bar
        self.pbar.update_absolute(0)

    def update(self, processed_tokens: int) -> None:
        if processed_tokens < 0:
            processed_tokens = 0
        if processed_tokens > self.state.total:
            processed_tokens = self.state.total

        self.pbar.update_absolute(processed_tokens)

        # Console log throttled to ~1 Hz
        now = time.time()
        if now - self.state.last_print_t < 1.0 and processed_tokens < self.state.total:
            return

        elapsed = max(now - self.state.start_t, 1e-9)
        pct = (processed_tokens / max(self.state.total, 1)) * 100.0

        tok_per_s = processed_tokens / elapsed if processed_tokens > 0 else 0.0
        if tok_per_s >= 1.0:
            rate = f"{tok_per_s:.2f} tok/s"
        else:
            s_per_tok = (elapsed / processed_tokens) if processed_tokens > 0 else float("inf")
            rate = f"{s_per_tok:.2f} s/tok"

        print(f"[{self.label}] {processed_tokens}/{self.state.total} ({pct:.1f}%) @ {rate}")

        self.state.last_print_t = now
        self.state.last_print_tokens = processed_tokens


# ----------------------------
# Image helpers
# ----------------------------

def _comfy_image_to_pil(image_tensor: Any) -> Optional[Image.Image]:
    if image_tensor is None or not isinstance(image_tensor, torch.Tensor):
        return None

    t = image_tensor
    if t.dim() == 4:
        t = t[0]
    if t.dim() != 3:
        return None

    # Handle both HWC and CHW
    # If it looks like CHW (C in {1,3,4} and last dim not channels), permute to HWC.
    if t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
        t = t.permute(1, 2, 0)

    # Keep RGB
    if t.shape[-1] >= 3:
        t = t[..., :3]

    x = t.detach().float().cpu()

    # If the tensor is in [0,255], normalize
    if x.numel() > 0 and x.max().item() > 1.0:
        x = x / 255.0

    arr = (x.clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def _pil_to_data_url_png(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ----------------------------
# Backends
# ----------------------------

def _llama_supports_gpu_offload() -> bool:
    try:
        import llama_cpp
        # Low-level capability probe exposed by llama-cpp-python. :contentReference[oaicite:1]{index=1}
        return bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def _load_llama_cpp(model_path: str, device: str, n_ctx: int) -> Any:
    from llama_cpp import Llama

    if device == "cpu":
        return Llama(
            model_path=model_path,
            n_ctx=int(n_ctx),
            n_gpu_layers=0,              # 0 => CPU :contentReference[oaicite:2]{index=2}
            main_gpu=0,                  # GPU index when applicable :contentReference[oaicite:3]{index=3}
            verbose=False,
        )

    gpu_capable = _llama_supports_gpu_offload()

    if device == "cuda" and not gpu_capable:
        raise RuntimeError(
            "Device='cuda' requested, but llama-cpp-python reports no GPU offload support. "
            "Install a CUDA-enabled llama-cpp-python build/wheel."
        )

    # default/cuda: prefer GPU offload
    if gpu_capable:
        # Never include 0 in the cuda-enforced path; ensure at least some offload.
        layer_trials = (-1, 80, 60, 40, 20, 10, 5, 1)  # -1 => all layers :contentReference[oaicite:4]{index=4}
        if device == "default":
            # default may fall back to CPU if everything fails
            layer_trials = layer_trials + (0,)
    else:
        # default with no GPU backend: force CPU, but warn.
        print("[LLM] llama-cpp-python has no GPU backend; GGUF will run on CPU. "
              "Install CUDA-enabled llama-cpp-python for GPU offload.")
        layer_trials = (0,)

    last_err = None
    for n_gpu_layers in layer_trials:
        try:
            return Llama(
                model_path=model_path,
                n_ctx=int(n_ctx),
                n_gpu_layers=int(n_gpu_layers),
                main_gpu=0,
                verbose=False,
            )
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError("Failed to load GGUF model.") from last_err


def _run_llama_cpp(
    llm: Any,
    instructions: str,
    prompt: str,
    image_pil: Optional[Image.Image],
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    reporter: ProgressReporter,
) -> str:
    # Build messages; if image provided, use content parts with image_url.
    messages: List[Dict[str, Any]] = []
    ins = (instructions or "").strip()
    if ins:
        messages.append({"role": "system", "content": ins})

    if image_pil is not None:
        data_url = _pil_to_data_url_png(image_pil)
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt or ""},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": prompt or ""})

    out_parts: List[str] = []
    processed = 0

    stop_seqs = ["</s>", "<|eot_id|>", "<|end_of_text|>", "<|end|>", "<|im_end|>"]

    # Use "unlimited" generation at the backend level (max_tokens <= 0) and apply
    # our own stopping based on the node's max_tokens counter for robustness.
    # (llama-cpp-python documents that max_tokens <= 0 means unlimited and depends on n_ctx.)
    kwargs = dict(
        messages=messages,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        seed=int(seed),
        max_tokens=-1,
        stream=True,
    )

    try:
        stream = llm.create_chat_completion(**kwargs, stop=stop_seqs)
    except TypeError:
        # Older llama-cpp-python builds may not accept `stop=` for chat completion.
        stream = llm.create_chat_completion(**kwargs)


    # Robust streaming parse
    for chunk in stream:
        try:
            choice0 = chunk.get("choices", [{}])[0]
            delta = choice0.get("delta", {}) or {}
            piece = delta.get("content") or ""
            if not piece:
                # some implementations may stream different keys; ignore empties
                continue

            out_parts.append(piece)
            processed += 1
            reporter.update(processed)

            if processed >= max_tokens:
                break
        except Exception:
            # keep going; worst case you still get output at the end
            continue

    reporter.update(min(processed, max_tokens))
    return "".join(out_parts)


def _load_transformers(model_root: str, device: str, want_vision: bool) -> Dict[str, Any]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    bundle: Dict[str, Any] = {}
    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True, use_fast=True)
    bundle["tokenizer"] = tokenizer

    torch_dtype = torch.float16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32

    model = None
    processor = None

    if want_vision:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_root, trust_remote_code=True)
        except Exception:
            processor = None

        # Prefer the VLM auto-class used by the image-text-to-text task
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_root,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if device != "cpu" else None,
            )
        except Exception:
            model = None

    if model is None:
        # Fallback to causal LM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_root,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if device != "cpu" else None,
            )
        except Exception:
            # Last-resort: load without device_map then move
            model = AutoModelForCausalLM.from_pretrained(
                model_root,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )

    if device == "cpu":
        model = model.to("cpu")
    else:
        if torch.cuda.is_available():
            # If device_map was used, accelerate handles placement; otherwise move to cuda
            try:
                if getattr(model, "hf_device_map", None) is None:
                    model = model.to("cuda")
            except Exception:
                pass

    model.eval()
    bundle["model"] = model
    bundle["processor"] = processor
    return bundle


def _run_transformers(
    bundle: Dict[str, Any],
    instructions: str,
    prompt: str,
    image_pil: Optional[Image.Image],
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    reporter: ProgressReporter,
) -> str:
    from transformers import StoppingCriteria, StoppingCriteriaList

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    processor = bundle.get("processor", None)

    # Combine prompt
    ins = (instructions or "").strip()
    user = (prompt or "").strip()
    if ins:
        full_text = f"{ins}\n\n{user}"
    else:
        full_text = user

    # Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.manual_seed(int(seed))

    # Prepare inputs
    if image_pil is not None and processor is not None:
        inputs = processor(images=image_pil, text=full_text, return_tensors="pt")
    else:
        inputs = tokenizer(full_text, return_tensors="pt")

    prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

    # Best-effort move inputs to model device (works for non-device_map models)
    try:
        target_device = getattr(model, "device", torch.device("cpu"))
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(target_device)
    except Exception:
        pass

    class _ProgressStopper(StoppingCriteria):
        def __init__(self, base_len: int, max_new: int, rep: ProgressReporter):
            super().__init__()
            self.base_len = base_len
            self.max_new = max_new
            self.rep = rep
            self.last = 0

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            cur_new = int(input_ids.shape[-1]) - self.base_len
            if cur_new != self.last:
                self.last = cur_new
                self.rep.update(min(cur_new, self.max_new))
            return cur_new >= self.max_new

    stopper = _ProgressStopper(prompt_len, int(max_tokens), reporter)

    do_sample = float(temperature) > 0.0
    gen = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        top_k=int(top_k) if do_sample else None,
        stopping_criteria=StoppingCriteriaList([stopper]),
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )

    # Decode only newly generated tokens
    new_tokens = gen[0][prompt_len:] if (gen is not None and gen.shape[0] > 0) else gen[0]
    reporter.update(min(int(new_tokens.shape[-1]), int(max_tokens)))
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# -----------------------------
# Model discovery (models/asr)
# -----------------------------
def _models_root() -> str:
    # Prefer folder_paths.models_dir when available; otherwise derive from module location.
    if hasattr(folder_paths, "models_dir"):
        return folder_paths.models_dir
    return os.path.join(os.path.dirname(os.path.realpath(folder_paths.__file__)), "models")


def _asr_root() -> str:
    return os.path.join(_models_root(), "asr")


def list_asr_model_dirs() -> List[str]:
    base = _asr_root()
    if not os.path.isdir(base):
        return ["(models/asr not found)"]
    dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    return dirs if dirs else ["(no models found)"]


# -----------------------------
# Audio preprocessing utilities
# -----------------------------
def _to_mono(waveform_bct: torch.Tensor) -> torch.Tensor:
    """
    waveform_bct: [B, C, T] -> returns [B, 1, T] (mean over channels if C>1)
    """
    if waveform_bct.dim() != 3:
        raise ValueError(f"Expected AUDIO waveform tensor [B,C,T], got shape {tuple(waveform_bct.shape)}")
    if waveform_bct.shape[1] == 1:
        return waveform_bct
    return waveform_bct.mean(dim=1, keepdim=True)


def _resample_linear(waveform_b1t: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """
    Fallback resampler (linear interpolation). For higher quality install torchaudio and use it upstream.
    waveform_b1t: [B,1,T]
    """
    if src_sr == dst_sr:
        return waveform_b1t
    b, c, t = waveform_b1t.shape
    new_t = int(round(t * (dst_sr / float(src_sr))))
    # interpolate expects [N,C,L]
    return F.interpolate(waveform_b1t, size=new_t, mode="linear", align_corners=False)


def _write_wav_mono_16k(path: str, waveform_b1t: torch.Tensor, sr: int) -> None:
    """
    Writes PCM16 WAV, mono, sr.
    """
    waveform = waveform_b1t.detach().to(torch.float32).cpu()
    waveform = waveform.squeeze(0).squeeze(0)  # [T]
    waveform = waveform.clamp(-1.0, 1.0)

    pcm16 = (waveform.numpy() * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


def prepare_mono_wav_16k(audio: Dict, out_dir: Optional[str] = None) -> str:
    """
    audio: ComfyUI AUDIO dict: { "waveform": Tensor[B,C,T], "sample_rate": int }.[2]
    Returns a temp WAV filepath (16kHz mono).
    """
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Expected ComfyUI AUDIO dict with key 'waveform'.")

    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate", 16000))

    # Ensure batch exists
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1,C,T]

    waveform = _to_mono(waveform)              # [B,1,T]
    waveform = _resample_linear(waveform, sr, 16000)  # [B,1,T] @ 16k

    # Use first item only (single-file ASR). If you pass a batch, only index 0 is transcribed.
    waveform_0 = waveform[0:1, :, :]

    if out_dir is None:
        # Prefer ComfyUI temp directory when available.[3]
        try:
            out_dir = folder_paths.get_temp_directory()
        except Exception:
            out_dir = os.path.join(os.getcwd(), "temp")

    cache_dir = os.path.join(out_dir, "canary_qwen_asr_cache")
    fname = f"asr_{uuid.uuid4().hex}.wav"
    wav_path = os.path.join(cache_dir, fname)

    _write_wav_mono_16k(wav_path, waveform_0, 16000)
    return wav_path


# -----------------------------
# Model cache
# -----------------------------
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = {}  # (model_path, device, use_bf16) -> model


def load_salm(model_path: str, device: str, use_bf16: bool):
    """
    Loads NeMo SALM from a local folder (models/asr/<dir>) or HF-style folder contents.
    """
    # Import lazily to avoid breaking ComfyUI startup if deps are missing.
    from nemo.collections.speechlm2.models import SALM  # requires NeMo per model card.[1]

    key = (model_path, device, bool(use_bf16))
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        model = SALM.from_pretrained(model_path)  # local path supported by NeMo/HF layout.[1]

        # Move model
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            if use_bf16:
                try:
                    model = model.to(dtype=torch.bfloat16)
                except Exception:
                    pass
        else:
            model = model.cpu()

        model.eval()
        _MODEL_CACHE[key] = model
        return model


# -----------------------------
# ComfyUI Nodes
# -----------------------------
class CanaryQwenASR:
    """
    Audio -> Text transcription using NVIDIA NeMo Canary-Qwen-2.5B (ASR mode).[1]
    """

    @classmethod
    def INPUT_TYPES(cls):
        # COMBO is defined by providing a list[str] in INPUT_TYPES.[2]
        return {
            "required": {
                "audio": ("AUDIO",),  # ComfyUI AUDIO dict: waveform + sample_rate.[2]
                "model_dir": (list_asr_model_dirs(),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "use_bf16": ("BOOLEAN", {"default": True}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
            },
            "optional": {
                "user_prompt": ("STRING", {"default": "Transcribe the following:"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "lopi999/llm"

    def transcribe(
        self,
        audio: Dict,
        model_dir: str,
        device: str,
        use_bf16: bool,
        max_new_tokens: int,
        user_prompt: str = "Transcribe the following:",
    ) -> Tuple[str]:

        if model_dir.startswith("("):
            raise RuntimeError("No ASR model directory found. Put the model under ComfyUI/models/asr/<folder>.")

        model_path = os.path.join(_asr_root(), model_dir)
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Selected model_dir does not exist: {model_path}")

        wav_path = prepare_mono_wav_16k(audio)

        model = load_salm(model_path, device=device, use_bf16=use_bf16)

        # Per NVIDIA example: prompts include content with the audio locator tag + audio file path list.[1]
        content = f"{user_prompt} {model.audio_locator_tag}"
        prompts = [[{"role": "user", "content": content, "audio": [wav_path]}]]

        with torch.inference_mode():
            answer_ids = model.generate(prompts=prompts, max_new_tokens=int(max_new_tokens))

        text = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        return (text,)

class node_LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        _refresh_model_index()
        model_list = list(_MODEL_INDEX.keys()) or ["(no models found)"]

        return {
            "required": {
                "instructions": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "model": (model_list,),
                "device": (["default", "cpu", "cuda"],),
                "tokens": ("INT", {"default": 256, "min": 1, "max": 262144}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 262144}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100000}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("MAIN", "THINKING")
    FUNCTION = "run"
    CATEGORY = "lopi999/llm"

    def run(
        self,
        instructions: str,
        prompt: str,
        seed: int,
        model: str,
        device: str,
        tokens: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        image: Any = None,
    ):
        if model not in _MODEL_INDEX:
            raise ValueError("Selected model not found. Refresh model list and try again.")

        # Enforce: if max_tokens > tokens, clamp max_tokens = tokens
        tokens_i = max(int(tokens), 1)
        max_tokens_i = max(int(max_tokens), 1)
        if max_tokens_i > tokens_i:
            max_tokens_i = tokens_i

        abs_path = _MODEL_INDEX[model]
        ext = os.path.splitext(abs_path)[1].lower()

        image_pil = _comfy_image_to_pil(image)
        want_vision = image_pil is not None

        # Backend-specific progress + caching:
        try:
            if ext == ".gguf":
                # llama.cpp generation length is constrained by the model context (n_ctx):
                # prompt tokens + generated tokens must fit inside n_ctx.
                prompt_chars = len((instructions or "")) + len((prompt or ""))
                approx_prompt_tokens = max(32, prompt_chars // 4)  # ~4 chars/token heuristic
                approx_image_tokens = 1024 if want_vision else 0
                desired_n_ctx = max(512, min(tokens_i + approx_prompt_tokens + approx_image_tokens + 64, 262144))

                cache_key = (abs_path, device, desired_n_ctx)
                if cache_key not in _MODEL_CACHE:
                    llm = _load_llama_cpp(abs_path, device, desired_n_ctx)
                    _MODEL_CACHE[cache_key] = {"backend": "llama_cpp", "llm": llm}
                llm = _MODEL_CACHE[cache_key]["llm"]

                # Clamp max_tokens to what actually fits into the context window to avoid early stops.
                try:
                    ctx = int(llm.n_ctx()) if hasattr(llm, "n_ctx") else int(desired_n_ctx)
                    prompt_for_count = ((instructions or "") + "\n\n" + (prompt or "")).strip()
                    prompt_tok = len(llm.tokenize(prompt_for_count.encode("utf-8"))) if prompt_for_count else 0
                    max_new_allowed = max(ctx - prompt_tok - 64, 1)
                    if max_tokens_i > max_new_allowed:
                        print(f"[LLM] Clamping max_tokens {max_tokens_i} -> {max_new_allowed} (n_ctx={ctx}, prompt≈{prompt_tok} tok)")
                        max_tokens_i = max_new_allowed
                except Exception:
                    pass

                reporter = ProgressReporter(total_tokens=max_tokens_i, label="LLM")

                text = _run_llama_cpp(
                    llm=llm,
                    instructions=instructions,
                    prompt=prompt,
                    image_pil=image_pil,
                    seed=seed,
                    max_tokens=max_tokens_i,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    reporter=reporter,
                )
                main, thinking = _split_main_and_thinking(text)
                return (main, thinking)

            if ext == ".safetensors":
                reporter = ProgressReporter(total_tokens=max_tokens_i, label="LLM")

                hf_root = _find_hf_root_from_file(abs_path)
                if hf_root is None:
                    raise ValueError(
                        "This .safetensors file is not inside a HuggingFace-style model folder "
                        "(missing config.json in parents). Put it under models/LLM/<model_name>/ with config/tokenizer."
                    )

                # Cache separately for text-only vs vision bundle (processor/model class can differ)
                cache_key_tf = (hf_root + ("|vision" if want_vision else "|text"), device)
                if cache_key_tf not in _MODEL_CACHE:
                    bundle = _load_transformers(hf_root, device, want_vision)
                    _MODEL_CACHE[cache_key_tf] = {"backend": "transformers", **bundle}

                bundle = _MODEL_CACHE[cache_key_tf]
                text = _run_transformers(
                    bundle=bundle,
                    instructions=instructions,
                    prompt=prompt,
                    image_pil=image_pil,
                    seed=seed,
                    max_tokens=max_tokens_i,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    reporter=reporter,
                )
                main, thinking = _split_main_and_thinking(text)
                return (main, thinking)

            raise ValueError(f"Unsupported model extension: {ext}")

        except Exception as e:
            print("[LLM] ERROR:", str(e))
            traceback.print_exc()
            raise


NODE_CLASS_MAPPINGS = {
    "CanaryQwenASR": CanaryQwenASR,
    "LLMNode(lopi999)": node_LLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanaryQwenASR": "Canary-Qwen-2.5B ASR (Audio→Text)",
    "LLMNode(lopi999)": "lopi999 LLM Node",
}
