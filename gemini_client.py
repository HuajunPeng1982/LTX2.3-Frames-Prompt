"""Gemini API wrapper for LTX2.3 Frames Prompt generation."""

import time
import random

import torch
import numpy as np
from PIL import Image
import io

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class FramePrompt(BaseModel):
    duration_seconds: float = Field(description="Suggested video duration in seconds, typically 2-8")
    prompt_cn: str = Field(description="Detailed Chinese prompt for LTX2.3 video generation")
    prompt_en: str = Field(description="Equivalent English prompt for LTX2.3 video generation")


class FramePromptList(BaseModel):
    frames: list[FramePrompt] = Field(description="List of prompts, one per adjacent frame pair")


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def _tensor_to_image_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a ComfyUI IMAGE tensor (H, W, C) float32 [0,1] to JPEG bytes."""
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a professional cinematographer and video director. "
    "Given two consecutive keyframe images and a user's creative direction, "
    "analyze the transition and describe:\n"
    "- Camera movement (pan, zoom, dolly, etc.)\n"
    "- Subject action and motion\n"
    "- Scene changes and visual effects\n\n"
    "For each pair, provide:\n"
    "1. Suggested video duration in seconds (2-8 seconds)\n"
    "2. A detailed Chinese prompt suitable for LTX2.3 video generation\n"
    "3. An equivalent English prompt for LTX2.3 video generation"
)


def build_contents(
    images: list[torch.Tensor],
    prompt_format: str,
    user_text: str,
) -> list:
    """Build the Gemini API contents list.

    Sends all images in one call with clear pair-boundary markers so the model
    can return one FramePrompt per adjacent pair.
    """
    parts: list = []

    # System-level context from the formatted prompt
    if prompt_format.strip():
        parts.append(f"[System Context]\n{prompt_format.strip()}\n")

    parts.append(SYSTEM_INSTRUCTION)

    if user_text.strip():
        parts.append(f"\n[User Creative Direction]\n{user_text.strip()}\n")

    # Number the images and declare the pairs to analyse
    pair_descriptions = []
    for i in range(len(images) - 1):
        pair_descriptions.append(f"Pair {i + 1}: Image {i + 1} -> Image {i + 2}")

    parts.append(
        f"\nAnalyze the following {len(pair_descriptions)} frame pair(s) sequentially:\n"
        + "\n".join(pair_descriptions)
    )
    parts.append(
        "\nReturn exactly one FramePrompt per pair in order, as a JSON array under 'frames'."
    )
    parts.append("\nBelow are the images in order (Image 1, Image 2, ...):")

    # Append images inline
    from google.genai import types as genai_types

    for idx, img_tensor in enumerate(images):
        img_bytes = _tensor_to_image_bytes(img_tensor)
        parts.append(f"\n--- Image {idx + 1} ---")
        parts.append(
            genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        )

    return parts


# ---------------------------------------------------------------------------
# Main API call
# ---------------------------------------------------------------------------

def generate_prompts(
    images: list[torch.Tensor],
    prompt_format: str,
    user_text: str,
    api_key: str,
    base_url: str,
    model_name: str = "gemini-3.1-pro-preview",
) -> tuple[str, str]:
    """Call Gemini to generate frame transition prompts.

    Returns (output_text, status_text) — output_text is the formatted prompts
    (or error), status_text is the process log.
    """
    log: list[str] = []

    def log_add(msg: str):
        log.append(msg)

    log_add(f"[开始] LTX2.3 Frames Prompt 生成")
    log_add(f"[输入] 图片数量: {len(images)}, 相邻帧对: {len(images) - 1}")

    if not api_key.strip():
        err = "ERROR: API key not set."
        log_add(f"[错误] {err}")
        return (err, "\n".join(log))

    log_add(f"[模型] {model_name.strip()}")
    log_add(f"[地址] {base_url.strip()}")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        err = "ERROR: google-genai package not installed. Run: pip install google-genai"
        log_add(f"[错误] {err}")
        return (err, "\n".join(log))

    log_add("[连接] 正在创建 API 客户端...")
    client = genai.Client(
        api_key=api_key.strip(),
        http_options={
            "base_url": base_url.strip(),
            "api_version": "",
            "timeout": 360000,  # 6 minutes in milliseconds
        },
    )

    log_add("[构建] 正在组织提示词和图片...")
    contents = build_contents(images, prompt_format, user_text)

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FramePromptList,
        temperature=0.4,
        max_output_tokens=4096,
    )

    # Retry with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        attempt_num = attempt + 1
        if attempt > 0:
            log_add(f"[重试] 第 {attempt_num}/{max_retries} 次尝试...")
        else:
            log_add(f"[请求] 正在调用 Gemini API (超时: 360s)...")
        try:
            response = client.models.generate_content(
                model=model_name.strip(),
                contents=contents,
                config=config,
            )

            log_add("[响应] API 调用成功，正在解析结果...")
            parsed: FramePromptList = response.parsed  # type: ignore[assignment]
            if parsed is None:
                err = f"ERROR: Model returned empty or unparseable response.\nRaw: {getattr(response, 'text', 'N/A')}"
                log_add(f"[错误] {err}")
                return (err, "\n".join(log))

            output = _format_output(parsed, len(images))
            log_add(f"[完成] 成功生成 {len(parsed.frames)} 个镜头提示词")
            return (output, "\n".join(log))

        except Exception as exc:
            log_add(f"[异常] {type(exc).__name__}: {exc}")
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            err = f"ERROR: API call failed after {max_retries} attempts.\n{type(exc).__name__}: {exc}"
            log_add(f"[失败] {err}")
            return (err, "\n".join(log))

    err = "ERROR: Unexpected error."
    log_add(f"[失败] {err}")
    return (err, "\n".join(log))


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_output(result: FramePromptList, image_count: int) -> str:
    """Format structured response into the required display format."""
    lines: list[str] = []
    for i, fp in enumerate(result.frames):
        lines.append(f"{i + 1}. 镜头{i + 1}（图片{i + 1}-图片{i + 2}）时长：{fp.duration_seconds}s")
        lines.append(f"   [中文] {fp.prompt_cn}")
        lines.append(f"   [EN]   {fp.prompt_en}")
        lines.append("")
    return "\n".join(lines).strip()
