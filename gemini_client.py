"""Gemini API wrapper for LTX2.3 Frames Prompt generation."""

import time
import random
import json
import re as _re
import base64
import io

import torch
import numpy as np
from PIL import Image

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

def _tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a ComfyUI IMAGE tensor (H, W, C) float32 [0,1] to base64 JPEG."""
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


def build_prompt_text(
    image_count: int,
    prompt_format: str,
    user_text: str,
) -> str:
    """Build the full text prompt (without images — images are sent separately)."""
    parts: list[str] = []

    if prompt_format.strip():
        parts.append(f"[System Context]\n{prompt_format.strip()}\n")

    parts.append(SYSTEM_INSTRUCTION)

    if user_text.strip():
        parts.append(f"\n[User Creative Direction]\n{user_text.strip()}\n")

    pair_descriptions = []
    for i in range(image_count - 1):
        pair_descriptions.append(f"Pair {i + 1}: Image {i + 1} -> Image {i + 2}")

    parts.append(
        f"\nAnalyze the following {len(pair_descriptions)} frame pair(s) sequentially:\n"
        + "\n".join(pair_descriptions)
    )
    parts.append(
        "\nReturn your response as a JSON object with this exact structure, no other text:\n"
        '{"frames": [{"duration_seconds": 4.0, "prompt_cn": "...", "prompt_en": "..."}]}\n'
        "Important: return ONLY the JSON, no markdown fences, no explanation."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main API call (using requests directly, not google-genai SDK)
# ---------------------------------------------------------------------------

def generate_prompts(
    images: list[torch.Tensor],
    prompt_format: str,
    user_text: str,
    api_key: str,
    base_url: str,
    model_name: str = "gemini-3.1-pro-preview",
) -> tuple[str, str, str, str]:
    """Call Gemini API to generate frame transition prompts.

    Returns (output_text, status_text, cn_text, en_text).
    """
    log: list[str] = []

    def log_add(msg: str):
        log.append(msg)

    log_add(f"[开始] LTX2.3 Frames Prompt 生成")
    log_add(f"[输入] 图片数量: {len(images)}, 相邻帧对: {len(images) - 1}")

    if not api_key.strip():
        err = "ERROR: API key not set."
        log_add(f"[错误] {err}")
        return (err, "\n".join(log), "", "")

    log_add(f"[模型] {model_name.strip()}")
    log_add(f"[地址] {base_url.strip()}")

    try:
        import requests
    except ImportError:
        err = "ERROR: requests package not installed. Run: pip install requests"
        log_add(f"[错误] {err}")
        return (err, "\n".join(log), "", "")

    log_add("[构建] 正在组织提示词和图片...")
    text_prompt = build_prompt_text(len(images), prompt_format, user_text)

    # Build OpenAI-compatible content array: text + images as data URIs
    content: list[dict] = [{"type": "text", "text": text_prompt}]
    for idx, img_tensor in enumerate(images):
        b64 = _tensor_to_base64(img_tensor)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    request_body = {
        "model": model_name.strip(),
        "messages": [
            {"role": "user", "content": content},
        ],
        "temperature": 0.4,
        "max_tokens": 4096,
    }

    base = base_url.strip().rstrip("/")
    endpoint = f"{base}/v1/chat/completions"

    log_add(f"[请求] 正在调用 API (超时: 360s, 端点: {endpoint})...")

    max_retries = 5
    for attempt in range(max_retries):
        attempt_num = attempt + 1
        if attempt > 0:
            log_add(f"[重试] 第 {attempt_num}/{max_retries} 次尝试...")

        try:
            resp = requests.post(
                endpoint,
                json=request_body,
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type": "application/json",
                },
                timeout=360,
                proxies={"http": None, "https": None},  # bypass system proxy
            )

            log_add(f"[响应] HTTP {resp.status_code}, 长度: {len(resp.text)} 字符")

            if resp.status_code == 429:
                log_add("[过载] 服务器限流(429)，等待更长时间后重试...")
                delay = 10 + random.uniform(0, 5)
                time.sleep(delay)
                continue

            if resp.status_code != 200:
                log_add(f"[调试] 响应内容前300字符: {resp.text[:300]}")
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            log_add(f"[调试] 响应JSON keys: {list(data.keys())}")

            # Extract the assistant's reply
            raw_text = ""
            if "choices" in data and len(data["choices"]) > 0:
                raw_text = data["choices"][0].get("message", {}).get("content", "")
            elif "candidates" in data:  # Gemini native format fallback
                raw_text = data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")

            log_add(f"[调试] 提取文本长度: {len(raw_text)}, 前200字符: {raw_text[:200]}")

            if not raw_text.strip():
                raise RuntimeError("Model returned empty content.")

            # Parse JSON from the response text
            clean = raw_text.strip()
            clean = _re.sub(r"^```(?:json)?\s*\n?", "", clean)
            clean = _re.sub(r"\n?```\s*$", "", clean)

            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                match = _re.search(r'\{[\s\S]*"frames"[\s\S]*\}', clean)
                if match:
                    data = json.loads(match.group())
                else:
                    log_add("[警告] 无法解析JSON，使用原始文本输出")
                    return (raw_text, "\n".join(log), "", "")

            if "frames" not in data:
                log_add("[警告] 响应缺少frames字段，使用原始文本")
                return (raw_text, "\n".join(log), "", "")

            parsed = FramePromptList.model_validate(data)
            output = _format_output(parsed, len(images))
            cn_output = _format_cn(parsed)
            en_output = _format_en(parsed)
            log_add(f"[完成] 成功生成 {len(parsed.frames)} 个镜头提示词")
            return (output, "\n".join(log), cn_output, en_output)

        except Exception as exc:
            log_add(f"[异常] {type(exc).__name__}: {exc}")
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            err = f"ERROR: API call failed after {max_retries} attempts.\n{type(exc).__name__}: {exc}"
            log_add(f"[失败] {err}")
            return (err, "\n".join(log), "", "")

    err = "ERROR: Unexpected error."
    log_add(f"[失败] {err}")
    return (err, "\n".join(log), "", "")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_output(result: FramePromptList, image_count: int) -> str:
    """Format structured response into the combined display format."""
    lines: list[str] = []
    for i, fp in enumerate(result.frames):
        lines.append(f"{i + 1}. 镜头{i + 1}（图片{i + 1}-图片{i + 2}）时长：{fp.duration_seconds}s")
        lines.append(f"   [中文] {fp.prompt_cn}")
        lines.append(f"   [EN]   {fp.prompt_en}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_cn(result: FramePromptList) -> str:
    """Format Chinese-only prompts for side-by-side display."""
    lines: list[str] = []
    for i, fp in enumerate(result.frames):
        lines.append(f"镜头{i + 1}（图片{i + 1}-图片{i + 2}）时长：{fp.duration_seconds}s")
        lines.append(fp.prompt_cn)
        lines.append("")
    return "\n".join(lines).strip()


def _format_en(result: FramePromptList) -> str:
    """Format English-only prompts with cumulative time ranges."""
    lines: list[str] = []
    cumulative = 0.0
    for i, fp in enumerate(result.frames):
        start = cumulative
        end = cumulative + fp.duration_seconds
        cumulative = end
        start_str = f"{start:g}"
        end_str = f"{end:g}"
        line = f"{start_str}~{end_str}s(Shot {i + 1}):{fp.prompt_en}"
        if i == len(result.frames) - 1:
            line += "zhuanchang,"
        lines.append(line)
    return "\n".join(lines)
