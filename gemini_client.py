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
    duration_seconds: float = Field(description="Segment display duration in seconds (分段时长), typically 2-8")
    transition_seconds: float = Field(default=0.0, description="Overlap transition time from previous frame to this frame in seconds (过渡时长). First frame always 0.0")
    prompt_cn: str = Field(description="Detailed Chinese prompt for LTX2.3 video generation")
    prompt_en: str = Field(description="Equivalent English prompt for LTX2.3 video generation")


class FramePromptList(BaseModel):
    frames: list[FramePrompt] = Field(description="List of prompts, one per input image")
    global_prompt_cn: str = Field(default="", description="Global Chinese prompt describing the entire video sequence")
    global_prompt_en: str = Field(default="", description="Global English prompt describing the entire video sequence")


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
    "You are a professional prompt engineer for the LTX2.3 video generation model.\n\n"
    "## Core Logic\n"
    "You will receive a SEQUENCE of keyframe images in temporal order "
    "(首帧→中帧→尾帧). Your task is to produce TWO types of output:\n\n"
    "### 1. Global Prompt (全局提示词) — 50-100 words\n"
    'The global prompt describes "what stays UNCHANGED" throughout the entire video. '
    "It is the foundational setting that ensures consistency across all segments:\n"
    "- Core subject & identity (gender, age, appearance, clothing, distinctive features)\n"
    "- Overall scene & environment (location, setting, spatial relationships)\n"
    "- Visual style & quality (cinematic, realistic, lighting, color tone, resolution)\n"
    "- Base camera setup (only if consistent across the whole video)\n\n"
    "The global prompt MUST be concise (50-100 words). "
    "Do NOT include any time-varying content (actions, camera movements, scene changes).\n\n"
    "### 2. Local Prompts (局部提示词) — One per input image\n"
    'Each local prompt describes "what CHANGES" during its specific time segment. '
    "Write each segment independently, but maintain awareness of its position in the sequence "
    "(how it connects from the previous frame and leads to the next). "
    "Priority order: core action > camera movement > facial expression > environment details.\n\n"
    "For each segment provide:\n"
    "- duration_seconds: segment display time (2-8s, match the action complexity)\n"
    "- transition_seconds: overlap blend time from previous segment (first=0.0, others=0.3-2.0s)\n"
    "- prompt_cn: Chinese prompt describing this segment's action and change\n"
    "- prompt_en: Equivalent English prompt\n\n"
    "## LTX2.3 Prompt Style\n"
    "- Write as a single coherent paragraph (not bullet points), 4-8 descriptive sentences merged into one.\n"
    "- Camera instruction at the START of every local prompt (固定机位, 缓慢推近, etc.)\n"
    "- Actions must be concrete and continuous: prefer 'slowly raises hand' over 'raises hand'.\n"
    "- Use long-tail keyword weighting for subjects: (30岁男性职业讲师:1.3).\n"
    "- Avoid abstract language, sudden movements ('突然', '立刻'), logos, text, or chaotic physics.\n"
    "- Chinese and English versions convey the same meaning, not a literal translation."
)


def build_prompt_text(
    image_count: int,
    prompt_format: str,
    user_text: str,
) -> str:
    """Build the full text prompt (without images — images are sent separately).

    When prompt_format is provided, it takes priority over the built-in SYSTEM_INSTRUCTION.
    """
    parts: list[str] = []

    if prompt_format.strip():
        # User-supplied structured prompt has HIGHER priority than SYSTEM_INSTRUCTION
        parts.append(f"{prompt_format.strip()}\n")
    else:
        # Fallback to built-in instruction only when no user prompt_format
        parts.append(SYSTEM_INSTRUCTION)

    if user_text.strip():
        parts.append(f"\n## User Creative Direction\n{user_text.strip()}\n")

    # Label images by their sequence position
    if image_count == 1:
        labels = ["Image 1 (唯一关键帧)"]
    elif image_count == 2:
        labels = ["Image 1 (首帧/起点)", "Image 2 (尾帧/终点)"]
    else:
        labels = ["Image 1 (首帧/起点)"]
        for i in range(1, image_count - 1):
            labels.append(f"Image {i + 1} (中帧/过程 {i})")
        labels.append(f"Image {image_count} (尾帧/终点)")

    parts.append(
        f"## Input Keyframes (temporal sequence: 首帧 → 中帧 → 尾帧)\n"
        + "\n".join(labels)
    )
    parts.append(
        "\n## Required JSON Output Format (MUST follow exactly)\n"
        "Return ONLY a JSON object. No markdown fences, no explanation, no other text.\n\n"
        "{\n"
        '  "global_prompt_cn": "全局提示词中文（50-100词，描述不变的主体/场景/风格）",\n'
        '  "global_prompt_en": "global prompt english (50-100 words, unchanging elements only)",\n'
        '  "frames": [\n'
        '    {\n'
        '      "duration_seconds": 3.0,\n'
        '      "transition_seconds": 0.0,\n'
        '      "prompt_cn": "镜头指令+核心动作+表情+环境变化（中文）",\n'
        '      "prompt_en": "camera instruction + core action + expression + environment (english)"\n'
        '    },\n'
        '    ...\n'
        '  ]\n'
        '}\n\n'
        f"Rules:\n"
        f"- Provide exactly {image_count} frame entries, one per input image.\n"
        f"- Frame 1 transition_seconds MUST be 0.0. Subsequent frames: 0.3-2.0s.\n"
        f"- duration_seconds: pure display time for this segment (2-8s, match action complexity).\n"
        f"- transition_seconds: overlap blend from previous frame to this frame.\n"
        f"- Each local prompt: start with camera instruction, then action, expression, environment.\n"
        f"- Global prompt (50-100 words): only unchanging elements — subject, scene, style.\n"
        f"- Do NOT repeat global content in local prompts.\n"
        f"- Write each prompt as a coherent paragraph, not bullet points."
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
) -> tuple[str, str, str, str, str, str]:
    """Call Gemini API to generate per-frame prompts.

    Returns (output_text, status_text, cn_text, en_text, global_cn, global_en).
    """
    log: list[str] = []

    def log_add(msg: str):
        log.append(msg)

    log_add(f"[开始] LTX2.3 Frames Prompt 生成")
    log_add(f"[输入] 图片数量: {len(images)}")

    if not api_key.strip():
        err = "ERROR: API key not set."
        log_add(f"[错误] {err}")
        return (err, "\n".join(log), "", "", "", "")

    log_add(f"[模型] {model_name.strip()}")
    log_add(f"[地址] {base_url.strip()}")

    try:
        import requests
    except ImportError:
        err = "ERROR: requests package not installed. Run: pip install requests"
        log_add(f"[错误] {err}")
        return (err, "\n".join(log), "", "", "", "")

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
                    return (raw_text, "\n".join(log), "", "", "", "")

            if "frames" not in data:
                log_add("[警告] 响应缺少frames字段，使用原始文本")
                return (raw_text, "\n".join(log), "", "", "", "")

            parsed = FramePromptList.model_validate(data)
            output = _format_output(parsed, len(images))
            cn_output = _format_cn(parsed)
            en_output = _format_en(parsed)
            global_cn = parsed.global_prompt_cn
            global_en = parsed.global_prompt_en
            log_add(f"[全局] 全局中文提示词长度: {len(global_cn)} 字符, 全局英文提示词长度: {len(global_en)} 字符")
            log_add(f"[完成] 成功生成 {len(parsed.frames)} 个局部提示词 + 1 个全局提示词")
            return (output, "\n".join(log), cn_output, en_output, global_cn, global_en)

        except Exception as exc:
            log_add(f"[异常] {type(exc).__name__}: {exc}")
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            err = f"ERROR: API call failed after {max_retries} attempts.\n{type(exc).__name__}: {exc}"
            log_add(f"[失败] {err}")
            return (err, "\n".join(log), "", "", "", "")

    err = "ERROR: Unexpected error."
    log_add(f"[失败] {err}")
    return (err, "\n".join(log), "", "", "", "")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_output(result: FramePromptList, image_count: int) -> str:
    """Format structured response into the combined display format."""
    lines: list[str] = []
    for i, fp in enumerate(result.frames):
        lines.append(f"{i + 1}. 镜头{i + 1}（图片{i + 1}）时长：{fp.duration_seconds}s")
        lines.append(f"   [中文] {fp.prompt_cn}")
        lines.append(f"   [EN]   {fp.prompt_en}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_cn(result: FramePromptList) -> str:
    """Format Chinese-only prompts for side-by-side display."""
    lines: list[str] = []
    for i, fp in enumerate(result.frames):
        lines.append(f"镜头{i + 1}（图片{i + 1}）时长：{fp.duration_seconds}s")
        lines.append(fp.prompt_cn)
        lines.append("")
    return "\n".join(lines).strip()


def _format_en(result: FramePromptList) -> str:
    """Format English prompts as a table with cumulative time ranges and transition durations."""
    lines: list[str] = []
    lines.append("| Segment | Start | End | Transition | Prompt |")
    lines.append("|---------|-------|-----|------------|--------|")
    cumulative = 0.0
    for i, fp in enumerate(result.frames):
        start = cumulative
        end = cumulative + fp.duration_seconds
        cumulative = end
        t = fp.transition_seconds
        zhuanchang = ",zhuanchang" if i < len(result.frames) - 1 else ""
        lines.append(f"| {i + 1} | {start:.1f} | {end:.1f} | {t:.1f} | {fp.prompt_en}{zhuanchang} |")
    return "\n".join(lines)
