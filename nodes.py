"""LTX2.3 Frames Prompt — ComfyUI custom node."""

import re
import torch

from .gemini_client import generate_prompts


class LTX23FramesPrompt:
    """Generates Chinese/English video prompts with durations for LTX2.3
    multi-frame video generation, powered by Gemini API.

    Input images are analysed in adjacent pairs: (img1,img2), (img2,img3), ...
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image_1": ("IMAGE",),
            "prompt_format": ("STRING", {"multiline": True, "default": ""}),
            "user_text": ("STRING", {"multiline": True, "default": ""}),
            "api_key": ("STRING", {"default": ""}),
            "base_url": ("STRING", {"default": "https://ai.t8star.org"}),
        }
        optional = {}
        for i in range(2, 17):
            optional[f"image_{i}"] = ("IMAGE",)
        optional["model_name"] = ("STRING", {"default": "gemini-3.1-pro-preview"})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompts", "status")
    FUNCTION = "generate"
    CATEGORY = "LTX2.3"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Generate Chinese & English video prompts with suggested durations "
        "for LTX2.3 multi-frame generation. Analyses adjacent image pairs "
        "via Gemini API. Connect 2-16 images, formatted prompt, and user text. "
        "Connect outputs to ShowText nodes to view results."
    )

    def generate(self, **kwargs):
        try:
            return self._generate(**kwargs)
        except Exception as exc:
            import traceback
            err = f"ERROR: Node execution failed.\n{type(exc).__name__}: {exc}"
            status = traceback.format_exc()
            return {"ui": {"prompts": [err], "status": [status]}, "result": (err, status)}

    def _generate(self, **kwargs):
        # Collect all connected image inputs in order
        images: list[torch.Tensor] = []
        image_keys = sorted(
            [k for k in kwargs if re.match(r"^image_\d+$", k)],
            key=lambda k: int(re.search(r"\d+", k).group()),
        )
        for key in image_keys:
            val = kwargs[key]
            if val is None:
                continue
            # Each IMAGE input from ComfyUI is (B, H, W, C) — take first batch item
            if isinstance(val, torch.Tensor):
                if val.ndim == 4:
                    images.append(val[0])
                elif val.ndim == 3:
                    images.append(val)
                else:
                    continue

        if len(images) < 2:
            output = f"ERROR: At least 2 images required (found {len(images)})."
            status = (
                f"[开始] LTX2.3 Frames Prompt 生成\n"
                f"[输入] 图片数量: {len(images)}, 相邻帧对: {len(images) - 1}\n"
                f"[错误] 至少需要 2 张图片 (实际连接: {len(images)})"
            )
            return {"ui": {"prompts": [output], "status": [status]}, "result": (output, status)}

        prompt_format = kwargs.get("prompt_format", "")
        user_text = kwargs.get("user_text", "")
        api_key = kwargs.get("api_key", "")
        base_url = kwargs.get("base_url", "https://ai.t8star.org")
        model_name = kwargs.get("model_name", "gemini-3.1-pro-preview")

        output, status = generate_prompts(
            images=images,
            prompt_format=prompt_format,
            user_text=user_text,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

        return {"ui": {"prompts": [output], "status": [status]}, "result": (output, status)}
