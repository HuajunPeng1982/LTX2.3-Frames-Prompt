"""LTX2.3 Frames Prompt — ComfyUI custom node."""

import re
import torch

from .gemini_client import generate_prompts


class LTX23FramesPrompt:
    """输入序列关键帧（首帧→中帧→尾帧），经由 Gemini 兼容 API 生成视频提示词。
    输出 1 个全局提示词（不变的主体/场景/风格）+ N 个局部提示词（每张参考图一个，描述变化）。
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompts", "status", "prompts_cn", "prompts_en", "global_cn", "global_en")
    FUNCTION = "generate"
    CATEGORY = "LTX2.3"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "输入序列关键帧（首帧→中帧→尾帧），大模型根据用户提示词和结构化提示词，"
        "输出1个全局提示词（描述不变的主体/场景/风格，50-100词）和若干局部提示词（每张图一个，描述变化）。"
        "结果连接到 ShowText 节点查看。"
    )

    def generate(self, **kwargs):
        try:
            return self._generate(**kwargs)
        except Exception as exc:
            import traceback
            err = f"ERROR: Node execution failed.\n{type(exc).__name__}: {exc}"
            status = traceback.format_exc()
            return {"ui": {"prompts": [err], "status": [status], "prompts_cn": [err], "prompts_en": [err], "global_cn": [err], "global_en": [err]}, "result": (err, status, err, err, err, err)}

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
                f"[输入] 图片数量: {len(images)}\n"
                f"[错误] 至少需要 2 张图片 (实际连接: {len(images)})"
            )
            return {"ui": {"prompts": [output], "status": [status], "prompts_cn": [output], "prompts_en": [output], "global_cn": [output], "global_en": [output]}, "result": (output, status, output, output, output, output)}

        prompt_format = kwargs.get("prompt_format", "")
        user_text = kwargs.get("user_text", "")
        api_key = kwargs.get("api_key", "")
        base_url = kwargs.get("base_url", "https://ai.t8star.org")
        model_name = kwargs.get("model_name", "gemini-3.1-pro-preview")

        output, status, cn_output, en_output, global_cn, global_en = generate_prompts(
            images=images,
            prompt_format=prompt_format,
            user_text=user_text,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

        return {"ui": {"prompts": [output], "status": [status], "prompts_cn": [cn_output], "prompts_en": [en_output], "global_cn": [global_cn], "global_en": [global_en]}, "result": (output, status, cn_output, en_output, global_cn, global_en)}
