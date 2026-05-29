"""LTX2.3 Frames Prompt — ComfyUI custom node for video prompt generation."""

from .nodes import LTX23FramesPrompt

NODE_CLASS_MAPPINGS = {
    "LTX23FramesPrompt": LTX23FramesPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23FramesPrompt": "LTX2.3 Frames Prompt(Grsai)",
}

