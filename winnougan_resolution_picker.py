import torch

NODE_NAME = "Winnougan Resolution Picker"


class WinnouganResolutionPicker:

    NAME = NODE_NAME
    CATEGORY = "Winnougan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "width":      ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                                       "tooltip": "Override width (used when connected via subgraph)"}),
                "height":     ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                                       "tooltip": "Override height (used when connected via subgraph)"}),
                "batch_size": ("INT", {"default": 1,    "min": 1,  "max": 64,   "step": 1,
                                       "tooltip": "Number of latents to generate"}),
            },
            "hidden": {
                "_width":      ("INT", {"default": 1024}),
                "_height":     ("INT", {"default": 1024}),
                "_batch_size": ("INT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "LATENT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "LATENT")
    FUNCTION = "pick_resolution"

    def pick_resolution(
        self,
        width=None,  height=None,  batch_size=None,
        _width=1024, _height=1024, _batch_size=1,
    ):
        # Prefer explicit inputs (subgraph connections) over the UI-driven hidden values
        w = width      if width      is not None else _width
        h = height     if height     is not None else _height
        b = batch_size if batch_size is not None else _batch_size

        latent = torch.zeros(
            [b, 16, h // 8, w // 8],
            dtype=torch.float32,
        )
        return (w, h, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "WinnouganResolutionPicker": WinnouganResolutionPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WinnouganResolutionPicker": "Winnougan Resolution Picker",
}
