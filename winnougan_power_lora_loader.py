import folder_paths
from typing import Union
from nodes import LoraLoader

NODE_NAME = 'Winnougan Power Lora Loader'


def get_lora_by_filename(filename, log_node=None):
    """Get a lora file by its filename from folder_paths."""
    if not filename or filename == "None":
        return None
    loras = folder_paths.get_filename_list("loras")
    for lora in loras:
        if lora == filename or lora.endswith(filename):
            return lora
    if log_node:
        print(f"[{log_node}] WARNING: Could not find lora: {filename}")
    return None


class FlexibleOptionalInputType(dict):
    """A dict subclass that allows any key as a valid optional input."""
    def __init__(self, type, data=None):
        super().__init__(data or {})
        self.type = type

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        return (self.type,)


any_type = "*"


class WinnouganPowerLoraLoader:
    """
    The Winnougan Power Lora Loader is a powerful, flexible node
    to add multiple LoRAs to a model/clip in a single compact node.

    Each lora row in the UI corresponds to a kwarg named lora_N.
    That kwarg can be either:
      - A dict  (the normal widget value: {on, lora, strength, ...})
      - A str   (a filename piped in via a node connection)
    When a string is received the row's widget strength values still apply,
    so the node reads them from the companion lora_N_strength / lora_N_on
    kwargs if present, or falls back to sensible defaults.
    """

    NAME = NODE_NAME
    CATEGORY = "Winnougan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(type=any_type, data={
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"

    def load_loras(self, model=None, clip=None, **kwargs):
        """Loops over the provided loras in kwargs and applies valid ones.

        Each lora slot can arrive as:
          • A dict  – the standard widget payload  {on, lora, strength, [strengthTwo]}
          • A str   – a lora filename wired in from another node; the row is
                      treated as enabled with strength 1.0 (both model & clip)
                      unless the widget dict is also present in kwargs under the
                      same key (which won't happen simultaneously, but is safe).
        """
        for key, value in kwargs.items():
            key_upper = key.upper()
            if not key_upper.startswith('LORA_'):
                continue

            # ── Dict payload (normal widget path) ─────────────────────────────
            if isinstance(value, dict):
                if not ('on' in value and 'lora' in value and 'strength' in value):
                    continue

                strength_model = value['strength']
                strength_clip  = value.get('strengthTwo', None)

                if clip is None:
                    if strength_clip is not None and strength_clip != 0:
                        print(f'[{NODE_NAME}] WARNING: Received clip strength even though no clip supplied!')
                    strength_clip = 0
                else:
                    strength_clip = strength_clip if strength_clip is not None else strength_model

                if value['on'] and (strength_model != 0 or strength_clip != 0):
                    lora = get_lora_by_filename(value['lora'], log_node=NODE_NAME)
                    if model is not None and lora is not None:
                        model, clip = LoraLoader().load_lora(
                            model, clip, lora, strength_model, strength_clip
                        )

            # ── String payload (wired-in filename from a subgraph connection) ─
            elif isinstance(value, str) and value and value != "None":
                lora = get_lora_by_filename(value, log_node=NODE_NAME)
                if model is not None and lora is not None:
                    # Default to strength 1.0 for both when wired externally.
                    # If the user also has a strength companion kwarg in future
                    # widget designs, it could be read here.
                    strength_model = 1.0
                    strength_clip  = 0.0 if clip is None else 1.0
                    model, clip = LoraLoader().load_lora(
                        model, clip, lora, strength_model, strength_clip
                    )

        return (model, clip)

    @classmethod
    def get_enabled_loras_from_prompt_node(cls, prompt_node: dict) -> list[dict[str, Union[str, float]]]:
        """Gets enabled loras of a node within a server prompt."""
        result = []
        for name, lora in prompt_node['inputs'].items():
            if name.startswith('lora_') and isinstance(lora, dict) and lora.get('on'):
                lora_file = get_lora_by_filename(lora['lora'], log_node=cls.NAME)
                if lora_file is not None:
                    lora_dict = {
                        'name': lora['lora'],
                        'strength': lora['strength'],
                        'path': folder_paths.get_full_path("loras", lora_file)
                    }
                    if 'strengthTwo' in lora:
                        lora_dict['strength_clip'] = lora['strengthTwo']
                    result.append(lora_dict)
        return result


# ── ComfyUI registration ──────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "WinnouganPowerLoraLoader": WinnouganPowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WinnouganPowerLoraLoader": "Winnougan Power Lora Loader",
}
