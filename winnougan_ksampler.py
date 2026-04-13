"""
Winnougan KSampler
──────────────────
A compact sampler node exposing all ClownSampler / RES4LYF samplers
plus the standard ComfyUI schedulers, with a bongmath toggle.
"""

import logging
import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_management
import comfy.utils
import latent_preview

log = logging.getLogger("Winnougan")

NODE_NAME = "Winnougan KSampler"

# ── Full clown sampler list (from RES4LYF) ────────────────────────────────────
CLOWN_SAMPLERS = [
    "none",
    # multistep
    "multistep/res_2m", "multistep/res_3m", "multistep/dpmpp_2m", "multistep/dpmpp_3m",
    "multistep/abnorsett_2m", "multistep/abnorsett_3m", "multistep/abnorsett_4m",
    "multistep/deis_2m", "multistep/deis_3m", "multistep/deis_4m",
    # exponential
    "exponential/res_2s_rkmk2e", "exponential/res_2s", "exponential/res_2s_stable",
    "exponential/res_3s", "exponential/res_3s_non-monotonic", "exponential/res_3s_alt",
    "exponential/res_3s_cox_matthews", "exponential/res_3s_lie", "exponential/res_3s_sunstar",
    "exponential/res_3s_strehmel_weiner", "exponential/res_4s_krogstad",
    "exponential/res_4s_krogstad_alt", "exponential/res_4s_strehmel_weiner",
    "exponential/res_4s_strehmel_weiner_alt", "exponential/res_4s_cox_matthews",
    "exponential/res_4s_cfree4", "exponential/res_4s_friedli", "exponential/res_4s_minchev",
    "exponential/res_4s_munthe-kaas", "exponential/res_5s", "exponential/res_5s_hochbruck-ostermann",
    "exponential/res_6s", "exponential/res_8s", "exponential/res_8s_alt",
    "exponential/res_10s", "exponential/res_15s", "exponential/res_16s",
    "exponential/etdrk2_2s", "exponential/etdrk3_a_3s", "exponential/etdrk3_b_3s",
    "exponential/etdrk4_4s", "exponential/etdrk4_4s_alt",
    "exponential/dpmpp_2s", "exponential/dpmpp_sde_2s", "exponential/dpmpp_3s",
    "exponential/lawson2a_2s", "exponential/lawson2b_2s", "exponential/lawson4_4s",
    "exponential/lawson41-gen_4s", "exponential/lawson41-gen-mod_4s", "exponential/ddim",
    # hybrid
    "hybrid/pec423_2h2s", "hybrid/pec433_2h3s", "hybrid/abnorsett2_1h2s",
    "hybrid/abnorsett3_2h2s", "hybrid/abnorsett4_3h2s", "hybrid/lawson42-gen-mod_1h4s",
    "hybrid/lawson43-gen-mod_2h4s", "hybrid/lawson44-gen-mod_3h4s", "hybrid/lawson45-gen-mod_4h4s",
    # linear
    "linear/ralston_2s", "linear/ralston_3s", "linear/ralston_4s",
    "linear/midpoint_2s", "linear/heun_2s", "linear/heun_3s",
    "linear/houwen-wray_3s", "linear/kutta_3s", "linear/ssprk3_3s", "linear/ssprk4_4s",
    "linear/rk38_4s", "linear/rk4_4s", "linear/rk5_7s", "linear/rk6_7s",
    "linear/bogacki-shampine_4s", "linear/bogacki-shampine_7s",
    "linear/dormand-prince_6s", "linear/dormand-prince_13s",
    "linear/tsi_7s", "linear/euler",
    # diag implicit
    "diag_implicit/irk_exp_diag_2s", "diag_implicit/kraaijevanger_spijker_2s",
    "diag_implicit/qin_zhang_2s", "diag_implicit/pareschi_russo_2s",
    "diag_implicit/pareschi_russo_alt_2s", "diag_implicit/crouzeix_2s",
    "diag_implicit/crouzeix_3s", "diag_implicit/crouzeix_3s_alt",
    # fully implicit
    "fully_implicit/gauss-legendre_2s", "fully_implicit/gauss-legendre_3s",
    "fully_implicit/gauss-legendre_4s", "fully_implicit/gauss-legendre_4s_alternating_a",
    "fully_implicit/gauss-legendre_4s_ascending_a", "fully_implicit/gauss-legendre_4s_alt",
    "fully_implicit/gauss-legendre_5s", "fully_implicit/gauss-legendre_5s_ascending",
    "fully_implicit/radau_ia_2s", "fully_implicit/radau_ia_3s",
    "fully_implicit/radau_iia_2s", "fully_implicit/radau_iia_3s", "fully_implicit/radau_iia_3s_alt",
    "fully_implicit/radau_iia_5s", "fully_implicit/radau_iia_7s", "fully_implicit/radau_iia_9s",
    "fully_implicit/radau_iia_11s",
    "fully_implicit/lobatto_iiia_2s", "fully_implicit/lobatto_iiia_3s", "fully_implicit/lobatto_iiia_4s",
    "fully_implicit/lobatto_iiib_2s", "fully_implicit/lobatto_iiib_3s", "fully_implicit/lobatto_iiib_4s",
    "fully_implicit/lobatto_iiic_2s", "fully_implicit/lobatto_iiic_3s", "fully_implicit/lobatto_iiic_4s",
    "fully_implicit/lobatto_iiic_star_2s", "fully_implicit/lobatto_iiic_star_3s",
    "fully_implicit/lobatto_iiid_2s", "fully_implicit/lobatto_iiid_3s",
]

def _get_schedulers():
    scheds = list(comfy.samplers.KSampler.SCHEDULERS)
    for extra in ("beta57", "beta", "linear_quadratic"):
        if extra not in scheds:
            scheds.append(extra)
    return scheds


class WinnouganKSampler:
    NAME     = NODE_NAME
    CATEGORY = "Winnougan"

    @classmethod
    def INPUT_TYPES(cls):
        std_samplers = comfy.samplers.KSampler.SAMPLERS

        return {
            "required": {
                "model":          ("MODEL",),
                "positive":       ("CONDITIONING",),
                "negative":       ("CONDITIONING",),
                "latent_image":   ("LATENT",),
                "seed":           ("INT",    {"default": 0,    "min": 0,    "max": 0xffffffffffffffff}),
                "steps":          ("INT",    {"default": 20,   "min": 1,    "max": 10000}),
                "cfg":            ("FLOAT",  {"default": 7.0,  "min": 0.0,  "max": 100.0, "step": 0.1}),
                "sampler":        (std_samplers,),
                "clown_sampler":  (CLOWN_SAMPLERS, {"default": "none",
                                   "tooltip": "ClownSampler / RES4LYF sampler. Overrides 'sampler' when not 'none'."}),
                "scheduler":      (_get_schedulers(),),
                "denoise":        ("FLOAT",  {"default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                # ── bongmath ──────────────────────────────────────────────
                "bongmath":       ("BOOLEAN", {
                    "default": False,
                    "label_on":  "bongmath on",
                    "label_off": "bongmath off",
                    "tooltip": (
                        "Enable RES4LYF 'bongmath' high-precision denoising. "
                        "Passes extra numerical parameters to the sampler for "
                        "improved accuracy on Flux / DiT models."
                    ),
                }),
                "bongmath_cfg_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "CFG scale used inside bongmath (independent of the outer cfg).",
                }),
                "bongmath_scale":  ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Bongmath noise scale / step multiplier.",
                }),
            },
            "optional": {
                "sigmas":         ("SIGMAS", {"tooltip": "Override the scheduler with a custom sigma schedule."}),
                "options":        ("OPTIONS", {"tooltip": "Extra ClownSampler options block (RES4LYF OPTIONS type)."}),
            },
        }

    RETURN_TYPES  = ("LATENT", "LATENT")
    RETURN_NAMES  = ("latent", "denoised_output")
    FUNCTION      = "sample"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler,
        clown_sampler,
        scheduler,
        denoise,
        bongmath,
        bongmath_cfg_scale,
        bongmath_scale,
        sigmas=None,
        options=None,
    ):
        # Determine effective sampler name
        sampler_name = clown_sampler if clown_sampler != "none" else sampler

        # Build extra model options for bongmath
        model_options = {}
        if bongmath:
            model_options["transformer_options"] = model_options.get("transformer_options", {})
            model_options["transformer_options"]["bongmath"]           = True
            model_options["transformer_options"]["bongmath_cfg_scale"] = bongmath_cfg_scale
            model_options["transformer_options"]["bongmath_scale"]     = bongmath_scale
            log.info(
                f"[{NODE_NAME}] bongmath enabled — "
                f"cfg_scale={bongmath_cfg_scale}, scale={bongmath_scale}"
            )

        # Patch model options if needed
        if model_options:
            model = model.clone()
            for k, v in model_options.items():
                if k == "transformer_options":
                    model.model_options.setdefault("transformer_options", {}).update(v)
                else:
                    model.model_options[k] = v

        # If options block supplied (RES4LYF OPTIONS), apply it
        if options is not None:
            try:
                model = options.apply(model)
            except Exception as e:
                log.warning(f"[{NODE_NAME}] Could not apply OPTIONS block: {e}")

        # Build latent / noise mask — mirror official common_ksampler exactly
        latent        = latent_image
        latent_image_ = latent["samples"]

        # fix_empty_latent_channels on INPUT is required for Cosmos/Anima/WAN:
        # these models store downscale_ratio_spacial in the latent dict and use
        # it to set up internal state. Without this the IndexError occurs.
        latent_image_ = comfy.sample.fix_empty_latent_channels(
            model, latent_image_, latent.get("downscale_ratio_spacial", None)
        )

        batch_inds   = latent.get("batch_index", None)
        noise        = comfy.sample.prepare_noise(latent_image_, seed, batch_inds)
        noise_mask   = latent.get("noise_mask", None)
        callback     = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Resolve clown sampler paths; standard names passed as strings to
        # comfy.sample.sample() which accepts both strings and sampler objects.
        if "/" in sampler_name:
            try:
                from custom_nodes.RES4LYF.beta.samplers import get_clown_sampler
                sampler_obj = get_clown_sampler(sampler_name)
            except Exception:
                sampler_obj = comfy.samplers.sampler_object(sampler_name)
        else:
            sampler_obj = sampler_name   # comfy.sample.sample accepts the string

        # Use comfy.sample.sample (high-level, identical to official KSampler)
        # so that model-specific setup (Anima, Cosmos, WAN…) runs correctly.
        # Custom sigmas: if provided, we convert denoise to match step count.
        if sigmas is not None:
            # With custom sigmas we use the low-level path so we can pass them in
            sampler_obj_resolved = (
                sampler_obj if not isinstance(sampler_obj, str)
                else comfy.samplers.sampler_object(sampler_obj)
            )
            samples = comfy.samplers.sample(
                model,
                noise,
                positive,
                negative,
                cfg,
                comfy.model_management.get_torch_device(),
                sampler_obj_resolved,
                sigmas,
                model_options = model.model_options,
                latent_image  = latent_image_,
                denoise_mask  = noise_mask,
                disable_pbar  = disable_pbar,
                seed          = seed,
            )
        else:
            samples = comfy.sample.sample(
                model,
                noise,
                steps,
                cfg,
                sampler_obj,
                scheduler,
                positive,
                negative,
                latent_image_,
                denoise            = denoise,
                disable_noise      = False,
                start_step         = None,
                last_step          = None,
                force_full_denoise = True,
                noise_mask         = noise_mask,
                callback           = callback,
                disable_pbar       = disable_pbar,
                seed               = seed,
            )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples

        out_denoised = latent.copy()
        out_denoised.pop("downscale_ratio_spacial", None)
        out_denoised["samples"] = comfy.sample.fix_empty_latent_channels(model, samples)

        return (out, out_denoised)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WinnouganKSampler": WinnouganKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WinnouganKSampler": "Winnougan KSampler",
}
