# Normalized Attention Guidance (NAG) for Stable Diffusion WebUI Forge
#
# Based on:
# - Paper: "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models"
#   https://arxiv.org/abs/2505.21179
# - ComfyUI implementation: https://github.com/pamparamm/sd-perturbed-attention
#
# NAG normalizes attention weights to create a flattened/uniform attention distribution
# that serves as negative guidance.

import gradio as gr
import torch
import torch.nn.functional as F

from modules import scripts
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from backend.patcher.base import set_model_options_patch_replace
from backend.sampling.sampling_function import calc_cond_uncond_batch
from modules.ui_components import InputAccordion


def create_nag_attention_processor(norm_type: str, temperature: float = 1.0):
    """Create attention processor that normalizes attention weights."""
    
    def nag_attention(q, k, v, extra_options):
        head_dim = q.shape[-1]
        
        scores = torch.bmm(q, k.transpose(-2, -1))
        scores = scores / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply normalization based on type
        if norm_type == "L1":
            normalized_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        elif norm_type == "L2":
            normalized_weights = F.normalize(attn_weights, p=2, dim=-1)
        elif norm_type == "softmax_temp":
            normalized_weights = F.softmax(scores / temperature, dim=-1)
        else:
            normalized_weights = attn_weights
        
        output = torch.bmm(normalized_weights, v)
        return output
    
    return nag_attention


class NAGImproved(scripts.Script):
    sorting_priority = 16

    def title(self):
        return "NAG - Normalized Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown("💡 **Tip**: NAG creates more uniform attention. Good for reducing artifacts.")
            
            with gr.Row():
                # Both are equally important for NAG
                scale = gr.Slider(
                    label='Guidance Scale',
                    minimum=1.0,
                    maximum=8.0,
                    step=0.5,
                    value=3.0,
                    info="Strength of guidance. 2.0-4.0 typical."
                )
                
                norm_type = gr.Radio(
                    label='Normalization Type',
                    choices=['L2', 'L1', 'softmax_temp'],
                    value='L2',
                    info="L2 (recommended) | L1 (simple) | softmax_temp (advanced)"
                )
            
            with gr.Row():
                temperature = gr.Slider(
                    label='Temperature (softmax_temp mode)',
                    minimum=0.5,
                    maximum=3.0,
                    step=0.1,
                    value=1.5,
                    visible=False,
                    info="Higher = more uniform. 1.5-2.0 recommended."
                )

            with gr.Row():
                skip_hires = gr.Checkbox(
                    label='Skip HiRes Fix',
                    value=False,
                    visible=not is_img2img
                )
            
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    # Middle is best for NAG, rarely needs changing
                    unet_block = gr.Radio(
                        label='U-Net Block',
                        choices=['middle', 'output', 'input'],
                        value='middle',
                        info="Middle recommended for NAG"
                    )
                    unet_block_id = gr.Slider(
                        label='Block ID',
                        minimum=0,
                        maximum=5,
                        step=1,
                        value=0
                    )
                
                with gr.Row():
                    start_step = gr.Slider(
                        label='Start Step',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.0
                    )
                    end_step = gr.Slider(
                        label='End Step',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0
                    )
                            
            with gr.Accordion("Guide", open=False):
                gr.Markdown("""
                **Standard (Recommended)**:
                - Norm: L2, Scale: 3.0
                - Best balance of effectiveness and safety
                
                **Subtle Correction**:
                - Norm: L2, Scale: 2.0
                - Gentle attention balancing
                
                **Strong Effect**:
                - Norm: L2, Scale: 4.5
                - More dramatic attention flattening
                
                **Temperature-Based (Advanced)**:
                - Norm: softmax_temp, Temp: 1.5, Scale: 3.0
                - More control over uniformity
                - Higher temp = more uniform attention
                
                **How NAG Works**:
                - L1: Simple sum normalization (faster)
                - L2: Euclidean normalization (recommended, balanced)
                - softmax_temp: Temperature-scaled softmax (most flexible)
                
                **When to Use NAG**:
                - Attention artifacts or over-focus on certain regions
                - Alternative to PAG/SEG that's simpler to tune
                - Want more democratic token importance
                """)
            
            # Show temperature slider only for softmax_temp mode
            norm_type.change(
                fn=lambda x: gr.update(visible=(x == "softmax_temp")),
                inputs=[norm_type],
                outputs=[temperature],
            )

        self.infotext_fields = [
            (enabled, lambda d: d.get("nag_enabled", False)),
            (scale, "nag_scale"),
            (norm_type, "nag_norm_type"),
            (temperature, "nag_temperature"),
            (unet_block, "nag_block"),
            (unet_block_id, "nag_block_id"),
            (start_step, "nag_start_step"),
            (end_step, "nag_end_step"),
            (skip_hires, "nag_skip_hires"),
        ]

        return enabled, scale, norm_type, temperature, unet_block, unet_block_id, start_step, end_step, skip_hires

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, scale, norm_type, temperature, unet_block, unet_block_id, start_step, end_step, skip_hires = script_args

        if not enabled or (getattr(p, 'is_hr_pass', False) and skip_hires):
            return

        temp_info = f", temp={temperature:.2f}" if norm_type == "softmax_temp" else ""
        print(f"[NAG] scale={scale:.2f}, norm={norm_type}{temp_info}, block={unet_block}[{int(unet_block_id)}]")

        NAGImproved.scale = scale
        NAGImproved.norm_type = norm_type
        NAGImproved.temperature = temperature
        NAGImproved.unet_block = unet_block
        NAGImproved.unet_block_id = int(unet_block_id)
        NAGImproved.start_step = start_step
        NAGImproved.end_step = end_step
        NAGImproved.do_nag = True

        def denoiser_callback(params):
            current_step = params.sampling_step / (params.total_sampling_steps - 1)
            NAGImproved.do_nag = (
                current_step >= NAGImproved.start_step and
                current_step <= NAGImproved.end_step
            )

        on_cfg_denoiser(denoiser_callback)

        unet = p.sd_model.forge_objects.unet.clone()
        nag_attn = create_nag_attention_processor(norm_type, temperature)

        def post_cfg_function(args):
            denoised = args["denoised"]

            if NAGImproved.scale <= 0.0 or not NAGImproved.do_nag:
                return denoised

            model = args["model"]
            cond_denoised = args["cond_denoised"]
            cond = args["cond"]
            sigma = args["sigma"]
            x = args["input"]
            options = args["model_options"].copy()

            new_options = set_model_options_patch_replace(
                options, nag_attn, "attn1",
                NAGImproved.unet_block, NAGImproved.unet_block_id
            )

            nag_cond_denoised, _ = calc_cond_uncond_batch(model, cond, None, x, sigma, new_options)
            result = denoised + (cond_denoised - nag_cond_denoised) * NAGImproved.scale

            return result

        unet.set_model_sampler_post_cfg_function(post_cfg_function)
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            nag_enabled=enabled,
            nag_scale=scale,
            nag_norm_type=norm_type,
            nag_temperature=temperature if norm_type == "softmax_temp" else None,
            nag_block=unet_block,
            nag_block_id=int(unet_block_id),
            nag_start_step=start_step,
            nag_end_step=end_step,
        ))

    def postprocess(self, p, processed, *args):
        remove_current_script_callbacks()