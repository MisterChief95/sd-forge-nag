# NAG (Normalized Attention Guidance)

Implements NAG from "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models" (2025) via the Forge UI, letting you reshape attention weights into a flatter distribution and use the altered prediction as negative guidance.

## How it works
- `create_nag_attention_processor` normalizes attention via L1/L2 norms or a temperature-scaled softmax and pipes the result through the Forge post-CFG callback.
- It replaces the `attn1` submodule for a selected U-Net block and applies the difference between conditioned denoised outputs as the negative term.
- The script can skip High-Resolution Fix passes and clamps guidance to the configured start/end steps.

## UI controls
- **Guidance Scale** (`3.0` default, range `1.0-8.0`) controls how strongly the normalized attention diff is added.
- **Normalization Type** chooses between `L2` (balanced), `L1` (fast simple), and `softmax_temp` (temperature control). Use the temperature slider when `softmax_temp` is active.
- **U-Net block/ID** selectors target the attention layer the guidance should patch (middle block is the sensible default for stable results).
- **Start/End step** range gating lets you focus the effect on a portion of the sampling trajectory.
- **Skip HiRes Fix** avoids modifying HR passes when enabled off.

## Tips
- L2 normalization with scale ~3.0 is the default "safe" setting; higher scales or softmax temperatures amplify the flattening effect.
- The guide accordion explains suggested presets (subtle/standard/strong/temperature-tuned).

## References
- Paper: https://arxiv.org/abs/2505.21179
- ComfyUI inspiration: https://github.com/pamparamm/sd-perturbed-attention
