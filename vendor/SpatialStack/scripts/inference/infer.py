"""Visual input and attention helpers retained from SpatialStack."""
import base64
import copy
from io import BytesIO
from PIL import Image
from qwen_vl_utils import extract_vision_info


def resolve_model_family(config):
    if getattr(config, "model_type", None) not in {"qwen3_5", "qwen3_5_vl"}:
        raise ValueError("The released controllers require Qwen3.5")
    return "qwen3_5"


def resolve_model_class(model_family, use_geometry_model=False):
    if model_family != "qwen3_5" or use_geometry_model:
        raise ValueError("Only the released visual Qwen3.5 controller is supported")
    from transformers import Qwen3_5ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration


def decode_message_image(image) -> Image.Image:
    if isinstance(image, str) and "base64," in image:
        _, base64_data = image.split("base64,", 1)
        with BytesIO(base64.b64decode(base64_data)) as bio:
            return copy.deepcopy(Image.open(bio)).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def patch_qwen3_5_flash_attention():
    try:
        import transformers.modeling_flash_attention_utils as flash_attention_utils
    except ImportError:
        return

    if getattr(flash_attention_utils, "_spatialstack_qwen3_5_mrope_patch", False):
        return

    original_is_packed_sequence = flash_attention_utils._is_packed_sequence

    def patched_is_packed_sequence(position_ids, batch_size):
        # Qwen3.5 multimodal inputs use 3D MRoPE position ids and are not packed sequences.
        if position_ids is not None and getattr(position_ids, "ndim", None) == 3:
            return False
        return original_is_packed_sequence(position_ids, batch_size)

    flash_attention_utils._is_packed_sequence = patched_is_packed_sequence
    flash_attention_utils._spatialstack_qwen3_5_mrope_patch = True


def patch_qwen3_5_linear_attention_fallback():
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        return

    modeling_qwen3_5.FusedRMSNormGated = None
    modeling_qwen3_5.chunk_gated_delta_rule = None
    modeling_qwen3_5.fused_recurrent_gated_delta_rule = None
    modeling_qwen3_5.causal_conv1d_fn = None
    modeling_qwen3_5.causal_conv1d_update = None
    modeling_qwen3_5.is_fast_path_available = False


def prepare_raw_visual_inputs(messages):
    image_inputs = []
    for message in messages:
        vision_info = extract_vision_info(message)
        for ele in vision_info:
            if "image" not in ele:
                continue
            image_inputs.append(decode_message_image(ele["image"]))
    return image_inputs
