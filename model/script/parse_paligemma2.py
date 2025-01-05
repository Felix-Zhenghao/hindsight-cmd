"""
####################################################
PaliGemma checkpoint
Original implementation: https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/paligemma/paligemma.py
####################################################
"""

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch


"""
The paligemma2-3b-pt-448 model architecture:

PaliGemmaForConditionalGeneration(
  (vision_tower): SiglipVisionModel(
    (vision_model): SiglipVisionTransformer(
      (embeddings): SiglipVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
        (position_embedding): Embedding(1024, 1152)
      )
      (encoder): SiglipEncoder(
        (layers): ModuleList(
          (0-26): 27 x SiglipEncoderLayer(
            (self_attn): SiglipSdpaAttention(
              (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
              (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
              (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
              (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
            )
            (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
            (mlp): SiglipMLP(
              (activation_fn): PytorchGELUTanh()
              (fc1): Linear(in_features=1152, out_features=4304, bias=True)
              (fc2): Linear(in_features=4304, out_features=1152, bias=True)
            )
            (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
    )
  )
  (multi_modal_projector): PaliGemmaMultiModalProjector(
    (linear): Linear(in_features=1152, out_features=2304, bias=True)
  )
  (language_model): Gemma2ForCausalLM(
    (model): Gemma2Model(
      (embed_tokens): Embedding(257216, 2304, padding_idx=0)
      (layers): ModuleList(
        (0-25): 26 x Gemma2DecoderLayer(
          (self_attn): Gemma2Attention(
            (q_proj): Linear(in_features=2304, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2304, out_features=1024, bias=False)
            (v_proj): Linear(in_features=2304, out_features=1024, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2304, bias=False)
            (rotary_emb): Gemma2RotaryEmbedding()
          )
          (mlp): Gemma2MLP(
            (gate_proj): Linear(in_features=2304, out_features=9216, bias=False)
            (up_proj): Linear(in_features=2304, out_features=9216, bias=False)
            (down_proj): Linear(in_features=9216, out_features=2304, bias=False)
            (act_fn): PytorchGELUTanh()
          )
          (input_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
          (post_attention_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
          (pre_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
          (post_feedforward_layernorm): Gemma2RMSNorm((2304,), eps=1e-06)
        )
      )
      (norm): Gemma2RMSNorm((2304,), eps=1e-06)
    )
    (lm_head): Linear(in_features=2304, out_features=257216, bias=False)
  )
)
"""

if __name__ == '__main__':
  # login the huggingface account
  from huggingface_hub import login
  login(token="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")

  """
  ####################################################
  # Load the model and processor
  ####################################################
  """
  # load pretrained model
  model_id = "google/paligemma2-3b-pt-448"
  model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
  processor = PaliGemmaProcessor.from_pretrained(model_id)

  # example image input
  url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
  image = load_image(url)
  model_inputs = processor(text='', images=image, return_tensors="pt").to(torch.bfloat16)
  image_input_pixel_value = model_inputs.pixel_values


  """
  ####################################################
  # get the useful part: the image encoder (vision_tower)
  ####################################################
  """
  image_encoder = model.vision_tower
  image_encoder.push_to_hub("Felix-Zhenghao/paligemma2-3b-pt-448-img-encoder")


  """
  ####################################################
  # get the useful part: the projector from patch embedding to token embedding
  ####################################################
  """
  multimodal_projector = model.multi_modal_projector

  # the key of the pth file should be changed to projector.weight and projector.bias
  state_dict = {"projector.weight": multimodal_projector.linear.weight, "projector.bias": multimodal_projector.linear.bias}
  torch.save(state_dict, "model/weights/projector_1152_to_2304.pth")


"""
####################################################
# get the useful part: the text embedding model
####################################################
"""
state_dict = {"weight.weight": model.language_model.model.embed_tokens.weight}
torch.save(state_dict, "model/weights/token_id_to_embedding.pth")