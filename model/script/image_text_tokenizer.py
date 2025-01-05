import torch
from torch import nn
from transformers import AutoModel

import sentencepiece

class TextTokenEmbedding(nn.Module):
    """
    Gonvert token ids to embeddings.

    For example:

    `[2169, 476, 3311, 89788, 235269, 573, 54260, 235303, 235256, 5316, 17008]`
    -> `torch.Size([11, 2304])`
    """

    def __init__(self,):
        super().__init__()
        self.weight = nn.Embedding(num_embeddings=257216, embedding_dim=2304, padding_idx=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight(token_ids)

class MultiModelProjector(nn.Module):
    """
    A projector from patch embeddings to token embeddings.

    For example:

    `torch.Size([1, 1024, 1152])` -> `torch.Size([1, 1024, 2304])`
    """
    def __init__(self,
                 patch_dim: int,
                 token_dim: int):
        super().__init__()
        self.projector = nn.Linear(patch_dim, token_dim)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        return self.projector(patch_embeddings)

class TextTokenizer(nn.Module):
    """
    Convert original text to token embeddings.

    For example:
    
    `"As a general guideline, the CDC's average requirement"` -> `torch.Size([11, 2304])`
    """

    def __init__(self,
                 text_to_token_id_path: str,
                 token_id_to_embedding_path: str):
        super().__init__()
        self.text_token_id_generator = sentencepiece.SentencePieceProcessor(text_to_token_id_path)
        self.token_id_to_embedding = TextTokenEmbedding()
        self.token_id_to_embedding.load_state_dict(torch.load(token_id_to_embedding_path, weights_only=True))

    def forward(self, text: str) -> torch.Tensor:
        text_token_ids = torch.tensor(self.text_token_id_generator.encode(text))
        return self.token_id_to_embedding(text_token_ids)

class ImageTextTokenizer(nn.Module):
    """
    Convert image and text to token embeddings.

    For example:
    - image: `torch.Size([1, 3, 448, 448])`
    - text: `"As a general guideline, the CDC's average requirement"`

    -> img_emb, txt_emb = `torch.Size([1, 1024, 2304])`, `torch.Size([11, 2304])`
    """
    def __init__(self,
                 image_encoder_hg_repo: str,
                 patch_to_token_projector_path: str,
                 text_to_token_id_path: str,
                 token_id_to_embedding_path: str,
                 patch_dim: int,
                 token_dim: int):
        super().__init__()

        # Load the image encoder from Hugging Face repository
        self.image_encoder = AutoModel.from_pretrained(image_encoder_hg_repo)

        # Load the patch to token projector from pth file
        self.patch_to_token_projector = MultiModelProjector(patch_dim, token_dim)
        self.patch_to_token_projector.load_state_dict(torch.load(patch_to_token_projector_path))

        # Load the text tokenizer from Hugging Face repository
        self.text_tokenizer = TextTokenizer(text_to_token_id_path=text_to_token_id_path,
                                            token_id_to_embedding_path=token_id_to_embedding_path)

    def forward(self,
                image: torch.Tensor,
                text: str,
                )-> torch.Tensor:
        # Get the patch embeddings from the image encoder
        patch_embeddings = self.image_encoder(image).last_hidden_state

        # Project the patch embeddings to token embeddings
        img_token_embeddings = self.patch_to_token_projector(patch_embeddings)

        # Get the text token embeddings
        text_token_embeddings = self.text_tokenizer(text)

        return img_token_embeddings, text_token_embeddings


if __name__ == '__main__':
    from transformers import (
        PaliGemmaProcessor,
    )
    from transformers.image_utils import load_image

    # get the model and process the example image input
    model_id = "google/paligemma2-3b-pt-448"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    image = load_image(url)
    model_inputs = processor(images=image,
                             return_tensors="pt").to(torch.bfloat16)
    image_input_pixel_value = model_inputs.pixel_values

    # create the tokenizer
    tokenizer = ImageTextTokenizer(image_encoder_hg_repo="Felix-Zhenghao/paligemma2-3b-pt-448-img-encoder",
                                   patch_to_token_projector_path="model/weights/projector_1152_to_2304.pth",
                                   text_to_token_id_path="model/weights/paligemma_tokenizer.model",
                                   token_id_to_embedding_path="model/weights/token_id_to_embedding.pth",
                                   patch_dim=1152,
                                   token_dim=2304)
    img_token_embeddings, text_token_embeddings = tokenizer(image=image_input_pixel_value,
                                                            text="As a general guideline, the CDC's average requirement")
    print(img_token_embeddings.shape) # torch.Size([1, 1024, 2304])
    print(text_token_embeddings.shape) # torch.Size([11, 2304])
