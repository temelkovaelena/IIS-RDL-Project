"""Column encoders (numeric / categorical / timestamp / text) into a shared space.

Declarative via PyTorch Frame, so it only runs on the GPU side.

Text columns need an embedder supplied up front: torch_frame refuses to build a dataset if
any column is inferred as text and no embedder is given. In rel-f1 those columns are names
(driver, constructor, circuit, race), which carry little signal, but they still have to be
encoded for the graph to build at all.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Small and fast. Enough for short name strings, and it keeps the download light.
DEFAULT_TEXT_MODEL = "sentence-transformers/average_word_embeddings_glove.6B.300d"


class TextEmbedder:
    """Wraps a sentence-transformers model in the interface torch_frame expects."""

    def __init__(self, device: torch.device | None = None, model_name: str = DEFAULT_TEXT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, sentences: list[str]) -> Tensor:
        return torch.from_numpy(
            self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        )


def text_embedder_config(device: torch.device | None = None, batch_size: int = 256):
    from torch_frame.config.text_embedder import TextEmbedderConfig

    return TextEmbedderConfig(text_embedder=TextEmbedder(device=device), batch_size=batch_size)


def drop_text_columns(col_to_stype_dict: dict) -> dict:
    """Remove text columns from a stype proposal.

    A fallback for when the embedder model cannot be downloaded. It changes what the model
    sees, so numbers produced this way are not comparable to the published ones.
    """
    return {
        table: {col: s for col, s in mapping.items() if "text" not in str(s)}
        for table, mapping in col_to_stype_dict.items()
    }
