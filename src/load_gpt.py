from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn


def extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint formats."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected checkpoint to be a mapping, got {type(checkpoint).__name__}"
        )

    for container_key in ("model_state_dict", "state_dict"):
        possible_state = checkpoint.get(container_key)

        if isinstance(possible_state, Mapping):
            return possible_state

    # The loaded object itself appears to be the state_dict.
    return checkpoint


def convert_checkpoint_key(old_key: str) -> str:
    """Convert the checkpoint naming scheme to this GPTModel's scheme."""

    # Handle checkpoints saved from DistributedDataParallel.
    if old_key.startswith("module."):
        old_key = old_key.removeprefix("module.")

    replacements = (
        ("tok_emb.", "token_embeddings."),
        ("pos_emb.", "position_embeddings."),
        ("trf_blocks.", "transformer_block."),
        (".att.", ".multi_head_attention."),
        (".ff.", ".feed_forward."),
        ("final_norm.", "final_layer_norm."),
    )

    new_key = old_key

    for source_name, target_name in replacements:
        new_key = new_key.replace(source_name, target_name)

    return new_key


def load_converted_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    source_state = extract_state_dict(checkpoint)
    target_state = model.state_dict()

    converted_state: dict[str, torch.Tensor] = {}
    unmatched_keys: list[tuple[str, str]] = []

    for old_key, tensor in source_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        new_key = convert_checkpoint_key(old_key)

        # Causal masks are deterministic and should be created by the model.
        if new_key.endswith(".mask"):
            continue

        if new_key not in target_state:
            unmatched_keys.append((old_key, new_key))
            continue

        expected_shape = target_state[new_key].shape

        if tensor.shape != expected_shape:
            raise RuntimeError(
                f"Shape mismatch:\n"
                f"  checkpoint key: {old_key}\n"
                f"  converted key:  {new_key}\n"
                f"  checkpoint:     {tuple(tensor.shape)}\n"
                f"  model:          {tuple(expected_shape)}"
            )

        if new_key in converted_state:
            raise RuntimeError(
                f"Multiple checkpoint keys were converted to {new_key!r}"
            )

        converted_state[new_key] = tensor

    missing_keys = sorted(set(target_state) - set(converted_state))

    if unmatched_keys:
        details = "\n".join(
            f"  {old_key} -> {new_key}"
            for old_key, new_key in unmatched_keys
        )
        raise RuntimeError(f"Checkpoint contains unmatched keys:\n{details}")

    if missing_keys:
        details = "\n".join(f"  {key}" for key in missing_keys)
        raise RuntimeError(f"Model parameters missing from checkpoint:\n{details}")

    model.load_state_dict(converted_state, strict=True)

    print(f"Successfully loaded {len(converted_state)} tensors.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    torch.save(
        {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        "../models/gpt2_small_model_and_optimizer.pth"
    )