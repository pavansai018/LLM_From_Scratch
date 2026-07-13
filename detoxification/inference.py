from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import tiktoken
import torch
import torch.nn as nn

import config
from gpt_model import GPTModel


class GPT2DetoxInference:
    def __init__(self, checkpoint_path: str, device: str | None = None,):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eos_token_id = 50256
        self.instruction = "Rewrite the following message in neutral and respectful language while preserving its original non-harmful meaning."
        self.model, self.model_config = self._load_model(checkpoint_path=checkpoint_path)
        self.context_length = self.model_config["context_length"]
        print(f"Device: {self.device}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Context length: {self.context_length}")

    def _load_checkpoint(self, checkpoint_path: str,) -> Any:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        # weights_only is unavailable in some older PyTorch versions.
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=True,)
        except TypeError:
            checkpoint = torch.load(checkpoint_file, map_location=self.device,)

        return checkpoint

    def _load_model(self, checkpoint_path: str,) -> tuple[nn.Module, dict]:
        checkpoint = self._load_checkpoint(checkpoint_path)

        # Case 1:
        # {
        #     "model_state_dict": ...,
        #     "optimizer_state_dict": ...,
        #     "model_config": ...
        # }
        if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
            model_config = checkpoint.get("model_config", config.GPT2_SMALL_124M,)
            state_dict = checkpoint["model_state_dict"]

        # Case 2:
        # torch.save(model.state_dict(), path)
        elif isinstance(checkpoint, dict):
            model_config = config.GPT2_SMALL_124M
            state_dict = checkpoint

        else:
            raise TypeError(
                "Unsupported checkpoint format. Expected either "
                "a model state_dict or a training checkpoint "
                "containing 'model_state_dict'."
            )

        model = GPTModel(cfg=model_config).to(self.device)
        model.load_state_dict(state_dict, strict=True,)
        model.eval()
        return model, model_config

    def _create_prompt(self, toxic_message: str,) -> str:
        return (
            "### Instruction:\n"
            f"{self.instruction}\n\n"
            "### Input:\n"
            f"{toxic_message.strip()}\n\n"
            "### Response:\n")

    @torch.inference_mode()
    def generate(self, toxic_message: str, max_new_tokens: int = 100,) -> str:
        if not toxic_message.strip():
            raise ValueError(
                "The input message cannot be empty."
            )

        prompt = self._create_prompt(toxic_message=toxic_message)
        prompt_token_ids = self.tokenizer.encode(prompt, allowed_special=set(),)
        if len(prompt_token_ids) >= self.context_length:
            raise ValueError(
                f"Prompt contains {len(prompt_token_ids)} tokens, "
                f"but the model context length is "
                f"{self.context_length}."
            )

        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device,).unsqueeze(0)
        generated_token_ids: list[int] = []
        for _ in range(max_new_tokens):
            # Keep the sequence inside the model's context window.
            model_input = input_ids[:,-self.context_length:,]
            # logits:
            # [batch, sequence_length, vocabulary_size]
            logits = self.model(model_input)
            # Distribution for the token immediately after
            # the current sequence.
            next_token_logits = logits[:, -1, :]
            # Greedy decoding gives deterministic output.
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True,)
            token_id = int(next_token_id.item())
            if token_id == self.eos_token_id:
                break
            generated_token_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token_id], dim=1,)
        response = self.tokenizer.decode(generated_token_ids)
        return response.strip()


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=("Generate a neutral rewrite using the fine-tuned GPT-2 model."))
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pt checkpoint.",)
    parser.add_argument("--text", type=str, default=None, help=("Toxic message to rewrite. When omitted, interactive mode is started."),)
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of response tokens.",)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force CPU or CUDA. Default selects automatically.",)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    inference = GPT2DetoxInference(checkpoint_path=args.checkpoint, device=args.device,)

    if args.text is not None:
        response = inference.generate(toxic_message=args.text, max_new_tokens=args.max_new_tokens,)
        print("\nInput:")
        print(args.text)
        print("\nNeutral response:")
        print(response)
        return

    print("\nInteractive mode")
    print("Enter 'quit' to stop.\n")

    while True:
        toxic_message = input("Input: ").strip()
        if toxic_message.lower() in {"quit", "exit", "q",}:
            break
        if not toxic_message:
            continue
        response = inference.generate(
            toxic_message=toxic_message,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"Neutral response: {response}\n")


if __name__ == "__main__":
    main()