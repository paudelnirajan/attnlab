"""
Getting weights onto disk ahead of time, and loading them without the network.

The server runs with HF_HUB_OFFLINE=1 (D18). Nothing is downloaded while a
request waits, the Hugging Face Hub being down can't break a model load, and
the disk only grows when a deploy runs `scripts/fetch_models.py`. That needs
two things, both here:

  fetch()           downloads exactly the files a model needs: weights, config
                    and tokenizer. Not the whole repo, which for a tokenizer
                    source like EleutherAI/gpt-neox-20b would be ~40 GB of
                    weights nobody asked for.
  offline listing   TransformerLens finds the checkpoint of NeelNanda/* models
                    (attn-only-2l-demo, the default model) by listing the repo's
                    files over the network, even when the file is cached, so
                    those loads fail offline. `install_cached_listing()` makes
                    that listing read the local snapshot first.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# Enough to load a model: configs, tokenizer files, weights. `*final.pth` is
# the one NeelNanda checkpoint TransformerLens loads; those repos also hold
# model_init.pth, optimizer state and 100+ training checkpoints (24 GB for
# attn-only-2l-demo), none of which it reads. .model covers sentencepiece.
MODEL_FILES = ["*.json", "*.txt", "*.model", "*.safetensors", "*final.pth", "*.tiktoken"]
TOKENIZER_FILES = ["*.json", "*.txt", "*.model", "*.tiktoken"]
# Weights and tokenizers live at the repo root. Subfolders hold other formats
# (onnx/, coreml/) or training checkpoints (checkpoints/).
IGNORE = ["*/*"]


class _CachedFirstHfApi(HfApi):
    def list_repo_files(self, repo_id: str, *args, **kwargs) -> list[str]:  # type: ignore[override]
        local = cached_snapshot(repo_id)
        if local is not None:
            return sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
        return super().list_repo_files(repo_id, *args, **kwargs)


def cached_snapshot(repo_id: str) -> Path | None:
    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached, or a partial cache
        return None


def install_cached_listing() -> None:
    from transformer_lens import loading_from_pretrained

    loading_from_pretrained.HfApi = _CachedFirstHfApi  # type: ignore[misc]


def repos_for(tl_name: str) -> tuple[str, str | None]:
    """(weights repo, tokenizer repo if different) for a TransformerLens name."""
    from transformer_lens import loading_from_pretrained as tl

    official = tl.get_official_model_name(tl_name)
    tokenizer = tl.get_pretrained_model_config(official).tokenizer_name
    return official, (tokenizer if tokenizer and tokenizer != official else None)


def fetch(repo_id: str, *, tokenizer_only: bool = False) -> Path:
    return Path(
        snapshot_download(
            repo_id, allow_patterns=TOKENIZER_FILES if tokenizer_only else MODEL_FILES, ignore_patterns=IGNORE
        )
    )
