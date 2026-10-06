#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LUMORA 750M
# TRAINING VON NULL
# ============================================================
#
# Ordner:
#
# lumora/
# ├── train.py
# ├── tokenizer.js
# ├── server.js
# ├── transformer.js
# └── DATEN/
#     ├── a.json
#     ├── b.json
#     └── ...
#
# Das Modell wird NICHT von einem vorhandenen Modell geladen.
# Alle Gewichte werden zufällig initialisiert und anschließend
# durch Next-Token-Prediction trainiert.
#
# Architektur:
#
# Vocabulary     = 32.000
# Hidden         = 1.536
# Layers         = 26
# Heads          = 24
# Head Dimension = 64
# FFN            = 3.840
# Context        = 2.048
#
# Parameter      ≈ 754,7 Millionen
#
# ============================================================


# ============================================================
# DEFAULT MODEL CONFIG
# ============================================================

MODEL_CONFIG = {
    "model_name": "Lumora-750M",
    "architecture": "decoder-only-transformer",

    "vocab_size": 32000,

    "hidden_size": 1536,

    "num_layers": 26,

    "num_heads": 24,

    "head_dim": 64,

    "ffn_size": 3840,

    "max_seq_len": 2048,

    "rope_theta": 10000.0,

    "dropout": 0.0
}


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>"
]


# ============================================================
# TOKENIZER
# ============================================================

TOKEN_PATTERN = re.compile(
    r"https?://[^\s]+"
    r"|www\.[^\s]+"
    r"|[^\W_]+(?:['’\-][^\W_]+)*"
    r"|[^\s\w]",
    re.UNICODE
)


def normalize_text(text: str) -> str:

    text = str(text or "")

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    return text.strip()


def tokenize(text: str) -> List[str]:

    text = normalize_text(text)

    tokens = TOKEN_PATTERN.findall(text)

    return tokens


def word_tokens(text: str) -> List[str]:

    return [
        token
        for token in tokenize(text)
        if any(char.isalnum() for char in token)
    ]


# ============================================================
# JSON DATEIEN FINDEN
# ============================================================

def find_json_files(
    data_dir: Path
) -> List[Path]:

    files = []

    for path in data_dir.rglob("*.json"):

        if path.is_file():
            files.append(path)

    return sorted(files)


# ============================================================
# JSON TEXT REKURSIV AUSLESEN
# ============================================================

def extract_texts(
    value
) -> Iterable[str]:

    if isinstance(value, str):

        text = value.strip()

        if text:
            yield text

        return


    if isinstance(value, list):

        for item in value:

            yield from extract_texts(item)

        return


    if isinstance(value, dict):

        preferred_keys = [
            "text",
            "content",
            "prompt",
            "response",
            "answer",
            "question",
            "instruction",
            "output",
            "input",
            "message",
            "title",
            "description"
        ]

        used = set()

        for key in preferred_keys:

            if key in value:

                used.add(key)

                yield from extract_texts(
                    value[key]
                )


        if not used:

            for item in value.values():

                yield from extract_texts(item)


# ============================================================
# JSON TEXT ITERATOR
# ============================================================

def iterate_dataset(
    files: List[Path]
):

    for file_path in files:

        try:

            with file_path.open(
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)


            text_index = 0

            for text in extract_texts(data):

                yield (
                    file_path,
                    text_index,
                    text
                )

                text_index += 1


        except Exception as error:

            print(
                "[DATA ERROR]",
                file_path,
                error
            )


# ============================================================
# VOCABULARY AUFBAUEN
# ============================================================

def build_vocabulary(
    files: List[Path],
    target_size: int
):

    from collections import Counter

    counter = Counter()

    total_texts = 0

    total_tokens = 0


    print()
    print(
        "[VOCAB] Analysiere DATEN/"
    )


    for (
        _file,
        _index,
        text
    ) in iterate_dataset(files):

        tokens = tokenize(text)

        counter.update(tokens)

        total_tokens += len(tokens)

        total_texts += 1


    print(
        "[VOCAB] Texte:",
        total_texts
    )

    print(
        "[VOCAB] Tokens:",
        total_tokens
    )

    print(
        "[VOCAB] Unique:",
        len(counter)
    )


    vocabulary = []

    vocabulary.extend(
        SPECIAL_TOKENS
    )


    for token, _frequency in counter.most_common():

        if token in SPECIAL_TOKENS:
            continue

        vocabulary.append(token)

        if len(vocabulary) >= target_size:
            break


    # Falls weniger als 32k Tokens
    # vorhanden sind, wird das Vokabular
    # mit reservierten Tokens aufgefüllt.

    while len(vocabulary) < target_size:

        vocabulary.append(
            f"<extra_{len(vocabulary):05d}>"
        )


    vocabulary = vocabulary[:target_size]


    token_to_id = {
        token: index
        for index, token
        in enumerate(vocabulary)
    }


    return (
        token_to_id,
        vocabulary
    )


# ============================================================
# VOCABULARY SPEICHERN
# ============================================================

def save_vocabulary(
    output_dir: Path,
    token_to_id: Dict[str, int],
    vocabulary: List[str]
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    vocab_data = {

        "special_tokens":
            SPECIAL_TOKENS,

        "vocab_size":
            len(vocabulary),

        "token_to_id":
            token_to_id,

        "vocab":
            vocabulary
    }


    with (
        output_dir /
        "vocab.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vocab_data,
            file,
            ensure_ascii=False,
            indent=2
        )


    tokenizer_config = {

        "type":
            "lumora-tokenizer",

        "normalization":
            "NFKC + lowercase + trim",

        "special_tokens":
            SPECIAL_TOKENS
    }


    with (
        output_dir /
        "tokenizer_config.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tokenizer_config,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# TOKENSTREAM ERZEUGEN
# ============================================================

def build_token_stream(
    files: List[Path],
    token_to_id: Dict[str, int],
    output_file: Path
):

    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]
    unk_id = token_to_id["<unk>"]


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    total_tokens = 0

    buffer = []

    buffer_limit = 1_000_000


    print()
    print(
        "[DATA] Erzeuge Tokenstream..."
    )


    with output_file.open(
        "wb"
    ) as output:

        for (
            file_path,
            text_index,
            text
        ) in iterate_dataset(files):

            ids = []

            ids.append(
                bos_id
            )


            for token in tokenize(text):

                token_id = (
                    token_to_id.get(
                        token,
                        unk_id
                    )
                )

                ids.append(
                    token_id
                )


            ids.append(
                eos_id
            )


            buffer.extend(ids)

            total_tokens += len(ids)


            if len(buffer) >= buffer_limit:

                array = np.asarray(
                    buffer,
                    dtype=np.uint32
                )

                output.write(
                    array.tobytes()
                )

                buffer.clear()


        if buffer:

            array = np.asarray(
                buffer,
                dtype=np.uint32
            )

            output.write(
                array.tobytes()
            )


    print(
        "[DATA] Tokenstream:",
        total_tokens
    )


    return total_tokens


# ============================================================
# TOKEN MEMMAP
# ============================================================

def open_tokenstream(
    file_path: Path
):

    size = file_path.stat().st_size

    if size % 4 != 0:

        raise RuntimeError(
            "tokens.bin ist beschädigt."
        )


    count = size // 4


    return np.memmap(
        file_path,
        mode="r",
        dtype=np.uint32,
        shape=(count,)
    )


# ============================================================
# RMS NORM
# ============================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dimension,
        eps=1e-6
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dimension)
        )

        self.eps = eps


    def forward(
        self,
        x
    ):

        variance = (
            x.float()
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True
            )
        )


        x = x * torch.rsqrt(
            variance + self.eps
        )


        return (
            self.weight * x
        )


# ============================================================
# ROPE
# ============================================================

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dimension,
        max_seq_len,
        theta
    ):

        super().__init__()


        inverse_frequency = 1.0 / (
            theta ** (
                torch.arange(
                    0,
                    dimension,
                    2,
                    dtype=torch.float32
                )
                / dimension
            )
        )


        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32
        )


        frequencies = torch.outer(
            positions,
            inverse_frequency
        )


        self.register_buffer(
            "cos",
            frequencies.cos(),
            persistent=False
        )


        self.register_buffer(
            "sin",
            frequencies.sin(),
            persistent=False
        )


    def apply(
        self,
        q,
        k,
        sequence_length
    ):

        cos = self.cos[
            :sequence_length
        ]

        sin = self.sin[
            :sequence_length
        ]


        cos = cos[
            None,
            None,
            :,
            :
        ]


        sin = sin[
            None,
            None,
            :,
            :
        ]


        q_even = q[..., ::2]
        q_odd = q[..., 1::2]

        k_even = k[..., ::2]
        k_odd = k[..., 1::2]


        q_rot_even = (
            q_even * cos -
            q_odd * sin
        )

        q_rot_odd = (
            q_even * sin +
            q_odd * cos
        )


        k_rot_even = (
            k_even * cos -
            k_odd * sin
        )

        k_rot_odd = (
            k_even * sin +
            k_odd * cos
        )


        q = torch.stack(
            (
                q_rot_even,
                q_rot_odd
            ),
            dim=-1
        ).flatten(-2)


        k = torch.stack(
            (
                k_rot_even,
                k_rot_odd
            ),
            dim=-1
        ).flatten(-2)


        return q, k


# ============================================================
# MULTI HEAD ATTENTION
# ============================================================

class SelfAttention(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        hidden = (
            config["hidden_size"]
        )

        heads = (
            config["num_heads"]
        )

        head_dim = (
            config["head_dim"]
        )


        if heads * head_dim != hidden:

            raise ValueError(
                "num_heads * head_dim "
                "muss hidden_size entsprechen."
            )


        self.num_heads = heads

        self.head_dim = head_dim


        self.qkv = nn.Linear(
            hidden,
            hidden * 3,
            bias=False
        )


        self.output = nn.Linear(
            hidden,
            hidden,
            bias=False
        )


        self.rope = RotaryEmbedding(
            head_dim,
            config["max_seq_len"],
            config["rope_theta"]
        )


    def forward(
        self,
        x
    ):

        batch,
        sequence,
        hidden = x.shape


        qkv = self.qkv(x)


        q, k, v = torch.chunk(
            qkv,
            3,
            dim=-1
        )


        q = q.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        k = k.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        v = v.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )


        q, k = self.rope.apply(
            q,
            k,
            sequence
        )


        if hasattr(
            F,
            "scaled_dot_product_attention"
        ):

            result = (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=True
                )
            )

        else:

            scale = (
                self.head_dim ** -0.5
            )


            scores = (
                q @
                k.transpose(
                    -2,
                    -1
                )
            ) * scale


            mask = torch.tril(
                torch.ones(
                    sequence,
                    sequence,
                    dtype=torch.bool,
                    device=x.device
                )
            )


            scores = scores.masked_fill(
                ~mask,
                torch.finfo(
                    scores.dtype
                ).min
            )


            probabilities = F.softmax(
                scores,
                dim=-1
            )


            result = (
                probabilities @ v
            )


        result = (
            result
            .transpose(1, 2)
            .contiguous()
            .view(
                batch,
                sequence,
                hidden
            )
        )


        return self.output(
            result
        )


# ============================================================
# SWIGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        hidden = (
            config["hidden_size"]
        )

        ffn = (
            config["ffn_size"]
        )


        self.gate = nn.Linear(
            hidden,
            ffn,
            bias=False
        )


        self.up = nn.Linear(
            hidden,
            ffn,
            bias=False
        )


        self.down = nn.Linear(
            ffn,
            hidden,
            bias=False
        )


    def forward(
        self,
        x
    ):

        gate = F.silu(
            self.gate(x)
        )

        up = self.up(x)

        return self.down(
            gate * up
        )


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        hidden = (
            config["hidden_size"]
        )


        self.norm1 = RMSNorm(
            hidden
        )


        self.attention = SelfAttention(
            config
        )


        self.norm2 = RMSNorm(
            hidden
        )


        self.mlp = SwiGLU(
            config
        )


    def forward(
        self,
        x
    ):

        x = (
            x +
            self.attention(
                self.norm1(x)
            )
        )


        x = (
            x +
            self.mlp(
                self.norm2(x)
            )
        )


        return x


# ============================================================
# LUMORA TRANSFORMER
# ============================================================

class LumoraTransformer(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()


        self.config = config


        vocab_size = (
            config["vocab_size"]
        )

        hidden_size = (
            config["hidden_size"]
        )


        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size
        )


        self.blocks = nn.ModuleList(

            [
                TransformerBlock(config)

                for _ in range(
                    config["num_layers"]
                )
            ]

        )


        self.final_norm = RMSNorm(
            hidden_size
        )


        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )


        # Weight Tying
        self.lm_head.weight = (
            self.embedding.weight
        )


        self.apply(
            self._initialize_weights
        )


        self.lm_head.weight = (
            self.embedding.weight
        )


        self.gradient_checkpointing = False


    def _initialize_weights(
        self,
        module
    ):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )


        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


        if hasattr(
            module,
            "output"
        ):

            if isinstance(
                module.output,
                nn.Linear
            ):

                std = (
                    0.02 /
                    math.sqrt(
                        2 *
                        self.config[
                            "num_layers"
                        ]
                    )
                )


                nn.init.normal_(
                    module.output.weight,
                    mean=0.0,
                    std=std
                )


        if hasattr(
            module,
            "down"
        ):

            if isinstance(
                module.down,
                nn.Linear
            ):

                std = (
                    0.02 /
                    math.sqrt(
                        2 *
                        self.config[
                            "num_layers"
                        ]
                    )
                )


                nn.init.normal_(
                    module.down.weight,
                    mean=0.0,
                    std=std
                )


    def forward(
        self,
        input_ids,
        labels=None
    ):

        x = self.embedding(
            input_ids
        )


        for block in self.blocks:

            if (
                self.training and
                self.gradient_checkpointing
            ):

                from torch.utils.checkpoint import checkpoint

                x = checkpoint(
                    block,
                    x,
                    use_reentrant=False
                )

            else:

                x = block(x)


        x = self.final_norm(x)


        logits = self.lm_head(
            x
        )


        loss = None


        if labels is not None:

            prediction = (
                logits[:, :-1, :]
                .contiguous()
            )


            targets = (
                labels[:, 1:]
                .contiguous()
            )


            loss = F.cross_entropy(

                prediction.view(
                    -1,
                    prediction.size(-1)
                ),

                targets.view(-1)

            )


        return (
            logits,
            loss
        )


# ============================================================
# PARAMETER BERECHNUNG OHNE MODELL ZU LADEN
# ============================================================

def estimate_parameter_count(
    config
):

    vocab = (
        config["vocab_size"]
    )

    hidden = (
        config["hidden_size"]
    )

    layers = (
        config["num_layers"]
    )

    ffn = (
        config["ffn_size"]
    )


    embedding = (
        vocab *
        hidden
    )


    attention = (
        4 *
        hidden *
        hidden
    )


    mlp = (
        3 *
        hidden *
        ffn
    )


    norms = (
        4 *
        hidden
    )


    final_norm = (
        2 *
        hidden
    )


    return (
        embedding +
        layers *
        (
            attention +
            mlp +
            norms
        ) +
        final_norm
    )


# ============================================================
# MODELL CONFIG SPEICHERN
# ============================================================

def save_model_config(
    output_dir,
    config,
    parameters
):

    data = {

        "model_name":
            config["model_name"],

        "architecture":
            config["architecture"],

        "parameters":
            parameters,

        "parameters_millions":
            parameters / 1_000_000,

        "config":
            config,

        "weight_tying":
            True,

        "from_scratch":
            True,

        "tokenizer":
            "lumora-tokenizer"
    }


    with (
        output_dir /
        "model_config.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )


# ============================================================
# BATCH SAMPLING
# ============================================================

def get_batch(
    tokens,
    batch_size,
    sequence_length,
    start_min,
    start_max,
    device
):

    starts = np.random.randint(

        start_min,

        start_max + 1,

        size=batch_size

    )


    x = np.stack(

        [
            tokens[
                start:
                start + sequence_length
            ]

            for start in starts
        ]

    ).astype(
        np.int64
    )


    y = np.stack(

        [
            tokens[
                start:
                start + sequence_length
            ]

            for start in starts
        ]

    ).astype(
        np.int64
    )


    x = torch.from_numpy(
        x
    ).to(
        device,
        dtype=torch.long
    )


    y = torch.from_numpy(
        y
    ).to(
        device,
        dtype=torch.long
    )


    return (
        x,
        y
    )


# ============================================================
# EVAL
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    tokens,
    batch_size,
    sequence_length,
    start_min,
    start_max,
    batches,
    device,
    autocast_dtype
):

    model.eval()


    losses = []


    for _ in range(batches):

        x, y = get_batch(

            tokens,

            batch_size,

            sequence_length,

            start_min,

            start_max,

            device

        )


        enabled = (
            autocast_dtype is not None
        )


        with torch.autocast(

            device_type=device.type,

            dtype=autocast_dtype,

            enabled=enabled

        ):

            _logits, loss = model(
                x,
                y
            )


        losses.append(
            loss.item()
        )


    model.train()


    return (
        sum(losses) /
        len(losses)
    )


# ============================================================
# CHECKPOINT
# ============================================================

def atomic_save(
    data,
    path
):

    temporary = Path(
        str(path) + ".tmp"
    )


    torch.save(
        data,
        temporary
    )


    os.replace(
        temporary,
        path
    )


def save_checkpoint(
    output_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    best_loss,
    save_optimizer
):

    checkpoint_dir = (
        output_dir /
        "checkpoints"
    )


    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    state = {

        "step":
            step,

        "model":
            model.state_dict(),

        "config":
            model.config,

        "best_val_loss":
            best_loss
    }


    if save_optimizer:

        state["optimizer"] = (
            optimizer.state_dict()
        )

        state["scheduler"] = (
            scheduler.state_dict()
        )

        if scaler is not None:

            state["scaler"] = (
                scaler.state_dict()
            )


    latest_path = (
        checkpoint_dir /
        "latest.pt"
    )


    step_path = (
        checkpoint_dir /
        f"step_{step:08d}.pt"
    )


    atomic_save(
        state,
        latest_path
    )


    atomic_save(
        state,
        step_path
    )


    return latest_path


# ============================================================
# CHECKPOINT LADEN
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    load_optimizer=False
):

    print(
        "[CHECKPOINT] Lade:",
        path
    )


    checkpoint = torch.load(

        path,

        map_location="cpu",

        weights_only=False

    )


    model.load_state_dict(
        checkpoint["model"]
    )


    if (
        load_optimizer and
        optimizer is not None and
        "optimizer" in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )


    if (
        load_optimizer and
        scheduler is not None and
        "scheduler" in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )


    if (
        load_optimizer and
        scaler is not None and
        "scaler" in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint["scaler"]
        )


    return (

        int(
            checkpoint.get(
                "step",
                0
            )
        ),

        float(
            checkpoint.get(
                "best_val_loss",
                float("inf")
            )
        )

    )


# ============================================================
# LR SCHEDULE
# ============================================================

def learning_rate_lambda(
    step,
    warmup_steps,
    total_steps,
    min_lr_ratio
):

    if step < warmup_steps:

        return (
            max(step, 1) /
            max(warmup_steps, 1)
        )


    progress = (

        step -
        warmup_steps

    ) / max(

        total_steps -
        warmup_steps,

        1

    )


    progress = min(
        max(progress, 0.0),
        1.0
    )


    cosine = (
        0.5 *
        (
            1.0 +
            math.cos(
                math.pi *
                progress
            )
        )
    )


    return (
        min_lr_ratio +
        (
            1.0 -
            min_lr_ratio
        ) *
        cosine
    )


# ============================================================
# DEVICE
# ============================================================

def select_device(
    requested
):

    if requested == "auto":

        if torch.cuda.is_available():

            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )


    if requested == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA ist nicht verfügbar."
            )


        return torch.device(
            "cuda"
        )


    return torch.device(
        "cpu"
    )


# ============================================================
# AMP
# ============================================================

def select_autocast_dtype(
    device,
    requested
):

    if device.type != "cuda":
        return None


    if requested == "none":
        return None


    if requested == "float16":
        return torch.float16


    if requested == "bfloat16":
        return torch.bfloat16


    # AUTO

    if (
        hasattr(
            torch.cuda,
            "is_bf16_supported"
        )
        and
        torch.cuda.is_bf16_supported()
    ):

        return torch.bfloat16


    return torch.float16


# ============================================================
# SEED
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lumora 750M Transformer "
            "von null trainieren"
        )
    )


    parser.add_argument(
        "--data-dir",
        default="DATEN"
    )


    parser.add_argument(
        "--output-dir",
        default="LUMORA_MODEL"
    )


    parser.add_argument(
        "--steps",
        type=int,
        default=1000
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )


    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048
    )


    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16
    )


    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4
    )


    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5
    )


    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1
    )


    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100
    )


    parser.add_argument(
        "--save-every",
        type=int,
        default=100
    )


    parser.add_argument(
        "--eval-every",
        type=int,
        default=100
    )


    parser.add_argument(
        "--eval-batches",
        type=int,
        default=10
    )


    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu"
        ],
        default="auto"
    )


    parser.add_argument(
        "--amp",
        choices=[
            "auto",
            "float16",
            "bfloat16",
            "none"
        ],
        default="auto"
    )


    parser.add_argument(
        "--resume",
        default=""
    )


    parser.add_argument(
        "--save-optimizer",
        action="store_true"
    )


    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true"
    )


    parser.add_argument(
        "--build-only",
        action="store_true"
    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # VALIDIERUNG
    # --------------------------------------------------------

    if args.seq_len > MODEL_CONFIG["max_seq_len"]:

        raise ValueError(
            "seq-len ist größer als "
            "max_seq_len."
        )


    if args.batch_size < 1:

        raise ValueError(
            "batch-size muss >= 1 sein."
        )


    if args.grad_accum < 1:

        raise ValueError(
            "grad-accum muss >= 1 sein."
        )


    set_seed(
        args.seed
    )


    data_dir = Path(
        args.data_dir
    )


    output_dir = Path(
        args.output_dir
    )


    data_output = (
        output_dir /
        "data"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    data_output.mkdir(
        parents=True,
        exist_ok=True
    )


    if not data_dir.exists():

        raise FileNotFoundError(

            "DATEN-Ordner nicht gefunden: "

            + str(
                data_dir.resolve()
            )

        )


    # --------------------------------------------------------
    # JSON DATEIEN
    # --------------------------------------------------------

    json_files = find_json_files(
        data_dir
    )


    if not json_files:

        raise FileNotFoundError(

            "Keine JSON-Dateien unter DATEN/ gefunden."

        )


    print()
    print(
        "=============================================="
    )
    print(
        "          LUMORA 750M FROM SCRATCH"
    )
    print(
        "=============================================="
    )
    print(
        "JSON-Dateien:",
        len(json_files)
    )
    print(
        "DATEN:",
        data_dir.resolve()
    )
    print(
        "Ausgabe:",
        output_dir.resolve()
    )
    print(
        "=============================================="
    )


    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    token_to_id, vocabulary = (
        build_vocabulary(
            json_files,
            MODEL_CONFIG["vocab_size"]
        )
    )


    save_vocabulary(
        output_dir,
        token_to_id,
        vocabulary
    )


    # --------------------------------------------------------
    # TOKENSTREAM
    # --------------------------------------------------------

    token_file = (
        data_output /
        "tokens.bin"
    )


    if token_file.exists():

        print(
            "[DATA] Vorhandenes tokens.bin wird verwendet."
        )

        total_tokens = (
            token_file.stat().st_size //
            4
        )

    else:

        total_tokens = (
            build_token_stream(
                json_files,
                token_to_id,
                token_file
            )
        )


    if total_tokens < (
        args.seq_len + 10
    ):

        raise RuntimeError(
            "Nicht genügend Trainingsdaten."
        )


    # --------------------------------------------------------
    # MODEL CONFIG
    # --------------------------------------------------------

    config = dict(
        MODEL_CONFIG
    )


    config["training_seq_len"] = (
        args.seq_len
    )


    expected_parameters = (
        estimate_parameter_count(
            config
        )
    )


    save_model_config(
        output_dir,
        config,
        expected_parameters
    )


    print()
    print(
        "[MODEL] Parameter:",
        f"{expected_parameters:,}"
    )


    print(
        "[MODEL] Parameter in Millionen:",
        f"{expected_parameters / 1_000_000:.2f}M"
    )


    if args.build_only:

        print()
        print(
            "[BUILD] Datenaufbereitung fertig."
        )

        print(
            "[BUILD] Training wurde nicht gestartet."
        )

        print(
            "[BUILD] Tokenstream:",
            token_file
        )

        return


    # --------------------------------------------------------
    # TRAIN / VALIDIERUNG SPLIT
    # --------------------------------------------------------

    validation_tokens = max(

        args.seq_len + 2,

        int(
            total_tokens *
            args.val_ratio
        )

    )


    validation_tokens = min(

        validation_tokens,

        total_tokens // 5

    )


    train_end = (
        total_tokens -
        validation_tokens
    )


    validation_start = (
        train_end
    )


    train_start_min = 0

    train_start_max = (
        train_end -
        args.seq_len -
        1
    )


    validation_start_max = (
        total_tokens -
        args.seq_len -
        1
    )


    if train_start_max < 0:

        raise RuntimeError(
            "Trainingsbereich ist zu klein."
        )


    if validation_start_max < validation_start:

        raise RuntimeError(
            "Validierungsbereich ist zu klein."
        )


    print(
        "[DATA] Train-Tokens:",
        f"{train_end:,}"
    )


    print(
        "[DATA] Val-Tokens:",
        f"{total_tokens - validation_start:,}"
    )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = select_device(
        args.device
    )


    autocast_dtype = (
        select_autocast_dtype(
            device,
            args.amp
        )
    )


    print(
        "[DEVICE]",
        device
    )


    print(
        "[AMP]",
        autocast_dtype
    )


    # --------------------------------------------------------
    # MODEL ERSTELLEN
    # --------------------------------------------------------

    print()
    print(
        "[MODEL] Erzeuge 750M Transformer..."
    )


    model = LumoraTransformer(
        config
    )


    real_parameters = sum(

        parameter.numel()

        for parameter
        in model.parameters()

    )


    print(
        "[MODEL] Tatsächliche Parameter:",
        f"{real_parameters:,}"
    )


    model.to(
        device
    )


    if args.gradient_checkpointing:

        model.gradient_checkpointing = True

        print(
            "[MEMORY] Gradient checkpointing: AKTIV"
        )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.learning_rate,

        betas=(
            0.9,
            0.95
        ),

        eps=1e-8,

        weight_decay=args.weight_decay

    )


    min_lr_ratio = (

        args.min_learning_rate /
        max(
            args.learning_rate,
            1e-20
        )

    )


    min_lr_ratio = min(

        max(
            min_lr_ratio,
            0.0
        ),

        1.0

    )


    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(

            optimizer,

            lambda step:

                learning_rate_lambda(

                    step,

                    args.warmup_steps,

                    args.steps,

                    min_lr_ratio

                )

        )
    )


    scaler = None


    if (

        device.type == "cuda"

        and

        autocast_dtype ==
        torch.float16

    ):

        scaler = (
            torch.cuda.amp.GradScaler()
        )


    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    start_step = 0

    best_val_loss = float(
        "inf"
    )


    if args.resume:

        checkpoint_path = Path(
            args.resume
        )


        if not checkpoint_path.exists():

            raise FileNotFoundError(
                checkpoint_path
            )


        (
            start_step,
            best_val_loss
        ) = load_checkpoint(

            checkpoint_path,

            model,

            optimizer,

            scheduler,

            scaler,

            args.save_optimizer

        )


        print(
            "[RESUME] Step:",
            start_step
        )


    # --------------------------------------------------------
    # TOKEN MEMMAP
    # --------------------------------------------------------

    tokens = open_tokenstream(
        token_file
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()


    training_start = time.time()


    for step in range(
        start_step,
        args.steps
    ):

        step_number = (
            step + 1
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        accumulated_loss = 0.0


        # ----------------------------------------------------
        # GRADIENT ACCUMULATION
        # ----------------------------------------------------

        for micro_step in range(
            args.grad_accum
        ):

            x, y = get_batch(

                tokens,

                args.batch_size,

                args.seq_len,

                train_start_min,

                train_start_max,

                device

            )


            amp_enabled = (
                autocast_dtype is not None
            )


            with torch.autocast(

                device_type=device.type,

                dtype=autocast_dtype,

                enabled=amp_enabled

            ):

                _logits, loss = model(
                    x,
                    y
                )


                scaled_loss = (
                    loss /
                    args.grad_accum
                )


            if scaler is not None:

                scaler.scale(
                    scaled_loss
                ).backward()

            else:

                scaled_loss.backward()


            accumulated_loss += (
                loss.item()
            )


        # ----------------------------------------------------
        # GRADIENT CLIPPING
        # ----------------------------------------------------

        if scaler is not None:

            scaler.unscale_(
                optimizer
            )


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            1.0

        )


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        if scaler is not None:

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()


        scheduler.step()


        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        average_loss = (
            accumulated_loss /
            args.grad_accum
        )


        if (

            step_number == 1

            or

            step_number % 10 == 0

        ):

            elapsed = max(

                time.time() -
                training_start,

                0.001

            )


            steps_per_second = (

                (
                    step_number -
                    start_step
                ) /

                elapsed

            )


            current_lr = (
                optimizer.param_groups[0]["lr"]
            )


            print(

                "[TRAIN]",

                f"step={step_number}/{args.steps}",

                f"loss={average_loss:.5f}",

                f"lr={current_lr:.7f}",

                f"steps/s={steps_per_second:.3f}"

            )


        # ----------------------------------------------------
        # VALIDIERUNG
        # ----------------------------------------------------

        if (
            step_number %
            args.eval_every ==
            0
        ):

            validation_loss = evaluate(

                model,

                tokens,

                args.batch_size,

                args.seq_len,

                validation_start,

                validation_start_max,

                args.eval_batches,

                device,

                autocast_dtype

            )


            try:

                perplexity = math.exp(
                    min(
                        validation_loss,
                        20
                    )
                )

            except OverflowError:

                perplexity = float(
                    "inf"
                )


            print(

                "[EVAL]",

                f"step={step_number}",

                f"val_loss={validation_loss:.5f}",

                f"perplexity={perplexity:.2f}"

            )


            # ------------------------------------------------
            # BEST CHECKPOINT
            # ------------------------------------------------

            if (
                validation_loss <
                best_val_loss
            ):

                best_val_loss = (
                    validation_loss
                )


                best_state = {

                    "step":
                        step_number,

                    "model":
                        model.state_dict(),

                    "config":
                        config,

                    "best_val_loss":
                        best_val_loss

                }


                if args.save_optimizer:

                    best_state[
                        "optimizer"
                    ] = (
                        optimizer.state_dict()
                    )


                    best_state[
                        "scheduler"
                    ] = (
                        scheduler.state_dict()
                    )


                    if scaler is not None:

                        best_state[
                            "scaler"
                        ] = (
                            scaler.state_dict()
                        )


                best_path = (

                    output_dir /
                    "checkpoints" /
                    "best.pt"

                )


                best_path.parent.mkdir(

                    parents=True,

                    exist_ok=True

                )


                atomic_save(

                    best_state,

                    best_path

                )


                print(
                    "[CHECKPOINT] Neues bestes Modell:",
                    best_path
                )


        # ----------------------------------------------------
        # PERIODISCH SPEICHERN
        # ----------------------------------------------------

        if (

            step_number %
            args.save_every ==
            0

        ):

            checkpoint = save_checkpoint(

                output_dir,

                model,

                optimizer,

                scheduler,

                scaler,

                step_number,

                best_val_loss,

                args.save_optimizer

            )


            print(

                "[CHECKPOINT] Gespeichert:",

                checkpoint

            )


    # ========================================================
    # FINALES MODELL
    # ========================================================

    final_checkpoint = save_checkpoint(

        output_dir,

        model,

        optimizer,

        scheduler,

        scaler,

        args.steps,

        best_val_loss,

        args.save_optimizer

    )


    # Model-only checkpoint
    # einfacher zu transportieren

    model_only_path = (

        output_dir /
        "model_only.pt"

    )


    atomic_save(

        {

            "model":
                model.state_dict(),

            "config":
                config

        },

        model_only_path

    )


    # ========================================================
    # SERVER MANIFEST
    # ========================================================

    server_dir = (

        output_dir /
        "server_model"

    )


    server_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    manifest = {

        "model":
            "Lumora-750M",

        "architecture":
            "decoder-only-transformer",

        "from_scratch":
            True,

        "parameters":
            real_parameters,

        "parameters_millions":
            real_parameters / 1_000_000,

        "model_weights":
            "../model_only.pt",

        "latest_checkpoint":
            "../checkpoints/latest.pt",

        "best_checkpoint":
            "../checkpoints/best.pt",

        "model_config":
            "../model_config.json",

        "vocabulary":
            "../vocab.json",

        "tokenizer_config":
            "../tokenizer_config.json",

        "token_stream":
            "../data/tokens.bin"

    }


    with (
        server_dir /
        "manifest.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            manifest,

            file,

            indent=2

        )


    # ========================================================
    # ENDE
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "        LUMORA TRAINING FERTIG"
    )
    print(
        "=============================================="
    )

    print(
        "Parameter:",
        f"{real_parameters:,}"
    )

    print(
        "Final Checkpoint:",
        final_checkpoint
    )

    print(
        "Model-only:",
        model_only_path
    )

    print(
        "Beste Val-Loss:",
        f"{best_val_loss:.5f}"
    )

    print(
        "=============================================="
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
