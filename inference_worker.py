import sys
import json
import math
import re
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LUMORA 750M INFERENCE WORKER
# ============================================================

MODEL_DIR = Path(
    "LUMORA_MODEL"
)

MODEL_FILE = (
    MODEL_DIR /
    "model_only.pt"
)

VOCAB_FILE = (
    MODEL_DIR /
    "vocab.json"
)


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


def normalize_text(text):

    text = str(text or "")

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    return text.lower().strip()


def tokenize(text):

    return TOKEN_PATTERN.findall(
        normalize_text(text)
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


    def forward(self, x):

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

        return self.weight * x


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
# ATTENTION
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


    def forward(self, x):

        batch, sequence, hidden = (
            x.shape
        )

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
        ).transpose(1, 2)

        k = k.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        q, k = self.rope.apply(
            q,
            k,
            sequence
        )

        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True
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


    def forward(self, x):

        return self.down(
            F.silu(
                self.gate(x)
            ) *
            self.up(x)
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


    def forward(self, x):

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
# MODEL
# ============================================================

class LumoraTransformer(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()

        self.config = config

        vocab = (
            config["vocab_size"]
        )

        hidden = (
            config["hidden_size"]
        )

        self.embedding = nn.Embedding(
            vocab,
            hidden
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config
                )
                for _ in range(
                    config["num_layers"]
                )
            ]
        )

        self.final_norm = RMSNorm(
            hidden
        )

        self.lm_head = nn.Linear(
            hidden,
            vocab,
            bias=False
        )

        self.lm_head.weight = (
            self.embedding.weight
        )


    def forward(
        self,
        input_ids
    ):

        x = self.embedding(
            input_ids
        )

        for block in self.blocks:

            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# LOAD
# ============================================================

print(
    "[WORKER] Lade Lumora-Modell...",
    file=sys.stderr
)


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Modell nicht gefunden: {MODEL_FILE}"
    )


if not VOCAB_FILE.exists():

    raise FileNotFoundError(
        f"Vokabular nicht gefunden: {VOCAB_FILE}"
    )


with VOCAB_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    vocab_data = json.load(file)


VOCAB = vocab_data["vocab"]

TOKEN_TO_ID = {
    token: index
    for index, token
    in enumerate(VOCAB)
}


ID_TO_TOKEN = {
    index: token
    for index, token
    in enumerate(VOCAB)
}


CHECKPOINT = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)


CONFIG = CHECKPOINT["config"]


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "[WORKER] Device:",
    DEVICE,
    file=sys.stderr
)


MODEL = LumoraTransformer(
    CONFIG
)


MODEL.load_state_dict(
    CHECKPOINT["model"]
)


MODEL.to(
    DEVICE
)


MODEL.eval()


print(
    "[WORKER] Lumora 750M bereit.",
    file=sys.stderr
)


# ============================================================
# TOKEN <-> ID
# ============================================================

def text_to_ids(text):

    tokens = tokenize(text)

    unk_id = TOKEN_TO_ID.get(
        "<unk>",
        3
    )

    bos_id = TOKEN_TO_ID.get(
        "<bos>",
        1
    )

    ids = [bos_id]

    for token in tokens:

        ids.append(
            TOKEN_TO_ID.get(
                token,
                unk_id
            )
        )

    return ids


def ids_to_text(ids):

    result = []

    for token_id in ids:

        token = ID_TO_TOKEN.get(
            int(token_id),
            "<unk>"
        )

        if token in (
            "<bos>",
            "<pad>"
        ):

            continue

        if token == "<eos>":
            break

        if token.startswith("<extra_"):
            continue

        result.append(token)


    text = ""

    for token in result:

        if not text:

            text = token

        elif token in (
            ",",
            ".",
            "!",
            "?",
            ":",
            ";",
            "%",
            ")",
            "]",
            "}"
        ):

            text += token

        else:

            text += " " + token


    return text


# ============================================================
# TOP-K
# ============================================================

def sample_top_k(
    logits,
    k
):

    if k <= 0:

        return torch.argmax(
            logits,
            dim=-1
        )


    values, indices = torch.topk(
        logits,
        min(
            k,
            logits.shape[-1]
        )
    )


    probabilities = F.softmax(
        values,
        dim=-1
    )


    selected = torch.multinomial(
        probabilities,
        num_samples=1
    )


    return indices.gather(
        -1,
        selected
    ).squeeze(-1)


# ============================================================
# TOP-P
# ============================================================

def sample_top_p(
    logits,
    p
):

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    sorted_probabilities, sorted_indices = (
        torch.sort(
            probabilities,
            descending=True
        )
    )

    cumulative = torch.cumsum(
        sorted_probabilities,
        dim=-1
    )

    remove = (
        cumulative -
        sorted_probabilities
    ) > p

    sorted_probabilities[
        remove
    ] = 0

    sorted_probabilities = (
        sorted_probabilities /
        sorted_probabilities.sum(
            dim=-1,
            keepdim=True
        )
    )

    selected = torch.multinomial(
        sorted_probabilities,
        1
    )

    return (
        sorted_indices
        .gather(
            -1,
            selected
        )
        .squeeze(-1)
    )


# ============================================================
# GENERATE
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
    top_p=0.9
):

    token_ids = text_to_ids(
        prompt
    )


    max_length = (
        CONFIG["max_seq_len"]
    )


    for _ in range(
        max_new_tokens
    ):

        context = token_ids[
            -max_length:
        ]


        input_ids = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE
        )


        logits = MODEL(
            input_ids
        )


        next_logits = (
            logits[0, -1, :]
        )


        if temperature <= 0:

            next_id = (
                torch.argmax(
                    next_logits
                )
                .item()
            )

        else:

            next_logits = (
                next_logits /
                temperature
            )


            if top_k > 0:

                top_k_values, top_k_indices = (
                    torch.topk(
                        next_logits,
                        min(
                            top_k,
                            next_logits.shape[-1]
                        )
                    )
                )

                filtered = torch.full_like(
                    next_logits,
                    float("-inf")
                )

                filtered[
                    top_k_indices
                ] = top_k_values

                next_logits = filtered


            if top_p < 1.0:

                next_id = sample_top_p(
                    next_logits,
                    top_p
                ).item()

            else:

                next_id = sample_top_k(
                    next_logits,
                    0
                ).item()


        token_ids.append(
            next_id
        )


        if next_id == TOKEN_TO_ID.get(
            "<eos>",
            2
        ):

            break


    generated_ids = token_ids[
        len(text_to_ids(prompt)):]
    ]


    return ids_to_text(
        generated_ids
    )


# ============================================================
# JSONL SERVER
# ============================================================

for line in sys.stdin:

    line = line.strip()

    if not line:
        continue


    try:

        request = json.loads(
            line
        )

        request_id = (
            request.get("id")
        )

        action = (
            request.get("action")
        )


        if action == "health":

            response = {

                "id":
                    request_id,

                "ok":
                    True,

                "model":
                    CONFIG.get(
                        "model_name",
                        "Lumora-750M"
                    ),

                "device":
                    str(DEVICE),

                "parameters":
                    sum(
                        p.numel()
                        for p in MODEL.parameters()
                    )

            }


        elif action == "generate":

            prompt = str(
                request.get(
                    "text",
                    ""
                )
            )


            if not prompt.strip():

                raise ValueError(
                    "Leerer Prompt."
                )


            text = generate(

                prompt,

                max_new_tokens=int(
                    request.get(
                        "max_new_tokens",
                        150
                    )
                ),

                temperature=float(
                    request.get(
                        "temperature",
                        0.8
                    )
                ),

                top_k=int(
                    request.get(
                        "top_k",
                        40
                    )
                ),

                top_p=float(
                    request.get(
                        "top_p",
                        0.9
                    )
                )
            )


            response = {

                "id":
                    request_id,

                "ok":
                    True,

                "text":
                    text

            }


        else:

            raise ValueError(
                "Unbekannte Aktion."
            )


    except Exception as error:

        response = {

            "id":
                request_id
                if "request_id"
                in locals()
                else None,

            "ok":
                False,

            "error":
                str(error)

        }


    print(
        json.dumps(
            response,
            ensure_ascii=False
        ),
        flush=True
    )
