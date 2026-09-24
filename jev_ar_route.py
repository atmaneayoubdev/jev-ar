"""Jev-AR router: self-contained inference for the Jev-AR decision model (needs torch and transformers).

    pip install torch transformers huggingface_hub
    from jev_ar_route import JevAR

    router = JevAR()  # downloads atmaneayoub/jev-ar (gated: accept the licence on its Hugging Face page first)
    router.route("ابي اجدد الاقامة")  # the built-in 60-option UAE + KSA government-services list
    router.route("وين أقرب فرع؟", {"FAQ": "General how-to questions", "Nearest branch": "Where the nearest branch is"})

route() returns the chosen route, its confidence, `escalate` (the confidence is below the model's dev-fitted
threshold, so hand the message to a person) and the full probability distribution. It raises an error rather
than silently cutting an option list that doesn't fit the model's 512-token option window.

The model weights are licensed CC BY-NC 4.0 (non-commercial; see the model card). This file is Apache-2.0.
Portions of the model and input code are adapted from the laya package (Apache-2.0); see NOTICE.
"""
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

MODEL_ID = "atmaneayoub/jev-ar"
INSTRUCTION = {"en": "Which route should this message go to?", "ar": "إلى أي مسار يجب توجيه هذه الرسالة؟"}
OPTION_CAP = 48  # tokens per option that the model reads
TEMP_MIN, TEMP_MAX = 0.5, 5.0
RE_ARABIC = re.compile("[؀-ۿ]")


class _DecisionModel(nn.Module):
    """Encoder + a small transformer head; one score per option, read at the option's marker token."""

    def __init__(self, encoder, head_layers, n_act):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))  # unused here
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, markers):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb.weight[0]  # question type 0 = choice
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        return self.scorer(h[0, markers]).squeeze(-1).float()


def _temperature(cfg, k):
    bucket = "choice:" + ("2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")
    t = cfg.get("temperature_by_options", {}).get(bucket, cfg.get("temperature", [1.0])[0])
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if t != t else min(TEMP_MAX, max(TEMP_MIN, t))


class JevAR:
    def __init__(self, model=MODEL_ID, device=None, token=None):
        """model: a Hugging Face repo id or a local model folder. token: a Hugging Face token with access to it."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        path = Path(model)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(model, token=token))
        self.cfg = json.loads((path / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(path / "tokenizer")
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(path / "encoder"), attn_implementation="sdpa")
        self.model = _DecisionModel(encoder, self.cfg.get("head_layers", 2), len(self.cfg.get("act_costs", {})) + 1)
        self.model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        self.model.encoder.config.reference_compile = False  # eager path: faster for one message at a time
        self.model.to(self.device).eval()
        self.dtype = torch.bfloat16 if self.cfg.get("amp_dtype") == "bf16" else torch.float16
        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8:
            self.dtype = torch.float16
        self.threshold = float(self.cfg.get("escalate_threshold", 0.5))
        self.gov = json.loads((path / "gov_routes.json").read_text(encoding="utf-8"))
        self.gov_by_name = {r["name_ar"]: r for r in self.gov["routes"]}

    def _encode(self, text, instruction, criteria):
        """[CLS] choice question: <instruction> [SEP] [MASK] option 1 [MASK] option 2 ... [SEP] message [SEP]."""
        tok, mask = self.tok, self.tok.mask_token
        ids_of = lambda s: tok(s.replace(mask, " "), add_special_tokens=False)["input_ids"]  # noqa: E731
        options = [k if v is None or v == "" else f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                   for k, v in criteria.items()]
        head = ids_of("choice question: " + instruction)
        opts = [[tok.mask_token_id] + ids_of(" " + o) for o in options]
        if max(len(o) for o in opts) - 1 > OPTION_CAP:
            raise ValueError(f"an option is longer than {OPTION_CAP} tokens; shorten its name or description")
        head_max_len, max_len = self.cfg.get("head_max_len", 512), self.cfg.get("max_len", 640)
        if head_max_len - sum(len(o) for o in opts) < max(16, len(head)):
            raise ValueError(f"the options take {sum(len(o) for o in opts)} tokens; the model reads at most "
                             f"{head_max_len} (instruction included). Use fewer options or shorter descriptions")
        ids, markers = [tok.cls_token_id] + head + [tok.sep_token_id], []
        for o in opts:
            markers.append(len(ids))
            ids += o
        ids.append(tok.sep_token_id)
        ids += ids_of(text)[: max(0, max_len - len(ids) - 1)] + [tok.sep_token_id]
        return ids, markers

    @torch.no_grad()
    def probabilities(self, text, criteria, instruction):
        ids, markers = self._encode(text, instruction, criteria)
        x = torch.tensor([ids], device=self.device)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            logits = self.model(x, torch.ones_like(x), torch.tensor(markers, device=self.device))
        p = torch.softmax(logits / _temperature(self.cfg, len(markers)), -1).cpu().tolist()
        return dict(zip(criteria, (round(v, 4) for v in p)))

    def route(self, text, routes=None, instruction=None):
        """routes: {name: description or None} or a list of names; None = the government list.
        instruction: the routing question (default: English or Arabic, following the route names)."""
        government = routes is None
        if government:
            criteria, instruction = {name: None for name in self.gov_by_name}, instruction or self.gov["instruction"]
        else:
            criteria = dict.fromkeys(routes) if isinstance(routes, (list, tuple)) else dict(routes)
            if len(criteria) < 2:
                raise ValueError("give at least two routes")
            instruction = instruction or INSTRUCTION["ar" if any(RE_ARABIC.search(k) for k in criteria) else "en"]
        p = self.probabilities(text, criteria, instruction)
        top = max(p, key=p.get)
        result = {"route": top, "confidence": p[top], "escalate": p[top] < self.threshold,
                  "probabilities": dict(sorted(p.items(), key=lambda kv: -kv[1]))}
        if government:
            g = self.gov_by_name[top]
            not_gov = next(r["name_ar"] for r in self.gov["routes"] if r["id"] == "not_government")
            result.update(id=g["id"], category=g["category"], in_scope=round(1.0 - p[not_gov], 4),
                          escalate=result["escalate"] or g["id"] == "other_government")
        return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Route one message with Jev-AR")
    ap.add_argument("text")
    ap.add_argument("--routes", help='JSON {"name": "description", ...}; default: the government list')
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    res = JevAR(args.model).route(args.text, json.loads(args.routes) if args.routes else None)
    res["probabilities"] = dict(list(res["probabilities"].items())[:5])
    print(json.dumps(res, ensure_ascii=False, indent=1))
