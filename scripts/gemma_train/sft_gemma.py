#!/usr/bin/env python3
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
import requests
from datasets import load_dataset, Dataset as HFDataset
from transformers import TrainerCallback

# Project import
import sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from theworld import TheWorld  # <- use .from_pretrained()

log = logging.getLogger("sft_theworld_binary_flattened")
logging.basicConfig(level=logging.WARNING)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("SFT TheWorld Gemma-only (on-the-fly, flattened, batched-transform)")

    # Core
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--dataset_format", type=str, default=None)
    p.add_argument("--system_prompt", type=str,
                   default=("You are a vision-language model that performs binary visual entailment.\n"
                            "Task: Look at the image and evaluate the statement.\n"
                            "Output exactly one digit with no extra text: 1 if the statement is true, 0 if the statement is false."))
    p.add_argument("--image_key", type=str, default="image")
    p.add_argument("--question_key", type=str, default="caption")
    p.add_argument("--answer_key", type=str, default="label")

    # IO / training
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./sft_out_binary_vsr")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--reserve_assistant_tokens", type=int, default=24)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"])
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--dataloader_pin_memory", action="store_true")

    # Freezing knobs (forwarded to TheWorld.from_pretrained)
    p.add_argument("--freeze_gemma_vision", action="store_true")
    p.add_argument("--freeze_gemma_language", action="store_true")

    # Debug knobs
    p.add_argument("--debug_print_n", type=int, default=0)
    p.add_argument("--debug_print_decode", action="store_true")
    p.add_argument("--logit_probe_every", type=int, default=200)
    p.add_argument("--logit_probe_samples", type=int, default=6)

    # Loader knobs to match evaluation
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return p.parse_args()

# ---------------- Label parsing ----------------
_WORD_TO_BIN = {
    "1": 1, "0": 0,
    "true": 1, "false": 0,
    "yes": 1, "no": 0,
    "correct": 1, "incorrect": 0,
    "right": 1, "wrong": 0,
    "entails": 1, "contradiction": 0,
}
def parse_label_to_digit(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, list) and len(val) > 0:
        val = val[0]
    try:
        s = str(val).strip()
    except Exception:
        return None
    if re.fullmatch(r"[01]", s):
        return int(s)
    tok = re.sub(r"[^A-Za-z]", " ", s).strip().lower()
    if tok in _WORD_TO_BIN:
        return _WORD_TO_BIN[tok]
    try:
        v = int(s)
        return v if v in (0, 1) else None
    except Exception:
        return None

# ---------------- Images ----------------
def _try_open_url(url: str, timeout: float = 15.0) -> Image.Image:
    import requests, io
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def load_pil_image(val: Any, image_dir: Optional[Path]) -> Optional[Image.Image]:
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, (str, Path)):
        s = str(val)
        if s.startswith("http://") or s.startswith("https://"):
            return _try_open_url(s, timeout=15.0)
        p = Path(s)
        if p.is_absolute() and p.exists():
            return Image.open(p).convert("RGB")
        if image_dir is not None:
            cand = image_dir / p
            if cand.exists():
                return Image.open(cand).convert("RGB")
        if p.exists():
            return Image.open(p).convert("RGB")
    return None

# ---------------- Flatten helpers ----------------
def _column_is_listlike(ds: HFDataset, key: str, check_n: int = 50) -> bool:
    n = min(len(ds), check_n)
    for i in range(n):
        v = ds[i].get(key, None)
        if isinstance(v, list):
            return True
    return False

def flatten_vsr(ds: HFDataset, image_key: str, question_key: str, answer_key: str) -> HFDataset:
    qs_list = _column_is_listlike(ds, question_key)
    ys_list = _column_is_listlike(ds, answer_key)
    log.info("[FLATTEN] question list-like? %s  answer list-like? %s", qs_list, ys_list)
    if not (qs_list and ys_list):
        log.info("[FLATTEN] Dataset appears scalar already; no flattening performed.")
        return ds

    log.info("[FLATTEN] Exploding list-valued rows into single-statement examples...")

    def _explode(example):
        img = example.get(image_key)
        qs  = example.get(question_key)
        ys  = example.get(answer_key)
        if not isinstance(qs, list) or not isinstance(ys, list):
            return {image_key: [img], question_key: [qs], answer_key: [ys]}
        n = min(len(qs), len(ys))
        return {
            image_key: [img] * n,
            question_key: qs[:n],
            answer_key: ys[:n],
        }

    exploded = ds.map(_explode, remove_columns=ds.column_names, batched=False)

    images_flat: List[Any] = []
    questions_flat: List[Any] = []
    answers_flat: List[Any] = []

    for row_idx in range(len(exploded)):
        images_flat.extend(exploded[image_key][row_idx])
        questions_flat.extend(exploded[question_key][row_idx])
        answers_flat.extend(exploded[answer_key][row_idx])

    new_ds = HFDataset.from_dict({
        image_key: images_flat,
        question_key: questions_flat,
        answer_key: answers_flat,
    })
    log.info("[FLATTEN] Done. Old size=%d -> New size=%d", len(ds), len(new_ds))
    return new_ds

# ---------------- Normalization helpers ----------------
def _squash1(x: Any) -> Any:
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        y = x.strip()[1:-1].strip().strip("'").strip('"')
        try:
            return int(y)
        except Exception:
            return y
    return x

# ---------------- Pad token helper ----------------
def _ensure_pad_token(processor, model):
    if processor.tokenizer.pad_token_id is None:
        if processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        else:
            processor.tokenizer.add_special_tokens({"pad_token": " "})
            try:
                model.resize_token_embeddings(len(processor.tokenizer))
            except Exception:
                pass

# ---------------- Digit token helpers ----------------
def get_digit_token_ids(tokenizer) -> Tuple[Optional[int], Optional[int]]:
    one_ids = tokenizer.encode("1", add_special_tokens=False)
    zero_ids = tokenizer.encode("0", add_special_tokens=False)
    one_tid = one_ids[0] if one_ids else None
    zero_tid = zero_ids[0] if zero_ids else None
    return zero_tid, one_tid

# ---------------- Training example builder ----------------
def _force_keep_answer(processor, input_ids, attention_mask, labels, answer_text: str, max_length: int):
    tok = processor.tokenizer
    ans_ids = tok.encode(str(answer_text), add_special_tokens=False)
    ans_ids = torch.tensor(ans_ids, dtype=input_ids.dtype)

    need = ans_ids.numel()
    room = max_length - input_ids.numel()
    if room < need:
        drop = need - room
        if drop >= input_ids.numel():
            pad_id = tok.pad_token_id or (tok.eos_token_id if tok.eos_token_id is not None else 0)
            input_ids = torch.tensor([pad_id], dtype=input_ids.dtype)
            attention_mask = torch.tensor([1], dtype=attention_mask.dtype)
            labels = torch.tensor([-100], dtype=labels.dtype)
        else:
            input_ids = input_ids[drop:]
            attention_mask = attention_mask[drop:]
            labels = labels[drop:]

    input_ids = torch.cat([input_ids, ans_ids], dim=0)[:max_length]
    app_attn = torch.ones((need,), dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, app_attn], dim=0)[:max_length]
    app_labels = ans_ids.clone()
    labels = torch.cat([labels, app_labels], dim=0)[:max_length]
    return input_ids, attention_mask, labels

def _guarantee_min_len_two(processor, input_ids, attention_mask, labels):
    if input_ids.numel() >= 2:
        return input_ids, attention_mask, labels
    tok = processor.tokenizer
    eos = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
    input_ids = torch.cat([input_ids, torch.tensor([eos], dtype=input_ids.dtype)], dim=0)
    attention_mask = torch.cat([attention_mask, torch.tensor([1], dtype=attention_mask.dtype)], dim=0)
    labels = torch.cat([labels, torch.tensor([-100], dtype=labels.dtype)], dim=0)
    return input_ids, attention_mask, labels

def build_one_example(processor,
                      system_prompt: str,
                      statement: str,
                      answer_digit: int,
                      pil_image: Optional[Image.Image],
                      max_length: int,
                      reserve_assistant_tokens: int = 24,
                      debug_print_n: int = 0,
                      debug_print_decode: bool = False,
                      debug_counter: Dict[str, int] = None) -> Dict[str, List[int]]:
    user_text = f"Statement: {statement}\nAnswer (only '1' or '0'):"

    prompt_msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content":
            ( [{"type": "image", "image": pil_image}] if pil_image is not None else [] )
            + [{"type": "text", "text": user_text}]}
    ]
    full_msgs = prompt_msgs + [
        {"role": "assistant", "content": [{"type": "text", "text": str(int(answer_digit))}]}
    ]

    tok = processor.tokenizer
    old_side = getattr(tok, "truncation_side", "right")
    tok.truncation_side = "left"
    try:
        prompt_max = max(8, max_length - max(4, reserve_assistant_tokens))
        enc_prompt = processor.apply_chat_template(
            prompt_msgs,
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            truncation=True, max_length=prompt_max,
        )
        enc_full = processor.apply_chat_template(
            full_msgs,
            tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
            truncation=True, max_length=max_length,
        )
    finally:
        tok.truncation_side = old_side

    input_ids = enc_full["input_ids"][0]
    attn = enc_full["attention_mask"][0]
    prompt_len = enc_prompt["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    ans_txt = str(int(answer_digit))
    ans_ids = processor.tokenizer.encode(ans_txt, add_special_tokens=False)
    digit_tid = ans_ids[0] if len(ans_ids) > 0 else None

    first_digit_pos = None
    for j in range(prompt_len, input_ids.size(0)):
        if digit_tid is not None and int(input_ids[j].item()) == digit_tid:
            first_digit_pos = j
            break

    labels[prompt_len:] = -100
    if first_digit_pos is not None:
        labels[first_digit_pos] = input_ids[first_digit_pos]
    else:
        input_ids, attn, labels = _force_keep_answer(
            processor, input_ids, attn, labels, answer_text=ans_txt, max_length=max_length
        )

    input_ids, attn, labels = _guarantee_min_len_two(processor, input_ids, attn, labels)

    if debug_counter is not None and debug_counter.get("n", 0) < debug_print_n:
        ex_no = debug_counter["n"] + 1
        gt_ids = [int(t) for t in labels.tolist() if t != -100]
        log.warning(f"[EXDBG {ex_no}] label={answer_digit} digit_tid={digit_tid} prompt_len={prompt_len} "
                    f"first_digit_pos={first_digit_pos} supervised_ids={gt_ids}")
        if debug_print_decode:
            try:
                tail_k = 120
                prompt_tail = processor.tokenizer.decode(enc_prompt["input_ids"][0][-tail_k:], skip_special_tokens=False)
                full_tail = processor.tokenizer.decode(input_ids[-tail_k:], skip_special_tokens=False)
                sup_txt = processor.tokenizer.decode(torch.tensor(gt_ids), skip_special_tokens=False) if gt_ids else ""
                log.warning(f"[EXDBG {ex_no}] prompt_tail={repr(prompt_tail)}")
                log.warning(f"[EXDBG {ex_no}] full_tail={repr(full_tail)}")
                log.warning(f"[EXDBG {ex_no}] supervised_decoded={repr(sup_txt)}")
            except Exception as e:
                log.warning(f"[EXDBG {ex_no}] decode failed: {e}")
        debug_counter["n"] = ex_no

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attn.tolist(),
        "labels": labels.tolist(),
        # NOTE: If you want to train with vision features, you’ll add pixel_values here.
    }

# ---------------- Collator ----------------
class BinaryVSRCollator:
    def __init__(self, pad_token_id: Optional[int], eos_token_id: Optional[int]):
        self.pad = pad_token_id if pad_token_id is not None else 0
        self.eos = eos_token_id if eos_token_id is not None else self.pad

    def _pad_1d(self, seqs, pad_val: int) -> torch.Tensor:
        _norm: List[List[int]] = []
        for s in seqs:
            if isinstance(s, torch.Tensor):
                s = s.tolist()
            _norm.append(s)
        maxlen = max((len(s) for s in _norm), default=0)
        out = torch.full((len(_norm), maxlen), pad_val, dtype=torch.long)
        for i, s in enumerate(_norm):
            if s:
                out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {
                "input_ids": torch.empty((0, 2), dtype=torch.long),
                "attention_mask": torch.empty((0, 2), dtype=torch.long),
                "labels": torch.empty((0, 2), dtype=torch.long),
            }

        input_ids = [b["input_ids"] for b in batch]
        attn      = [b["attention_mask"] for b in batch]
        labels    = [b["labels"] for b in batch]

        X = self._pad_1d(input_ids, self.pad)
        A = self._pad_1d(attn, 0)
        Y = self._pad_1d(labels, -100)

        # Safety: ensure at least one supervised token
        for i in range(X.shape[0]):
            sup_count = torch.count_nonzero(Y[i] != -100).item()
            if sup_count == 0:
                actual_len = int(A[i].sum().item())
                if actual_len > 1:
                    pos = actual_len - 2
                elif actual_len == 1:
                    pos = 0
                else:
                    log.error(f"[COLLATOR] Row {i} has zero-length attention mask!")
                    continue
                Y[i, pos] = X[i, pos]
                log.warning(f"[COLLATOR] Forced supervision on row {i} at position {pos} (token {int(X[i, pos].item())})")

        if X.shape[1] < 2:
            extra_x = torch.full((X.shape[0], 1), self.eos, dtype=X.dtype)
            extra_a = torch.ones((A.shape[0], 1), dtype=A.dtype)
            extra_y = torch.full((Y.shape[0], 1), -100, dtype=Y.dtype)
            X = torch.cat([X, extra_x], dim=1)
            A = torch.cat([A, extra_a], dim=1)
            Y = torch.cat([Y, extra_y], dim=1)

        return {"input_ids": X, "attention_mask": A, "labels": Y}

# ---------------- Trainer callback: logits probe ----------------
class LogitProbeCallback(TrainerCallback):
    def __init__(self, tokenizer, probe_every: int, sample_k: int, zero_tid: Optional[int], one_tid: Optional[int]):
        self.tok = tokenizer
        self.probe_every = max(0, probe_every)
        self.sample_k = max(0, sample_k)
        self.zero_tid = zero_tid
        self.one_tid = one_tid

    def on_train_begin(self, args, state, control, **kwargs):
        log.warning(f"[PROBE] Digit token ids -> '0': {self.zero_tid}, '1': {self.one_tid}")

    def on_step_end(self, args, state, control, **kwargs):
        return control

    def on_train_batch_end(self, args, state, control, **kwargs):
        if self.probe_every == 0:
            return control
        if (state.global_step % self.probe_every) != 0:
            return control

        model = kwargs["model"]
        inputs = kwargs["inputs"]
        if not all(k in inputs for k in ("input_ids", "attention_mask", "labels")):
            return control

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        except Exception as e:
            log.warning(f"[PROBE] forward failed: {e}")
            return control

        sup_mask = (labels != -100)
        n_sup = int(sup_mask.sum().item())
        if n_sup == 0:
            log.warning("[PROBE] No supervised tokens in this batch.")
            return control

        gold_ids = labels[sup_mask]
        pred_ids = logits[sup_mask].argmax(dim=-1)

        acc = float((pred_ids == gold_ids).float().mean().item())

        def _count_eq(t, val):
            if val is None:
                return 0
            return int((t == val).sum().item())

        gold_zeros = _count_eq(gold_ids, self.zero_tid)
        gold_ones  = _count_eq(gold_ids, self.one_tid)
        pred_zeros = _count_eq(pred_ids, self.zero_tid)
        pred_ones  = _count_eq(pred_ids, self.one_tid)

        log.warning(f"[PROBE step {state.global_step}] supervised_positions={n_sup} acc={acc:.3f} | "
                    f"gold: '0'={gold_zeros}, '1'={gold_ones} | pred: '0'={pred_zeros}, '1'={pred_ones}")

        if self.sample_k > 0:
            try:
                sample_idx = torch.randperm(n_sup)[:self.sample_k]
                g_s = gold_ids[sample_idx].tolist()
                p_s = pred_ids[sample_idx].tolist()
                pairs = []
                for g, p in zip(g_s, p_s):
                    pairs.append((self.tok.decode([g], skip_special_tokens=False),
                                  self.tok.decode([p], skip_special_tokens=False)))
                log.warning(f"[PROBE step {state.global_step}] examples (gold -> pred): {pairs}")
            except Exception as e:
                log.warning(f"[PROBE] sampling decode failed: {e}")

        return control

# ---------------- Main ----------------
def _str_to_dtype(s: str):
    if s == "float32":
        return torch.float32
    if s == "float16":
        return torch.float16
    return torch.bfloat16

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.debug_print_n > 0 or args.logit_probe_every > 0:
        logging.getLogger().setLevel(logging.INFO)
        log.setLevel(logging.INFO)

    # Load splits
    if Path(args.dataset).exists():
        fmt = args.dataset_format or "json"
        loaded = load_dataset(fmt, data_files={args.split: args.dataset})
        train_ds: HFDataset = loaded[args.split]
        eval_ds: Optional[HFDataset] = None
        if args.eval_split:
            log.warning("Local file mode: --eval_split ignored unless provided separately.")
    else:
        train_ds = load_dataset(args.dataset, split=args.split)
        eval_ds = None
        if args.eval_split:
            try:
                eval_ds = load_dataset(args.dataset, split=args.eval_split)
            except Exception:
                log.warning(f"Eval split '{args.eval_split}' not found; proceeding without eval.")

    # Flatten if needed
    train_ds = flatten_vsr(train_ds, args.image_key, args.question_key, args.answer_key)
    if eval_ds is not None:
        eval_ds = flatten_vsr(eval_ds, args.image_key, args.question_key, args.answer_key)

    log.info(f"Train split='{args.split}' size={len(train_ds)} cols={train_ds.column_names}")
    if eval_ds is not None:
        log.info(f"Eval  split='{args.eval_split}' size={len(eval_ds)} cols={eval_ds.column_names}")

    # -------- NEW: Load TheWorld (Gemma-only by default) --------
    tw = TheWorld.from_pretrained(
        args.model_name_or_path,
        enable_world=False,                # SFT baseline: Gemma-only
        device=args.device,
        load_full_cosmos_pipeline=False,
        dtype=_str_to_dtype(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        torch_dtype=_str_to_dtype(args.dtype),
        freeze_gemma_vision=args.freeze_gemma_vision,
        freeze_gemma_language=args.freeze_gemma_language,
    )
    processor = tw.processor
    _ensure_pad_token(processor, tw)

    # Digit token ids
    zero_tid, one_tid = get_digit_token_ids(processor.tokenizer)
    log.warning(f"[DIGITS] tokenizer ids -> '0': {zero_tid}, '1': {one_tid}")

    # Dataset label histogram (pre-tokenization)
    def _dataset_label_hist(ds: HFDataset, ak: str) -> Tuple[int, int, int]:
        n0 = n1 = n_bad = 0
        for i in range(min(len(ds), 20000)):
            ans = parse_label_to_digit(ds[i].get(ak))
            if ans is None:
                n_bad += 1
            elif ans == 0:
                n0 += 1
            elif ans == 1:
                n1 += 1
        return n0, n1, n_bad
    n0, n1, nbad = _dataset_label_hist(train_ds, args.answer_key)
    log.warning(f"[DATA LABELS] train distribution -> 0: {n0}, 1: {n1}, unparsable: {nbad}")

    # Trainable params info (TheWorld is the model)
    trainable_params = sum(p.numel() for p in tw.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in tw.parameters())
    log.info(f"[MODEL] Trainable parameters: {trainable_params} / {total_params} "
             f"({(trainable_params / max(1,total_params))*100:.2f}%)")
    if trainable_params == 0:
        log.error("No trainable parameters! Check freeze flags.")

    image_dir = Path(args.image_dir) if args.image_dir else None
    ik, qk, ak = args.image_key, args.question_key, args.answer_key

    debug_counter = {"n": 0}

    # -------- Batched transform (dict-of-lists in, dict-of-lists out) --------
    def to_features(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        n = len(batch[ik])
        out_ids: List[List[int]] = []
        out_attn: List[List[int]] = []
        out_lbls: List[List[int]] = []

        for i in range(n):
            img_val = _squash1(batch[ik][i])
            lbl_val = _squash1(batch[ak][i])
            cap_val = _squash1(batch[qk][i])

            ans = parse_label_to_digit(lbl_val)
            if ans is None:
                ans = 0
                log.warning(f"Invalid label parsed as 0: {lbl_val}")

            statement = "" if cap_val is None else str(cap_val)

            pil = None
            try:
                pil = load_pil_image(img_val, image_dir)
                if pil is None:
                    log.warning(f"Failed to load image: {img_val}")
            except Exception as e:
                log.warning(f"Exception loading image {img_val}: {e}")
                pil = None

            built = build_one_example(
                processor=processor,
                system_prompt=args.system_prompt,
                statement=statement,
                answer_digit=int(ans),
                pil_image=pil,
                max_length=args.max_length,
                reserve_assistant_tokens=args.reserve_assistant_tokens,
                debug_print_n=args.debug_print_n,
                debug_print_decode=args.debug_print_decode,
                debug_counter=debug_counter,
            )

            if len(built["input_ids"]) < 2:
                pad_id = processor.tokenizer.pad_token_id or 0
                eos_id = processor.tokenizer.eos_token_id or pad_id
                built["input_ids"] = [pad_id, eos_id]
                built["attention_mask"] = [1, 1]
                built["labels"] = [-100, built["input_ids"][-1]]

            out_ids.append(built["input_ids"])
            out_attn.append(built["attention_mask"])
            out_lbls.append(built["labels"])

        return {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_lbls}
        # NOTE: If you want to also train the visual encoder, you can extend this to return
        # something like {"pixel_values": ...} and have the collator stack those and pass to the model.

    # Use set_transform so HF always gives us batched dict-of-lists
    train_ds = train_ds.with_format(type=None)  # raw python objects
    train_ds.set_transform(to_features)
    if eval_ds is not None:
        eval_ds = eval_ds.with_format(type=None)
        eval_ds.set_transform(to_features)

    # Trainer
    try:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False
    use_fp16 = torch.cuda.is_available() and not use_bf16

    from transformers import TrainingArguments, Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        logging_steps=100,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
    )

    collator = BinaryVSRCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # One-probe sanity
    try:
        probe_batch = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=2, collate_fn=collator)))
        log.info("[PROBE INIT] X=%s A=%s Y=%s supervised=%d",
                 tuple(probe_batch["input_ids"].shape),
                 tuple(probe_batch["attention_mask"].shape),
                 tuple(probe_batch["labels"].shape),
                 int((probe_batch["labels"] != -100).sum().item()))
    except Exception as e:
        log.error(f"[PROBE INIT] failed: {e}")

    trainer = Trainer(
        model=tw,                       # ← Use TheWorld directly
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[
            LogitProbeCallback(
                tokenizer=processor.tokenizer,
                probe_every=args.logit_probe_every,
                sample_k=args.logit_probe_samples,
                zero_tid=zero_tid,
                one_tid=one_tid,
            )
        ] if args.logit_probe_every > 0 else None,
    )

    trainer.train()

    if eval_ds is not None:
        try:
            metrics = trainer.evaluate()
            log.info(f"Validation metrics: {metrics}")
        except Exception as e:
            log.warning(f"Validation evaluate() failed: {e}")

    # This calls TheWorld.save_pretrained (trainable-only), as you implemented.
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log.info("✓ Training complete. Saved to %s", args.output_dir)

if __name__ == "__main__":
    main()
