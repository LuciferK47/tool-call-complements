"""Batched Hugging Face runner: greedy generation + prefill hidden states + token log-probs in ONE pass.

For each prompt it returns, from the same forward pass that produces the answer:
  * `hidden[layer]`  last-prompt-token hidden state (float16 numpy) for each requested layer,
  * the greedy continuation text, token ids, per-token log-probs and the first-token entropy.
Device is chosen from the config (`auto` = cuda if available else cpu); nothing here is Linux- or Windows-specific.
Prompts are chat-templated with the generation prompt appended, so the hidden state is the model's state
immediately before it would start answering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class NodeOutput:
    text: str
    n_tokens: int
    mean_logprob: float
    first_token_entropy: float
    n_prompt_tokens: int = 0
    hidden: Dict[int, np.ndarray] = field(default_factory=dict)
    # S3 early drafting signals: up to first 2 generated tokens
    gen_token_logprobs: List[float] = field(default_factory=list)
    gen_token_hidden: Dict[int, List[np.ndarray]] = field(default_factory=dict)


class HFRunner:
    SYSTEM = "Answer with a number."   # short on purpose: Qwen's default system prompt costs ~30 extra tokens/prompt

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto",
                 dtype: str = "float32", revision: Optional[str] = None, threads: Optional[int] = None,
                 system: Optional[str] = None):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if threads:
            torch.set_num_threads(threads)
        self.device = device
        self.model_id = model_id
        td = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        self.dtype_name = dtype
        if device == "cuda":                      # exact fp32 matmuls: no TF32, so fp32 runs are comparable with CPU runs
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = self._load(model_id, revision, td).to(device).eval()
        tcfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        self.num_layers = tcfg.num_hidden_layers          # text decoder only (multimodal checkpoints nest it)
        self.hidden_dim = tcfg.hidden_size
        self.revision = revision
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.system = self.SYSTEM if system is None else system
        gc = getattr(self.model, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None)
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        self.stop_ids = {i for i in eos + [self.tok.eos_token_id, self.tok.pad_token_id] if i is not None}

    @staticmethod
    def _load(model_id, revision, td):
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=td)
        except Exception as first:                # multimodal checkpoints (e.g. Qwen3.5) may need their own auto class; inputs stay text-only
            import transformers
            for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
                cls = getattr(transformers, name, None)
                if cls is None:
                    continue
                try:
                    return cls.from_pretrained(model_id, revision=revision, dtype=td)
                except Exception:
                    continue
            raise first

    def env_info(self) -> dict:
        info = {"device": self.device, "dtype": self.dtype_name, "resolved_revision": self.resolved_revision, "torch": torch.__version__}
        gc = getattr(self.model, "generation_config", None)
        # defaults inherited from the checkpoint's generation_config.json (e.g. Qwen2.5 ships repetition_penalty 1.1, applied even when greedy; Qwen3.5 ships none)
        info["generation_defaults"] = {k: getattr(gc, k, None) for k in ("repetition_penalty", "temperature", "top_p", "top_k", "do_sample")} if gc is not None else {}
        if self.device == "cuda":
            info.update({"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)})
        return info

    # ---- prompt formatting -------------------------------------------------------------------------
    def format(self, user_text: str) -> str:
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": user_text}]
        try:                                      # thinking models: switch thinking OFF (a no-op for templates that ignore the flag)
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- core call -----------------------------------------------------------------------------------
    @torch.inference_mode()
    def run(self, prompts: Sequence[str], layers: Sequence[int], max_new_tokens: int = 12,
            batch_size: int = 16) -> List[NodeOutput]:
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))   # minimise padding
        sorted_outs: List[NodeOutput] = []
        outs = sorted_outs
        for s in range(0, len(prompts), batch_size):
            chunk = [self.format(prompts[i]) for i in order[s:s + batch_size]]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id, return_dict_in_generate=True, output_scores=True,
                output_hidden_states=bool(layers))
            seq = gen.sequences[:, enc["input_ids"].shape[1]:]
            # Memory-efficient step-wise log-softmax (avoids allocating full (B, n_new, V) buffer at once)
            tok_logp_cols = []
            ent0 = torch.zeros(seq.shape[0], device=self.device)
            for t_idx, step_score in enumerate(gen.scores):
                step_lp = torch.log_softmax(step_score.float(), dim=-1)
                tok_logp_cols.append(step_lp.gather(-1, seq[:, t_idx:t_idx + 1]))
                if t_idx == 0:
                    ent0 = -(step_lp.exp() * step_lp).sum(-1)
            tok_logp = torch.cat(tok_logp_cols, dim=1) if tok_logp_cols else torch.zeros((seq.shape[0], 0), device=self.device)
            prompt_hs = gen.hidden_states[0] if layers else None        # tuple(len=L+1) of (B, S, H); layers=[] skips hidden states
            for b in range(seq.shape[0]):
                ids = seq[b].tolist()
                n = len(ids)
                for k, t in enumerate(ids):                              # stop at first EOS / pad
                    if t in self.stop_ids:
                        n = k
                        break
                text = self.tok.decode(ids[:n], skip_special_tokens=True).strip()
                mlp = float(tok_logp[b, :max(n, 1)].mean().item())
                hid = {l: prompt_hs[l][b, -1, :].float().cpu().numpy().astype(np.float16) for l in layers} if layers else {}
                # Cache token 1 and 2 early signals (S3)
                gt_logp = [float(tok_logp[b, step_idx].item()) for step_idx in range(min(2, n))]
                gt_hid = {l: [gen.hidden_states[1 + step_idx][l][b, -1, :].float().cpu().numpy().astype(np.float16)
                              for step_idx in range(min(2, n, len(gen.hidden_states) - 1))]
                           for l in layers} if layers else {}
                outs.append(NodeOutput(text=text, n_tokens=n, mean_logprob=mlp,
                                       first_token_entropy=float(ent0[b].item()), hidden=hid,
                                       n_prompt_tokens=int(enc['attention_mask'][b].sum().item()),
                                       gen_token_logprobs=gt_logp, gen_token_hidden=gt_hid))
        restored: List[Optional[NodeOutput]] = [None] * len(prompts)
        for pos, i in enumerate(order):
            restored[i] = sorted_outs[pos]
        return restored  # type: ignore[return-value]

    @torch.inference_mode()
    def sample(self, prompts: Sequence[str], n_samples: int, temperature: float = 0.7, max_new_tokens: int = 12,
               batch_size: int = 8, seed: int = 0) -> List[List[str]]:
        """Temperature samples (text only: the black-box interface). Seeded per call; batching is recorded by callers."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        res: List[Optional[List[str]]] = [None] * len(prompts)
        for bi, s in enumerate(range(0, len(prompts), batch_size)):
            idx = order[s:s + batch_size]
            enc = self.tok([self.format(prompts[i]) for i in idx], return_tensors="pt", padding=True).to(self.device)
            torch.manual_seed(seed + bi)
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=1.0,
                                      top_k=0, num_return_sequences=n_samples, pad_token_id=self.tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for b, i in enumerate(idx):
                res[i] = [self.tok.decode(new[b * n_samples + j], skip_special_tokens=True).strip() for j in range(n_samples)]
        return res  # type: ignore[return-value]

    @torch.inference_mode()
    def hidden_only(self, prompts: Sequence[str], layers: Sequence[int], batch_size: int = 16):
        """Prefill-only pass (no generation): last-prompt-token hidden states per layer."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        res: List[Optional[Dict[int, np.ndarray]]] = [None] * len(prompts)
        for s in range(0, len(prompts), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([self.format(prompts[i]) for i in idx], return_tensors="pt", padding=True).to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            for b, i in enumerate(idx):
                res[i] = {l: hs[l][b, -1, :].float().cpu().numpy().astype(np.float16) for l in layers}
        return res

    def timed_run(self, prompts, layers, **kw):
        t0 = time.perf_counter()
        out = self.run(prompts, layers, **kw)
        return out, time.perf_counter() - t0
