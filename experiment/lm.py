"""HF 언어 모델 래퍼: 모든 층의 잔차 흐름(residual stream)과 최종 출력 확률을 꺼낸다.

hidden_states[l]은 l번째 블록을 지난 잔차 (0 = 임베딩, L = 마지막 블록).
HF 구현은 마지막 원소에 최종 정규화를 적용해 돌려주므로(Llama 등), 최종 정규화 모듈에
forward pre-hook을 걸어 정규화 전 값으로 바꿔 넣는다. 그러면 모든 층이 같은 공간에 있고,
렌즈 읽기는 항상 unembed(final_norm(T_l(h)))로 통일된다. T_l = 항등이면 바닐라 로짓 렌즈.
"""
from __future__ import annotations

import torch

_NORM_PATHS = ["model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.final_layernorm",
               "transformer.norm"]


def _get(obj, path):
    for p in path.split("."):
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


class LM:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.norm = next((m for p in _NORM_PATHS if (m := _get(model, p)) is not None), None)
        if self.norm is None:
            raise ValueError("최종 정규화 모듈을 찾지 못함. lm._NORM_PATHS에 경로를 추가하세요.")
        self.unembed = model.get_output_embeddings() or getattr(model, "lm_head", None)
        if self.unembed is None:   # EXAONE 등 일부 커스텀 코드는 get_output_embeddings가 None
            raise ValueError("언임베딩(lm_head)을 찾지 못함")
        self.n_layers = model.config.num_hidden_layers
        self._pre_norm = None
        self.norm.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self._pre_norm = args[0]

    @classmethod
    def from_pretrained(cls, name: str, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, trust_remote_code=True)
        except TypeError:  # transformers 4.x
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True)
        model = model.to(device)
        lm = cls(model, tok, device)
        lm.sanity_check()
        return lm

    # ---------- 토큰 ----------
    def encode(self, text: str, special=True) -> list[int]:
        return self.tok(text, add_special_tokens=special)["input_ids"]

    def decode(self, ids) -> str:
        return self.tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.unembed.weight.shape[0]

    # ---------- 순전파 ----------
    @torch.no_grad()
    def run(self, ids: list[int]):
        """→ hs [L+1, T, d] (float32, 정규화 전 잔차), logits [T, V] (float32)"""
        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, output_hidden_states=True, use_cache=False)
        hs = list(out.hidden_states)
        if len(hs) != self.n_layers + 1:
            raise ValueError(f"hidden_states 길이 {len(hs)} != 층 수+1 ({self.n_layers + 1})")
        hs[-1] = self._pre_norm
        return torch.stack(hs)[:, 0].float(), out.logits[0].float()

    @torch.no_grad()
    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """잔차 h [..., d] → 로짓 [..., V] (최종 정규화 + 언임베딩)"""
        dt = self.unembed.weight.dtype
        return self.unembed(self.norm(h.to(dt))).float()

    @torch.no_grad()
    def score(self, prompt: str, cont: str):
        """프롬프트 뒤에 cont가 올 로그확률(토큰 합)과, 프롬프트 마지막 위치의 전 층 잔차.
        → (logp_sum, logp_first, n_tokens, hs_last [L+1, d])"""
        p = self.encode(prompt)
        c = self.encode(cont, special=False)
        hs, logits = self.run(p + c)
        lp = torch.log_softmax(logits, -1)
        tok_lp = [lp[len(p) - 1 + i, t].item() for i, t in enumerate(c)]
        return sum(tok_lp), tok_lp[0], len(c), hs[:, len(p) - 1]

    def sanity_check(self, text="한국어：봄\n中文："):
        """정규화 전 마지막 잔차를 다시 읽으면 모델 로짓과 같아야 한다."""
        hs, logits = self.run(self.encode(text))
        re = self.readout(hs[-1, -1])
        err = (re - logits[-1]).abs().max().item()
        scale = logits[-1].abs().max().item()
        if err > 1e-2 * max(scale, 1.0):
            raise RuntimeError(f"readout 검사 실패: 최대 오차 {err:.4f} (로짓 규모 {scale:.2f}). "
                               "최종 정규화 모듈 위치나 hidden_states 형식을 확인하세요.")
