"""가짜 모델로 전체 파이프라인(렌즈 적합 → 문항 실행 → 분석)이 도는지 확인한다.
HF 모델 인터페이스를 흉내 낸 작은 torch 모델과 글자 단위 토크나이저를 쓰므로 다운로드가 필요 없다.
수치 결과에는 의미가 없다.

  python -m experiment.tests.test_pipeline
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from experiment import prompts
from experiment.analyze import analyze
from experiment.config import ROOT
from experiment.data import load_A, load_B, load_C
from experiment.fit_lenses import fit
from experiment.lenses import load_lenses
from experiment.lm import LM
from experiment.run_items import run_all


class CharTokenizer:
    def __init__(self, texts):
        chars = sorted({ch for t in texts for ch in t} | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "))
        self.itos = ["<bos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __call__(self, text, add_special_tokens=True):
        ids = [self.stoi[c] for c in text if c in self.stoi]
        return {"input_ids": ([0] if add_special_tokens else []) + ids}

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class FakeLM(nn.Module):
    """HF CausalLM처럼 hidden_states 마지막 원소에 최종 정규화를 적용해 돌려준다."""

    def __init__(self, vocab, d=32, n_layers=8):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=n_layers)
        self.model = nn.Module()
        self.model.embed = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(n_layers))
        self.model.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.bias = nn.Parameter(torch.randn(d) * 3)   # 모든 층에 같은 방향 성분 (평균 성분 흉내)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, output_hidden_states=True, use_cache=False):
        h = self.model.embed(input_ids)
        hs = [h]
        for i, layer in enumerate(self.model.layers):
            mix = torch.cumsum(h, 1) / torch.arange(1, h.shape[1] + 1)[None, :, None]   # 인과적 평균
            h = h + torch.tanh(layer(mix)) + 0.1 * self.bias
            hs.append(h)
        h = self.model.norm(h)
        hs[-1] = h
        return SimpleNamespace(logits=self.lm_head(h), hidden_states=tuple(hs))


def main():
    torch.manual_seed(0)
    A = load_A(ROOT / "A_final.csv")[:12]
    bdraft = pd.read_csv(ROOT / "B_candidates_draft.csv", keep_default_na=False)
    tmp = Path(tempfile.mkdtemp())
    bdraft["word_etym"] = ["고유어쌍" if not (h1 or h2) else "혼합" for h1, h2 in zip(bdraft.hanja_target, bdraft.hanja_distractor)]
    bdraft.head(12).to_csv(tmp / "B.csv", index=False)
    B = load_B(tmp / "B.csv")
    pd.DataFrame([dict(id="C001", ko="바다", zh="海", en="sea", en_unrelated="chair", zh_unrelated="椅子", etym="고유어"),
                  dict(id="C002", ko="산", zh="山", en="mountain", en_unrelated="spoon", zh_unrelated="勺子", etym="고유어")]
                 ).to_csv(tmp / "C.csv", index=False)
    C = load_C(tmp / "C.csv")

    corpus = ["봄이 오면 꽃이 핀다.", "春天来了，花开了。", "한국어 문장입니다.", "这是中文句子。"] * 10
    texts = corpus + [prompts.build(t, v, it.ko, it.en_hint) + it.zh_answer + it.zh_distractor
                      for it in A + B + C for t in ("translate", "translate_en", "repeat") for v in ("v1", "v2", "v3", "v4")]
    texts += [c for it in A + B + C for c in it.controls] + [w for it in A + B + C for w in it.probes.values()]
    tok = CharTokenizer(texts)
    lm = LM(FakeLM(len(tok.itos)), tok, "cpu")
    lm.sanity_check()

    args = SimpleNamespace(holdout=0.2, max_tokens=64, max_pos=8, eval_max=200,
                           lambdas="1e-3,1e-1", stats_device="cpu")
    fit(lm, corpus, tmp, args)
    lenses = load_lenses(["vanilla", "mean", "affine"], tmp / "lenses.pt")
    # 아핀 렌즈의 마지막 층은 항등이어야 한다
    p = torch.load(tmp / "lenses.pt")
    assert torch.allclose(p["W"][-1], torch.eye(p["W"].shape[-1]))

    run_all(lm, {"A": A, "B": B, "C": C}, {"A": ["translate", "translate_en"], "B": ["translate"], "C": ["translate", "repeat"]},
            ["v1", "v2", "v3", "v4"], lenses, tmp)
    out = pd.read_csv(tmp / "outputs.csv")
    lens = pd.read_csv(tmp / "lens.csv.gz")
    n_prompts = sum((1 + len(it.controls)) * 4 * 2 for it in A) + sum((1 + len(it.controls)) * 4 for it in B) + len(C) * 4 * 2
    assert len(out) == n_prompts, (len(out), n_prompts)
    assert len(lens) == n_prompts * 3 * (lm.n_layers + 1)
    # 바닐라 렌즈의 마지막 층 = 모델 최종 출력
    it = A[0]
    pr = prompts.build("translate", "v1", it.ko)
    hs, logits = lm.run(lm.encode(pr))
    ids = lm.encode(it.zh_distractor, special=False)
    full = torch.log_softmax(logits[-1], -1)
    row = lens[(lens.id == it.id) & (lens.task == "translate") & (lens.version == "v1") & (lens.role == "input")
               & (lens.lens == "vanilla") & (lens.layer == lm.n_layers)]
    assert row.top1.iloc[0] == lm.decode([full.argmax().item()])
    _ = ids

    m = analyze(tmp, n_boot=200)
    assert "C" in m and "verdict" in m and "negative_control" in m and "positive_control" in m
    print("\n" + (tmp / "summary.md").read_text("utf-8")[:3000])
    print("\n[OK] 파일:", sorted(p.name for p in tmp.iterdir()))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
