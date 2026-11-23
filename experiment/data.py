"""데이터셋 A/B/C를 공통 문항(Item) 형식으로 읽는다.

Item 필드
  ko           입력 한국어 단어
  controls     대조어 목록 (간섭 = log p(방해어|입력어) - 평균 log p(방해어|대조어))
  zh_answer    정답 중국어
  zh_distractor 방해어 중국어 (C는 무관 단어)
  probes       렌즈로 읽을 영어 단어 {이름: 단어}
  primary      산점도 x축에 쓰는 probe 이름
  group        bootstrap 묶음 단위 (A: 영어 단어, B: 한국어 입력어)
  etym         고유어/한자어/외래어/... (B는 고유어쌍/한자어쌍/혼합)
  en_hint      양성 대조 프롬프트에 넣을 영어 중간 단계
  en_controls  양성 대조에서 대조어 프롬프트에 넣을 대조어의 영어 (A_control_en.csv 매핑).
               없으면 translate_en에서는 대조어 프롬프트를 만들지 않는다. 대조어 프롬프트에 문항의 영어를
               넣으면(여름 + spring) 대조어 쪽 방해어 확률이 부풀어 간섭이 왜곡되기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Item:
    id: str
    dataset: str
    ko: str
    controls: list[str]
    zh_answer: str
    zh_distractor: str
    probes: dict[str, str]
    primary: str
    group: str
    etym: str
    en_hint: str = ""
    en_controls: list[str] = field(default_factory=list)   # 양성 대조용 대조어의 영어 (없으면 간섭 계산 안 함)
    meta: dict = field(default_factory=dict)


def _split(cell: str) -> list[str]:
    return [w.strip() for w in str(cell).split(";") if w.strip()]


def _etym_A(e: str) -> str:
    if e in ("고유어", "한자어", "외래어", "혼종어"):
        return e
    return "모호"  # AMBIG:*, UNKNOWN, NOT_FOUND


def load_A(path: Path, control_en: Path | None = None) -> list[Item]:
    """control_en: 대조어 → 영어 매핑 CSV(ko,en). 기본은 A 파일 옆의 A_control_en.csv.
    매핑이 없는 대조어가 하나라도 있으면 그 문항은 양성 대조 간섭을 계산하지 않는다."""
    d = pd.read_csv(path, keep_default_na=False)
    control_en = Path(control_en or Path(path).with_name("A_control_en.csv"))
    en_map = dict(pd.read_csv(control_en, keep_default_na=False)[["ko", "en"]].values) if control_en.exists() else {}
    items = []
    for r in d.itertuples():
        ctrl = _split(r.ko_control_final)
        en_ctrl = [en_map.get(c, "") for c in ctrl]
        items.append(Item(id=r.id, dataset="A", ko=r.ko_input, controls=ctrl,
                          zh_answer=r.zh_answer, zh_distractor=r.zh_distractor,
                          probes={"poly": r.en}, primary="poly", group=r.en,
                          etym=_etym_A(r.etym_ko_input), en_hint=r.en,
                          en_controls=en_ctrl if all(en_ctrl) else [],
                          meta={"direction": r.direction}))
    return items


def load_B(path: Path) -> list[Item]:
    """B의 영어 probe: 방해어 뜻의 영어(배→선박 행이면 pear)를 주 지표로 쓴다.
    A에서 spring이 방해어 뜻(용수철)을 실어 나르는 통로였던 것과 같은 자리다."""
    d = pd.read_csv(path, keep_default_na=False)
    ctrl = "ko_control_final" if "ko_control_final" in d.columns else "ko_control"
    return [Item(id=r.id, dataset="B", ko=r.ko_input, controls=_split(getattr(r, ctrl)),
                 zh_answer=r.zh_answer, zh_distractor=r.zh_distractor,
                 probes={"distractor": r.en_distractor, "target": r.en_target}, primary="distractor",
                 group=r.ko_input, etym=r.word_etym if "word_etym" in d.columns else "",
                 en_hint=r.en_target)
            for r in d.itertuples()]


def load_C(path: Path) -> list[Item]:
    """C 스키마(아직 미작성): id, ko, zh, en, en_unrelated, zh_unrelated, etym
    대조어가 없으므로 간섭은 계산하지 않고, 렌즈에서 정답 영어(sea)와 무관 영어(chair)를 비교한다."""
    d = pd.read_csv(path, keep_default_na=False)
    return [Item(id=r.id, dataset="C", ko=r.ko, controls=[],
                 zh_answer=r.zh, zh_distractor=r.zh_unrelated,
                 probes={"target": r.en, "unrelated": r.en_unrelated}, primary="target",
                 group=r.ko, etym=getattr(r, "etym", ""), en_hint=r.en)
            for r in d.itertuples()]


LOADERS = {"A": load_A, "B": load_B, "C": load_C}


def load(name: str, path: Path) -> list[Item]:
    return LOADERS[name](Path(path))
