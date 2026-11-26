#!/usr/bin/env python3
"""
묶음 A(영어 다의어) 후보 표에 자동 검사·부착을 수행한다.
 
  1) 중국어 정답/방해어 글자 공유 검사     (오프라인)
  2) wordfreq Zipf 빈도 부착 (en/ko/zh)      (오프라인)
  3) 한국어 어원 부착: 우리말샘 또는 표준국어대사전 오픈 API의 '원어' 필드
     - 원어에 한자가 있으면 한자어, 원어가 없으면 고유어, 그 밖의 원어는 외래어,
       한자와 다른 원어가 섞이면 혼종어
     - 동형어(같은 표기의 다른 표제어) 개수도 함께 부착 → 한국어 다의어 필터 보조
  4) (선택) 두 모델 토크나이저 기준 토큰 수
 
사용법
  pip install "wordfreq[cjk]" requests
  export URIMALSAEM_API_KEY=...        # 또는 STDICT_API_KEY
  python annotate_A.py A_candidates_draft.csv -o A_candidates_annotated.csv \
      --etym-api urimalsaem
  # 어원 없이 빠르게: --etym-api none
  # 토큰 수까지:     --tokenizers meta-llama/Llama-3.2-3B,LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
 
API 응답은 --cache(기본 etym_cache.json)에 저장되므로 재실행 시 다시 호출하지 않는다.
"""
from __future__ import annotations
 
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
 
HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
HANGUL_RE = re.compile(r"[가-힣]")
 
 
# ----------------------------------------------------------------------------
# 1) 글자 공유
# ----------------------------------------------------------------------------
def han_chars(s: str) -> set[str]:
    return set(HAN_RE.findall(s or ""))
 
 
def shared_chars(a: str, b: str) -> str:
    return "".join(sorted(han_chars(a) & han_chars(b)))
 
 
# ----------------------------------------------------------------------------
# 2) 빈도
# ----------------------------------------------------------------------------
_wf = None
 
 
def zipf(word: str, lang: str) -> float | None:
    global _wf
    if _wf is None:
        import wordfreq  # noqa: WPS433
        _wf = wordfreq
    word = (word or "").strip()
    if not word:
        return None
    # 여러 단어(공백 포함)는 wordfreq가 구 빈도로 근사한다(하한값에 가까움).
    return round(_wf.zipf_frequency(word, lang), 2)
 
 
def split_multi(cell: str) -> list[str]:
    return [w.strip() for w in (cell or "").split(";") if w.strip()]
 
 
# ----------------------------------------------------------------------------
# 3) 어원 (우리말샘 / 표준국어대사전)
# ----------------------------------------------------------------------------
API = {
    "urimalsaem": {
        "env": "URIMALSAEM_API_KEY",
        "search": "https://opendict.korean.go.kr/api/search",
        "view": "https://opendict.korean.go.kr/api/view",
        "search_params": {"req_type": "json", "part": "word", "num": 100, "sort": "dict",
                          "advanced": "y", "method": "exact", "target": 1},
        "view_params": {"req_type": "json", "method": "target_code"},
    },
    "stdict": {
        "env": "STDICT_API_KEY",
        "search": "https://kdict.korean.go.kr/standard/api/search.do",
        "view": "https://kdict.korean.go.kr/standard/api/view.do",
        "search_params": {"req_type": "json", "num": 100, "advanced": "y", "method": "exact"},
        "view_params": {"req_type": "json", "method": "target_code"},
    },
}
 
 
def _walk(obj):
    """JSON 전체를 재귀 순회하며 (key, value) 쌍을 낸다. API마다 필드 위치가 달라 구조에 의존하지 않기 위함."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
 
 
def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]
 
 
def _norm_headword(w: str) -> str:
    # 표제어 표기의 ^, -, 공백, 동형어 번호 제거
    return re.sub(r"[\^\-\s0-9]", "", w or "")
 
 
class EtymClient:
    def __init__(self, backend: str, key: str, cache_path: Path, sleep: float = 0.2):
        import requests  # noqa: WPS433
        self.req = requests
        self.cfg = API[backend]
        self.backend = backend
        self.key = key
        self.sleep = sleep
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}
 
    def save(self):
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), "utf-8")
 
    def _get(self, url, params):
        ck = self.backend + "|" + url + "|" + json.dumps(params, sort_keys=True, ensure_ascii=False)
        if ck in self.cache:
            return self.cache[ck]
        p = dict(params, key=self.key)
        for attempt in range(4):
            try:
                r = self.req.get(url, params=p, timeout=20)
                r.raise_for_status()
                txt = r.text.strip()
                data = json.loads(txt) if txt else {}
                break
            except Exception as e:  # 네트워크/파싱 오류 → 재시도
                if attempt == 3:
                    print(f"  [warn] API 실패 {params.get('q')}: {e}", file=sys.stderr)
                    return None
                time.sleep(1.5 * (attempt + 1))
        time.sleep(self.sleep)
        self.cache[ck] = data
        return data
 
    def entries(self, word: str) -> list[dict]:
        """표기가 정확히 같은 명사 표제어 목록: [{target_code, sup_no, pos, definition}]"""
        data = self._get(self.cfg["search"], dict(self.cfg["search_params"], q=word))
        if not data:
            return []
        if "error" in data:
            raise RuntimeError(f"API 오류 응답: {data['error']}")
        items = _as_list((data.get("channel") or {}).get("item"))
        out = []
        for it in items:
            if _norm_headword(it.get("word", "")) != _norm_headword(word):
                continue
            senses = _as_list(it.get("sense"))
            pos = ",".join(sorted({s.get("pos", "") for s in senses if isinstance(s, dict)})) or it.get("pos", "")
            if pos and "명사" not in pos:
                continue
            defin = senses[0].get("definition", "") if senses and isinstance(senses[0], dict) else ""
            tc = it.get("target_code") or (senses[0].get("target_code") if senses and isinstance(senses[0], dict) else None)
            out.append({"target_code": tc, "sup_no": it.get("sup_no", ""), "pos": pos,
                        "definition": re.sub(r"<[^>]+>", "", defin)[:60]})
        # 우리말샘 검색은 뜻풀이 단위로 같은 표제어가 여러 번 나올 수 있어 target_code로 중복 제거
        seen, uniq = set(), []
        for e in out:
            k = e["target_code"] or (e["sup_no"], e["definition"])
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return uniq
 
    def origin(self, target_code, word="", sup_no="") -> tuple[str, str]:
        """(원어 문자열, 언어 유형 문자열). 원어가 없으면 ("", "")."""
        if not target_code:
            return "", ""
        data = self._get(self.cfg["view"], dict(self.cfg["view_params"], q=str(target_code)))
        if not data and self.backend == "stdict" and word:
            # view.do가 빈 응답을 줄 때: 검색 API의 type2(어종) 필터로 이 target_code가 어느 어종에 걸리는지 확인
            kinds = {"native": "고유어", "chinese": "한자어", "loanword": "외래어", "hybrid": "혼종어"}
            for t2, label in kinds.items():
                d = self._get(self.cfg["search"], dict(self.cfg["search_params"], q=word, type2=t2))
                if not d or "error" in d:
                    continue
                for it in _as_list((d.get("channel") or {}).get("item")):
                    if str(it.get("target_code")) == str(target_code):
                        origs = [x.get("origin", "") for x in _as_list(it.get("sense")) if isinstance(x, dict)]
                        return "".join(o for o in origs if o), label
            return "", "UNKNOWN"
        if not data:
            return "", ""
        origs, types = [], []
        for k, v in _walk(data):
            kl = k.lower()
            if not isinstance(v, str) or not v.strip():
                continue
            if kl == "original_language":
                origs.append(v.strip())
            elif kl == "origin" and "＜" not in v and "<" not in v:
                # 표준국어대사전 origin은 옛말 형태(＜나모＜용가＞)라 원어가 아님
                origs.append(v.strip())
            elif kl in ("language_type", "word_type"):
                # 표준국어대사전 view는 word_type(고유어/한자어/외래어/혼종어)
                types.append(v.strip())
        return "".join(dict.fromkeys(origs)), "+".join(dict.fromkeys(types))
 
 
def classify(orig: str, ltype: str) -> str:
    if ltype in ("고유어", "한자어", "외래어", "혼종어", "UNKNOWN"):
        return ltype
    if not orig and not ltype:
        return "고유어"
    has_han = bool(HAN_RE.search(orig)) or "한자" in ltype
    has_hangul = bool(HANGUL_RE.search(orig))
    other = bool(re.search(r"[A-Za-z]", orig)) or any(t and t != "한자" and t != "고유어" for t in ltype.split("+"))
    if has_han and (other or has_hangul or "고유어" in ltype):
        return "혼종어"
    if has_han:
        return "한자어"
    if other:
        return "외래어"
    return "고유어"
 
 
def etym_for(client: EtymClient | None, word: str) -> dict:
    word = (word or "").strip()
    if client is None or not word:
        return {}
    if " " in word:
        return {"etym": "MULTIWORD", "origin": "", "homograph_n": "", "entries": ""}
    ents = client.entries(word)
    if not ents:
        return {"etym": "NOT_FOUND", "origin": "", "homograph_n": 0, "entries": ""}
    labs = []
    for e in ents:
        o, t = client.origin(e["target_code"], word, e["sup_no"])
        e["origin"], e["etym"] = o, classify(o, t)
        labs.append(f"{e['sup_no'] or '-'}:{e['etym']}{'(' + o + ')' if o else ''}:{e['definition']}")
    kinds = sorted({e["etym"] for e in ents})
    return {
        # 동형어가 여러 개면 첫 표제어를 대표로 쓰되, 종류가 섞이면 ETYM_AMBIG로 표시 → 사람이 확인
        "etym": ents[0]["etym"] if len(kinds) == 1 else "AMBIG:" + "/".join(kinds),
        "origin": ents[0]["origin"],
        "homograph_n": len(ents),
        "entries": " | ".join(labs),
    }
 
 
# ----------------------------------------------------------------------------
# 4) 토큰 수 (선택)
# ----------------------------------------------------------------------------
def load_tokenizers(names: list[str]):
    from transformers import AutoTokenizer  # noqa: WPS433
    toks = {}
    for n in names:
        short = n.split("/")[-1]
        toks[short] = AutoTokenizer.from_pretrained(n, trust_remote_code=True)
    return toks
 
 
def ntok(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])
 
 
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--etym-api", choices=["urimalsaem", "stdict", "none"], default="urimalsaem")
    ap.add_argument("--api-key", default=None, help="미지정 시 URIMALSAEM_API_KEY / STDICT_API_KEY 환경변수")
    ap.add_argument("--cache", default="etym_cache.json")
    ap.add_argument("--rare-zipf", type=float, default=2.5, help="중국어 방해어 희귀 기준 (Zipf 미만)")
    ap.add_argument("--freq-gap", type=float, default=1.0, help="입력어-대조어 빈도 차 허용치 (Zipf)")
    ap.add_argument("--tokenizers", default="", help="쉼표 구분 HF 모델 id (선택)")
    args = ap.parse_args()
 
    inp = Path(args.input)
    out = Path(args.output or inp.with_name(inp.stem.replace("_draft", "") + "_annotated.csv"))
    rows = list(csv.DictReader(open(inp, encoding="utf-8-sig")))
 
    client = None
    if args.etym_api != "none":
        key = args.api_key or os.environ.get(API[args.etym_api]["env"])
        if not key:
            print(f"[info] {API[args.etym_api]['env']} 가 없어 어원 부착을 건너뜁니다 (--etym-api none 과 동일).",
                  file=sys.stderr)
        else:
            client = EtymClient(args.etym_api, key, Path(args.cache))
 
    toks = load_tokenizers([t for t in args.tokenizers.split(",") if t.strip()]) if args.tokenizers else {}
 
    for i, r in enumerate(rows, 1):
        flags = []
        # 1) 글자 공유
        sh = shared_chars(r["zh_answer"], r["zh_distractor"])
        r["zh_shared_chars"] = sh
        if sh:
            flags.append("ZH_SHARE")
 
        # 2) 빈도
        r["zipf_en"] = zipf(r["en"], "en")
        r["zipf_ko_input"] = zipf(r["ko_input"], "ko")
        r["zipf_ko_reverse"] = zipf(r["ko_reverse_input"], "ko")
        r["zipf_zh_answer"] = zipf(r["zh_answer"], "zh")
        r["zipf_zh_distractor"] = zipf(r["zh_distractor"], "zh")
        ctl = split_multi(r["ko_control"])
        cz = [zipf(c, "ko") for c in ctl]
        r["zipf_ko_control"] = ";".join(str(z) for z in cz)
        if r["zipf_zh_distractor"] is not None and r["zipf_zh_distractor"] < args.rare_zipf:
            flags.append("RARE_DIST")
        if r["zipf_ko_input"] and cz and any(z and abs(z - r["zipf_ko_input"]) > args.freq_gap for z in cz):
            flags.append("CTRL_FREQ_GAP")
        if r["zipf_ko_input"] == 0:
            flags.append("KO_FREQ_ZERO")  # wordfreq 목록 밖 → 수동 확인
 
        # 3) 어원
        if client is not None:
            e = etym_for(client, r["ko_input"])
            r["etym_ko_input"] = e.get("etym", "")
            r["origin_ko_input"] = e.get("origin", "")
            r["homograph_n_ko_input"] = e.get("homograph_n", "")
            r["entries_ko_input"] = e.get("entries", "")
            er = etym_for(client, r["ko_reverse_input"])
            r["etym_ko_reverse"] = er.get("etym", "")
            r["homograph_n_ko_reverse"] = er.get("homograph_n", "")
            ce = [etym_for(client, c).get("etym", "") for c in ctl]
            r["etym_ko_control"] = ";".join(ce)
            if isinstance(e.get("homograph_n"), int) and e["homograph_n"] > 1:
                flags.append("KO_HOMOGRAPH")
            if isinstance(er.get("homograph_n"), int) and er["homograph_n"] > 1:
                flags.append("KO_HOMOGRAPH_REV")
            if str(r["etym_ko_input"]).startswith("AMBIG"):
                flags.append("ETYM_AMBIG")
            base = r["etym_ko_input"]
            if base in ("고유어", "한자어", "외래어", "혼종어") and any(
                    c in ("고유어", "한자어", "외래어", "혼종어") and c != base for c in ce):
                flags.append("CTRL_ETYM_DIFF")
            if i % 20 == 0:
                client.save()
                print(f"  {i}/{len(rows)}", file=sys.stderr)
 
        # 4) 토큰 수
        for short, tok in toks.items():
            n_sp = ntok(tok, " " + r["en"])
            r[f"ntok_en_{short}"] = n_sp
            r[f"ntok_en_nospace_{short}"] = ntok(tok, r["en"])
            r[f"ntok_zh_answer_{short}"] = ntok(tok, r["zh_answer"])
            r[f"ntok_zh_distractor_{short}"] = ntok(tok, r["zh_distractor"])
            if n_sp > 1:
                flags.append(f"EN_MULTITOK_{short}")
 
        r["auto_flags"] = ";".join(flags)
 
    if client is not None:
        client.save()
 
    fields = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
 
    # 요약
    from collections import Counter
    c = Counter(fl for r in rows for fl in r["auto_flags"].split(";") if fl)
    clean = sum(1 for r in rows if not r["auto_flags"] and not r.get("draft_flags"))
    print(f"[done] {len(rows)}행 → {out}")
    print(f"       자동 플래그: {dict(c)}")
    print(f"       초안·자동 플래그 모두 없는 행: {clean}")
 
 
if __name__ == "__main__":
    main()
 