#!/usr/bin/env python3
"""A 문항마다 뜻별 영어 probe 초안을 만든다 → A_sense_probes_draft.csv
정답 뜻(sense_target)과 방해 뜻(sense_distractor)에만 붙는 영어 단어 3개씩.
다의 영어 자신, 그리고 반대 뜻과도 이어지는 단어는 피했다. 사람이 검토한 뒤 확정한다."""
import pandas as pd

# 뜻 이름 → 영어 probe (A 문항에서 쓰는 sense 라벨 기준, 같은 en 안에서만 의미 있음)
P = {
 ("spring","season"):"season summer winter", ("spring","coil"):"coil bounce mattress",
 ("fall","season (autumn)"):"autumn season leaves", ("fall","waterfall"):"waterfall cascade river",
 ("bat","animal"):"vampire wings cave", ("bat","sports club"):"baseball swing hitter",
 ("crane","bird"):"bird heron feathers", ("crane","machine"):"construction lift machine",
 ("palm","hand"):"hand fingers wrist", ("palm","tree"):"tree coconut tropical",
 ("bow","weapon"):"arrow archer weapon", ("bow","ribbon knot"):"ribbon knot tied",
 ("club","organization"):"members society group", ("club","weapon"):"stick weapon wooden",
 ("iron","metal"):"metal steel copper", ("iron","appliance"):"ironing clothes wrinkles",
 ("key","lock"):"lock door unlock", ("key","piano"):"piano keyboard musical",
 ("letter","mail"):"mail envelope postal", ("letter","alphabet"):"alphabet character uppercase",
 ("match","stick"):"fire flame lighter", ("match","contest"):"game tournament opponent",
 ("plant","vegetation"):"flower leaves garden", ("plant","factory"):"factory industrial manufacturing",
 ("pupil","eye"):"eye iris retina", ("pupil","student"):"student classroom teacher",
 ("ring","jewelry"):"jewelry wedding diamond", ("ring","bell sound"):"bell phone ringing",
 ("rock","stone"):"stone boulder cliff", ("rock","music"):"music guitar band",
 ("star","celestial"):"sky galaxy planet", ("star","celebrity"):"celebrity famous actor",
 ("cricket","insect"):"insect chirping grasshopper", ("cricket","sport"):"wicket batsman sport",
 ("coach","trainer"):"trainer team athletes", ("coach","carriage"):"carriage horse wagon",
 ("chest","body"):"breast lungs torso", ("chest","box"):"box treasure trunk",
 ("cell","biology"):"biology membrane organism", ("cell","prison"):"prison jail prisoner",
 ("mint","herb"):"herb peppermint leaves", ("mint","coin factory"):"coin currency minted",
 ("pit","hole"):"hole dig ditch", ("pit","fruit stone"):"seed peach cherry",
 ("suit","clothing"):"jacket tuxedo formal", ("suit","lawsuit"):"lawsuit court sue",
 ("yard","garden"):"garden lawn backyard", ("yard","unit"):"feet inches meters",
 ("bill","beak"):"beak bird duck", ("bill","invoice"):"invoice payment receipt",
 ("case","legal"):"court trial lawsuit", ("case","container"):"box container suitcase",
 ("deck","ship"):"ship boat sailor", ("deck","cards"):"cards poker shuffle",
 ("digit","finger"):"finger toe thumb", ("digit","number"):"number numeric decimal",
 ("engagement","marriage"):"marriage wedding fiance", ("engagement","battle"):"battle combat troops",
 ("game","hunting prey"):"prey hunting deer", ("game","play"):"play video player",
 ("glasses","eyewear"):"spectacles lenses eyewear", ("glasses","cups"):"cups wine drinking",
 ("log","wood"):"wood timber firewood", ("log","record"):"record logging entries",
 ("note","memo"):"memo message written", ("note","music"):"music melody pitch",
 ("orange","fruit"):"fruit citrus juice", ("orange","color"):"color yellow purple",
 ("pool","swimming"):"swimming swim water", ("pool","billiards"):"billiards cue snooker",
 ("racket","sports"):"tennis badminton ball", ("racket","noise"):"noise loud din",
 ("story","tale"):"tale narrative novel", ("story","floor"):"floor building upstairs",
 ("tip","gratuity"):"gratuity waiter money", ("tip","end"):"end point edge",
 ("top","toy"):"toy spinning spin", ("top","summit"):"summit peak highest",
 ("bar","pub"):"pub tavern drinks", ("bar","rod"):"rod metal pole",
 ("diamond","gem"):"gem jewel jewelry", ("diamond","shape"):"shape rhombus triangle",
 ("interest","finance"):"loan bank rate", ("interest","curiosity"):"curiosity hobby interested",
 ("litter","trash"):"trash garbage rubbish", ("litter","offspring"):"puppies kittens offspring",
 ("plot","story"):"storyline narrative novel", ("plot","land"):"land acre parcel",
 ("sage","herb"):"herb rosemary thyme", ("sage","wise man"):"wise wisdom philosopher",
 ("staff","personnel"):"employees workers personnel", ("staff","stick"):"stick cane rod",
 ("term","semester"):"semester school academic", ("term","word"):"word terminology definition",
 ("vessel","ship"):"ship boat sailing", ("vessel","blood"):"blood artery vein",
 ("ear","body"):"hearing listen head", ("ear","grain"):"grain wheat corn",
 ("cape","clothing"):"cloak hood superhero", ("cape","land"):"peninsula coast headland",
 ("gum","mouth"):"teeth tooth mouth", ("gum","candy"):"chewing bubble candy",
 ("nut","food"):"peanut almond walnut", ("nut","hardware"):"bolt screw wrench",
 ("tablet","medicine"):"pill medicine capsule", ("tablet","computer"):"computer ipad screen",
 ("turkey","bird"):"bird thanksgiving chicken", ("turkey","country"):"country istanbul greece",
 ("plane","aircraft"):"aircraft airplane flight", ("plane","tool"):"tool carpenter woodworking",
 ("poker","game"):"cards gambling casino", ("poker","tool"):"fireplace fire tool",
 ("iris","eye"):"eye pupil retina", ("iris","flower"):"flower lily petals",
 ("lash","eye"):"eyelash eyelid eye", ("lash","whip"):"whip whipping flogging",
 ("ward","hospital"):"hospital patients nurse", ("ward","person"):"guardian custody orphan",
 ("mold","fungus"):"fungus mildew bacteria", ("mold","form"):"form casting shape",
 ("rocket","vehicle"):"spacecraft launch missile", ("rocket","plant"):"arugula salad lettuce",
 ("boot","footwear"):"shoe shoes footwear", ("boot","car"):"trunk car luggage",
 ("tie","clothing"):"necktie shirt collar", ("tie","draw"):"draw score equal",
 ("chip","computer"):"processor semiconductor silicon", ("chip","food"):"potato fries snack",
 ("tank","military"):"army military armored", ("tank","container"):"water container fuel",
 ("wave","water"):"ocean sea surf", ("wave","gesture"):"hand goodbye waving",
 ("box","container"):"container package carton", ("box","sport"):"boxing boxer punch",
 ("kid","child"):"child children boy", ("kid","goat"):"goat lamb sheep",
 ("toast","bread"):"bread butter breakfast", ("toast","drink"):"cheers drink champagne",
 ("mass","church"):"church catholic priest", ("mass","physics"):"physics weight gravity",
 ("monitor","screen"):"screen display computer", ("monitor","lizard"):"lizard reptile dragon",
 ("lid","pot"):"cover pot jar", ("lid","eye"):"eyelid eye blink",
 ("crown","king"):"king queen royal", ("crown","tooth"):"tooth dental dentist",
 ("spine","body"):"back vertebrae bone", ("spine","book"):"book cover binding",
 ("corn","grain"):"maize grain wheat", ("corn","foot"):"foot callus toe",
 ("port","harbor"):"harbor ships dock", ("port","wine"):"wine sherry liquor",
 ("sound","noise"):"noise audio voice", ("sound","strait"):"strait channel inlet",
 ("square","shape"):"shape rectangle triangle", ("square","plaza"):"plaza downtown public",
 ("record","music"):"album vinyl music", ("record","data"):"data document history",
 ("magazine","publication"):"publication journal issue", ("magazine","weapon"):"ammunition rifle bullets",
 ("straw","drinking"):"drinking sip cup", ("straw","hay"):"hay farm barn",
 ("washer","machine"):"washing laundry dryer", ("washer","hardware"):"bolt screw nut",
 ("drone","aircraft"):"aircraft unmanned flying", ("drone","bee"):"bee bees hive",
 ("bowl","dish"):"dish soup plate", ("bowl","sport"):"bowling pins alley",
 ("cast","medical"):"broken fracture plaster", ("cast","actors"):"actors film movie",
 ("hide","skin"):"leather skin fur", ("hide","verb only"):"hiding conceal secret",
 ("duck","bird"):"bird goose swan", ("duck","cotton fabric"):"canvas fabric cotton",
 ("bridge","structure"):"river crossing structure", ("bridge","card game"):"cards card game",
 ("club","golf"):"golf golfer swing",
 ("scale","weighing"):"weigh weight balance", ("scale","fish"):"fish skin scales",
 ("pupil","student"):"student classroom teacher",
 ("plant","factory"):"factory industrial manufacturing",
 ("match","contest"):"game tournament opponent",
 ("letter","alphabet"):"alphabet character uppercase",
 ("cell","prison"):"prison jail prisoner",
 ("key","piano"):"piano keyboard musical",
 ("story","floor"):"floor building upstairs",
 ("interest","curiosity"):"curiosity hobby interested",
}
A = pd.read_csv("A_final.csv", keep_default_na=False)
rows, missing = [], []
for r in A.itertuples():
    t, d = P.get((r.en, r.sense_target)), P.get((r.en, r.sense_distractor))
    if t is None or d is None: missing.append((r.id, r.en, r.sense_target, r.sense_distractor))
    rows.append(dict(id=r.id, direction=r.direction, ko_input=r.ko_input, en=r.en,
                     sense_target=r.sense_target, en_sense_target=t or "",
                     sense_distractor=r.sense_distractor, en_sense_distractor=d or "",
                     zh_answer=r.zh_answer, zh_distractor=r.zh_distractor))
# (초안은 아래에서 A_sense_probes.csv에 합친다)
print("missing:", missing)

AMBIG = set("""point pitch game spin issue record trunk cover rate lift draw play swing pole rod stick fire lighter
equal public history school head edge end bank group society members score secret leaves cave tied structure crossing
document data written message highest machine wooden metal tool video player broken channel hand drinks sea""".split())
# 걸러진 뒤 2단어가 안 되는 묶음 보충 (1토큰 확인)
EXTRA = {
 ("A010","target"):"association organization", ("A010","distractor"):"beating",
 ("A029","target"):"bug insects", ("A029","distractor"):"innings",
 ("A036","target"):"herbs basil", ("A062","target"):"reminder notebook",
 ("A084","target"):"debt mortgage banking", ("A091","target"):"herbs basil",
 ("A099","target"):"deaf auditory", ("A108","target"):"seeds snack",
 ("A119","target"):"eyes blink", ("A192","target"):"tunnel highway road",
 ("A194","target"):"tee putt", ("A004","distractor"):"cargo lifting",
 ("A056","distractor"):"fun playing", ("A066","distractor"):"pocket",
 ("A073","distractor"):"sharp pointed", ("A078","distractor"):"steel beam",
 ("A092","distractor"):"walking", ("A117","distractor"):"ashes coal",
 ("A131","distractor"):"drawn", ("A140","distractor"):"fight fighter gloves",
 ("A168","distractor"):"archive database files", ("A194","distractor"):"association membership",
 ("A213","distractor"):"debt mortgage banking",
}
# ---- 기존 A_sense_probes.csv(뜻마다 1단어, 토큰 검사 완료)에 묶음 열을 덧붙인다
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
ntok = lambda w: len(tok.encode(" " + w, add_special_tokens=False))
S = pd.read_csv("A_sense_probes.csv", keep_default_na=False)
d = pd.DataFrame(rows).set_index("id")
for side, col, base in [("target", "en_sense_target", "en_target_sense"), ("distractor", "en_sense_distractor", "en_distractor_sense")]:
    sets, multi = [], []
    for r in S.itertuples():
        words = [getattr(r, base)] + [w for w in d.loc[r.id, col].split() if w != getattr(r, base) and w != r.en]
        seen, uniq = set(), []
        for w in words:
            if w.lower() not in seen: seen.add(w.lower()); uniq.append(w)
        # 검토 규칙: 다의 영어의 활용형(ringing, boxer…) 제외, 여러 뜻을 가진 단어 제외, Llama에서 1토큰이 아닌 단어 제외
        # (기존 1단어 probe는 이미 사람이 검사했으므로 그대로 둔다)
        b = getattr(r, base)
        words = [w for w in uniq if w == b or (r.en.lower() not in w.lower() and w.lower() not in AMBIG and ntok(w) == 1)]
        words += [w for w in EXTRA.get((r.id, side), "").split() if w not in words]
        sets.append(" ".join(words))
        multi.append(" ".join(w for w in words if ntok(w) > 1))
    S[f"en_{side}_set"] = sets
    S[f"multi_tok_{side}"] = multi
# 같은 문항에서 두 뜻 묶음에 동시에 들어간 단어 표시
S["overlap"] = [" ".join(set(a.lower().split()) & set(b.lower().split())) for a, b in zip(S.en_target_set, S.en_distractor_set)]
S.to_csv("A_sense_probes.csv", index=False)
print("overlap:", S[S.overlap != ""][["id", "overlap"]].values.tolist())
print(S[["id","en","en_target_set","en_distractor_set","multi_tok_target","multi_tok_distractor"]].to_string(index=False))
