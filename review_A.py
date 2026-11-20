import pickle, math, wordfreq, pandas as pd
idx = pickle.load(open('../cedict_idx.pkl','rb'))
fd = wordfreq.get_frequency_dict('ko')
zf = lambda w: round(math.log10(fd[w]*1e9),2) if w in fd else 0.0
d = pd.read_csv('/mnt/project-files/dataset_A/A_candidates_annotated.csv', keep_default_na=False)

# ---------- 1. CEDICT ----------
# word -> (status, fix, note). 사전에 있고 뜻이 맞으면 기본값 OK.
J = {
 '狗叫声':('COMPOSITIONAL','吠叫','표제어 없음. 狗+叫声 조합으로 뜻은 통함. CEDICT 표제어 吠叫(to bark, 동사)'),
 '象鼻':('COMPOSITIONAL','','표제어 없음. 象+鼻 조합'),
 '一副牌':('COMPOSITIONAL','扑克牌','표제어 없음. 一副(양사)+牌'),
 '一圈':('COMPOSITIONAL','','표제어 없음. 圈 = 바퀴 양사(CEDICT: laps of race)'),
 '一绺头发':('COMPOSITIONAL','','표제어 없음. 绺 = lock(머리 타래)'),
 '喧闹声':('COMPOSITIONAL','吵闹声','표제어 없음. 吵闹声은 CEDICT에 noise로 있음'),
 '裙裾':('NOT_FOUND','','CEDICT에 없음. 드레스 train 뜻 표제어도 없음(裙摆=치맛단). 원어민 확인 필요'),
 '一窝幼崽':('COMPOSITIONAL','','표제어 없음. 幼崽(동물 새끼)는 있음'),
 '拳师犬':('NOT_FOUND','','CEDICT에 복서(개) 표제어 없음. 일반 용어이긴 함'),
 '鳐鱼':('NOT_FOUND','鳐','CEDICT는 鳐(skate/ray). A102 방해어와 A111 정답이 같은 단어'),
 '狗鱼':('NOT_FOUND','','CEDICT에 pike(물고기) 없음. 동물학 표준명이긴 함'),
 '鳎鱼':('NOT_FOUND','鳎','CEDICT는 鳎(sole fish)'),
 '对勾':('NOT_FOUND','对号','CEDICT 对号 = tick / check mark'),
 '被监护人':('COMPOSITIONAL','','표제어 없음. 被+监护(to act as a guardian)+人'),
 '轻敲':('COMPOSITIONAL','敲','표제어 없음. 敲 = to tap / to knock'),
 '北美红雀':('NOT_FOUND','','CEDICT에 cardinal(새) 없음'),
 '小山羊':('COMPOSITIONAL','','표제어 없음. 小+山羊'),
 '一段时间':('COMPOSITIONAL','','표제어 없음. 一段+时间'),
 '裤子拉链':('COMPOSITIONAL','','표제어 없음. 裤子+拉链'),
 '保险库':('NOT_FOUND','金库','CEDICT 金库 = treasury, 保险柜 = safe'),
 '河鲈':('NOT_FOUND','鲈','CEDICT 鲈 = common perch / bass. A028 정답 鲈鱼(bass)와 글자 겹침'),
 '划水':('NOT_FOUND','泳姿','CEDICT 泳姿 = swimming stroke'),
 '指示犬':('NOT_FOUND','','CEDICT에 pointer(개) 없음(塞特犬=setter만 있음)'),
 '浅口鞋':('NOT_FOUND','高跟鞋','CEDICT에 pumps(구두) 없음. 高跟鞋(하이힐)이 근사치'),
 '煤玉':('NOT_FOUND','','CEDICT에 jet(보석) 없음. 방해어 자체가 매우 드묾'),
 '封蜡':('NOT_FOUND','','CEDICT에 sealing wax 없음'),
 '水壶':('PARTIAL','水罐','CEDICT 水壶 = kettle/canteen, 水罐 = jug, pitcher. 다만 한국어 입력 주전자는 水壶 쪽에 가까움'),
 '水流':('PARTIAL','洋流','CEDICT 水流 = river/stream. 입력 해류(海流)면 洋流(ocean current)'),
 '漏洞':('PARTIAL','程错','CEDICT 漏洞 = leak/loophole(보안 구멍). 소프트웨어 bug는 程错(glitch/bug) 또는 故障'),
 '大腿':('MISMATCH','膝盖','CEDICT 大腿 = thigh. 입력 무릎은 膝盖(knee)로 갈 가능성이 큼. lap 대응 자체가 약함'),
 '判决':('PARTIAL','刑罚','CEDICT 判决 = judgment/to sentence. 역방향 입력 형벌이면 刑罚(sentence/penalty)'),
 '看守':('PARTIAL','哨兵','CEDICT 看守 = to guard(동사). 역방향 입력 보초면 哨兵(sentinel)'),
 '球茎':('PARTIAL','鳞茎','CEDICT 球茎 = corm, 鳞茎 = bulb'),
 '围栏':('PARTIAL','畜栏','CEDICT 围栏 = fence. 가축 우리(pen)는 畜栏'),
 '拖鞋':('PARTIAL','穆勒鞋','CEDICT 拖鞋 = slippers, 穆勒鞋 = mule shoes'),
 '象':('NOT_IN_GLOSS','','CEDICT 뜻풀이에 체스 bishop 없음(elephant). 국제체스 bishop을 象이라 부르긴 함'),
 '极地':('PARTIAL','极','CEDICT 极地 = polar region, 极 = pole'),
 '芯片':('OK','半导体','영어 chip과는 맞음. 단 한국어 입력 반도체는 半导体(semiconductor)로 갈 가능성'),
 '教练':('OK','','영어 coach와 맞음. 한국어 감독은 导演/主教练 쪽도 가능'),
 '拳打':('OK','','동사(to punch)'),
 '鞭打':('OK','','동사(to whip)'),
 '挥手':('OK','','동사(to wave)'),
 '交战':('OK','','동사(to fight)'),
 '躲藏':('OK','','동사. sense_distractor가 verb only'),
 '干杯':('OK','','동사/감탄사'),
 '玩笑':('OK','','joke. prank보다 넓음'),
 '白蜡树':('OK','','Chinese ash(물푸레나무속)'),
 '帆布':('OK','','canvas(duck cloth)'),
}
def ced(w):
    g = idx.get(w)
    found = 'Y' if g else 'N'
    gloss = (' || '.join(g))[:160] if g else ''
    st, fix, note = J.get(w, ('OK','',''))
    if fix and fix not in idx and fix not in ('',): note += ' [제안어도 CEDICT 없음]'
    return found, gloss, st, fix, note
for col, pre in [('zh_answer','cedict_answer'),('zh_distractor','cedict_distractor')]:
    res = d[col].map(ced)
    d[pre+'_found'] = res.map(lambda x:x[0])
    d[pre+'_gloss'] = res.map(lambda x:x[1])
    d[pre+'_ok'] = res.map(lambda x:x[2])
    d[pre+'_fix'] = res.map(lambda x:x[3])
    d[pre+'_note'] = res.map(lambda x:x[4])

# ---------- 2. 동형어 일상어 판단 ----------
# Y: 일상에서 흔히 쓰는 동형어가 있음 / ?: 경계 / N: 사전에만 있는 드문 동형어
H = {
 '도장':('Y','道場(태권도 도장), 塗裝(도장 공사)'), '은행':('Y','은행나무/은행 열매'),
 '팬':('Y','프라이팬(pan), 기계 팬(fan)'), '점':('Y','占(점을 보다), -점(가게)'),
 '못':('Y','연못(pond), 굳은살(못이 박히다); 부사 못(can\'t)도 표기 같음'),
 '자':('Y','者(그 자), 감탄사 자, 字'), '베이스':('Y','base(베이스캠프, 메이크업 베이스, 야구 베이스)'),
 '호박':('Y','琥珀(amber)'), '쥐':('Y','쥐가 나다(근육 경련)'), '경주':('Y','慶州(도시)'),
 '사원':('Y','社員(회사원)'), '무기':('Y','無期(무기징역)'), '볼트':('Y','volt(전압 단위)'),
 '수도':('Y','水道(수돗물), 修道'), '줄':('Y','끈·줄 서기(line), joule'),
 '발':('Y','發(총알 한 발), 대나무 발(햇빛 가리개)'), '쪽':('Y','방향(이쪽), 조각(한 쪽)'),
 '시동':('Y','始動(시동을 걸다)'), '극':('Y','劇(연극), 極'), '펀치':('Y','사무용 펀치(구멍 뚫는 기구)'),
 '권':('?','券(-권), 權(-권) 접미 용법은 흔하나 단독 명사로는 드묾'), '띠':('Y','띠(십이지, 무슨 띠야)'),
 '법정':('Y','法定(법정 공휴일)'), '코트':('Y','coat(외투)'),
 '도':('Y','道(경기도), 島, 度, 계이름 도, 조사 도'), '포크':('?','folk(포크 음악)'),
 '뿔':('?','뿔나다(화나다, 속어)'), '라임':('Y','rhyme(랩 라임)'), '선물':('?','先物(선물 거래)'),
 '대변':('Y','代辯(대변인, 대변하다)'), '표':('Y','票(기차표, 투표), 標'), '수성':('?','水性(수성 펜/물감)'),
 '연':('Y','年(연 2회), 蓮(연꽃)'), '창':('Y','窓(창문), 唱'), '더미':('?','dummy(더미 데이터)'),
 '만':('Y','萬(만 원), 조사 만, 滿(만 나이)'), '공':('Y','空(숫자 0), 功(공을 세우다)'),
 '잼':('?','재미 줄임말(노잼, 꿀잼) 속어'), '우리':('Y','대명사 우리(we/our)'),
 '장기':('Y','長期, 長技(장기자랑), 將棋(장기 두다)'), '주문':('Y','注文(주문하다)'),
 '파리':('Y','Paris(도시)'), '초':('Y','초(candle), 初(월초), 醋(식초)'),
 '기사':('Y','記事, 技師, 운전기사, 騎士, 棋士'), '나이트':('Y','night(나이트클럽)'),
 '주식':('Y','株式 / 主食 (둘 다 이 데이터셋 문항)'), '의지':('Y','依支(의지하다)'),
 '문자':('Y','문자 메시지'), '수술':('?','수술(stamen, 꽃의 수술)'), '주석':('Y','註釋(주석을 달다), 主席(국가주석)'),
 '관사':('?','官舍(관사)'), '기간':('?','基幹(기간산업)'), '결장':('?','缺場(경기 결장)'),
 '장관':('Y','壯觀(장관을 이루다)'), '파운드':('Y','무게 단위 pound, 파운드케이크'),
 '제비':('Y','제비뽑기(lot)'), '부채':('Y','負債(빚)'),
 '내각':('?','內角(수학·야구 용어)'), '우편':('?','右便(오른편)'),
}
def hom(r):
    fl = r['auto_flags']
    out = {}
    if 'KO_HOMOGRAPH' in fl.replace('KO_HOMOGRAPH_REV',''):
        out['input'] = H.get(r['ko_input'], ('N',''))
    if 'KO_HOMOGRAPH_REV' in fl:
        out['rev'] = H.get(r['ko_reverse_input'], ('N',''))
    return out
hh = d.apply(hom, axis=1)
def comb(h):
    if not h: return ''
    v=[x[0] for x in h.values()]
    return 'Y' if 'Y' in v else ('?' if '?' in v else 'N')
d['ko_homograph_everyday_input'] = hh.map(lambda h: h.get('input',('',''))[0])
d['ko_homograph_everyday_rev'] = hh.map(lambda h: h.get('rev',('',''))[0])
d['ko_homograph_everyday'] = hh.map(comb)
def which(h, r):
    parts=[]
    if 'input' in h and h['input'][1]: parts.append(f"입력 {r['ko_input']}: {h['input'][1]}")
    if 'rev' in h and h['rev'][1]: parts.append(f"역방향 {r['ko_reverse_input']}: {h['rev'][1]}")
    return ' / '.join(parts)
d['ko_homograph_everyday_which'] = [which(h, r) for h, (_, r) in zip(hh, d.iterrows())]

# ---------- 3. 대조어 대체 제안 ----------
# id -> (제안 대조어 목록 [(단어, 어종)], 메모)
S = {
 'A010':([('동호회','한자어'),('학회','한자어')],'단체(5.19) 대체. 모임은 유지 가능'),
 'A015':([('메일','외래어'),('택배','한자어')],'엽서·소포 둘 다 대체'),
 'A024':([('책','한자어')],'입력 자(6.29)는 者·감탄사 자 등으로 빈도가 부풀려져 매칭 의미 없음. 범위 안 문구류는 없음(칼 4.94도 1.35 차). 입력어 교체(예: 줄자) 검토'),
 'A026':([('하늘','고유어'),('바람','고유어')],'구름(4.39) 대체. 달은 유지 가능'),
 'A035':([('물결','고유어'),('해수','한자어')],'파도(4.17) 대체. 조수는 유지 가능'),
 'A037':([('고양이','고유어'),('여우','고유어')],'햄스터(3.39) 대체. 토끼는 유지 가능'),
 'A040':([('마라톤','외래어'),('등산','한자어')],'씨름(3.61) 대체. 수영은 유지 가능'),
 'A045':([('무릎','고유어'),('허리','고유어')],'손목(4.15) 대체. 어깨는 유지 가능'),
 'A053':([('청혼','한자어'),('재혼','한자어')],'결혼·이혼 둘 다 대체. 혼인(4.10)도 가능'),
 'A054':([('보고서','한자어'),('계약서','한자어')],'공책(3.14) 대체. 봉투는 유지 가능'),
 'A056':([('덫','고유어')],'먹이(4.48) 대체. 미끼는 유지. 먹잇감(3.07)은 사냥감과 거의 동의어라 제외'),
 'A068':([('헬멧','외래어'),('네트','외래어')],'장갑(4.15) 대체. 그물은 유지 가능'),
 'A069':([('자리','고유어')],'칸(4.77) 대체. 입력 줄(5.81)은 동형어로 부풀려진 값'),
 'A072':([('노래','고유어'),('영화','한자어')],'동화(4.30) 대체. 소설은 유지 가능'),
 'A074':([('요요','외래어'),('썰매','고유어')],'인형(4.55) 대체. 구슬은 유지 가능'),
 'A080':([('학점','한자어'),('석사','한자어')],'졸업장(3.26) 대체. 자격증은 유지 가능'),
 'A086':([('휴지','한자어'),('폐기물','한자어')],'오물(3.30) 대체. 먼지는 유지 가능'),
 'A089':([('결론','한자어'),('반전','한자어')],'주제(4.92) 대체. 결말은 유지 가능'),
 'A090':([('꽃','고유어'),('생일','한자어')],'꽃다발(3.51) 대체. 편지는 유지 가능'),
 'A094':([('소파','외래어'),('옷장','혼종어')],'의자·침대 둘 다 대체. 서랍(3.62)도 가능'),
 'A099':([('이마','고유어'),('턱','고유어')],'입(6.31) 대체. 코는 유지 가능'),
 'A105':([('목구멍','고유어'),('치아','한자어')],'혀(4.42) 대체. 입술은 유지 가능'),
 'A115':([('비둘기','고유어'),('타조','한자어')],'오리(4.30) 대체. 거위는 유지 가능'),
 'A125':([],'입력 만(6.49)은 萬·조사 만 때문에 빈도가 크게 부풀려짐. 대조어 교체로 해결 불가 → 입력어 교체(예: 해만) 또는 문항 제외 검토'),
 'A128':([('양말','한자어'),('슬리퍼','외래어')],'운동화·구두 둘 다 대체. 샌들(3.04)도 가능'),
 'A132':([('회로','한자어'),('부품','한자어')],'컴퓨터(5.05) 대체. 배터리는 유지 가능'),
 'A146':([('목걸이','고유어'),('귀걸이','고유어')],'안경(4.31) 대체. 팔찌는 유지 가능'),
 'A155':([('주연','한자어'),('조연','한자어')],'주인공(4.81) 대체. 악당은 유지 가능'),
 'A156':([('입원','한자어'),('처방','한자어')],'진찰(3.34) 대체. 치료는 유지 가능'),
 'A167':([('동그라미','고유어')],'원(5.95, 화폐 단위 원과 겹침) 대체. 직사각형·오각형은 wordfreq 목록 밖(0)이라 제외. 삼각형은 유지'),
 'A176':([('젓가락','고유어'),('접시','고유어')],'컵(4.71) 대체. 숟가락은 유지 가능'),
 'A187':([('전투기','한자어'),('여객기','한자어')],'비행기(4.52) 대체. 헬리콥터는 유지 가능'),
 'A192':([('운하','한자어'),('부두','한자어')],'도로(4.93) 대체. 터널은 유지 가능'),
 'A194':([('라켓','외래어'),('글러브','외래어')],'공·장갑 둘 다 대체. 배트는 영어 bat가 새므로 제외'),
 'A195':([('헬멧','외래어'),('라켓','외래어')],'공(5.11) 대체. 글러브는 유지. 막대기는 A065 입력어라 제외'),
 'A200':([('상처','한자어')],'입력 점(5.51)은 占·-점 등으로 부풀려짐. 주근깨는 wordfreq 목록 밖. 흉터(3.93)·기미(3.85)는 차가 1 넘음'),
 'A201':([],'입력 못(6.13)은 부사 못(can\'t)과 표기가 같아 빈도 매칭 불가. 가장 가까운 칼(4.94)도 1.19 차 → 입력어 교체 또는 문항 제외 검토'),
 'A202':([('온도계','한자어'),('나침반','한자어')],'자·시계 둘 다 대체'),
 'A205':([('대결','한자어'),('승부','한자어')],'경기·대회 둘 다 대체'),
 'A207':([('감옥','한자어'),('죄수','한자어')],'법정(4.49) 대체. 교도소는 유지 가능'),
 'A208':([('악보','한자어'),('음표','한자어')],'현(5.05, 現 등으로 부풀려짐) 대체. 페달은 유지 가능'),
 'A212':([('총리','한자어'),('여당','한자어')],'국회·정부 둘 다 대체. 야당(4.35)·행정부(4.05)도 가능'),
 'A213':([('흥미','한자어'),('애정','한자어')],'호기심(3.94) 대체. 열정은 유지 가능'),
}
d['ctrl_suggest']=''; d['ctrl_suggest_zipf']=''; d['ctrl_suggest_etym']=''; d['ctrl_suggest_note']=''
for i, r in d.iterrows():
    if 'CTRL_FREQ_GAP' not in r['auto_flags']: continue
    words, note = S.get(r['id'], ([], 'TODO'))
    d.at[i,'ctrl_suggest'] = ';'.join(w for w,_ in words)
    d.at[i,'ctrl_suggest_zipf'] = ';'.join(str(zf(w)) for w,_ in words)
    d.at[i,'ctrl_suggest_etym'] = ';'.join(e for _,e in words)
    d.at[i,'ctrl_suggest_note'] = note
    for w,_ in words:
        assert zf(w) > 0 and abs(zf(w) - float(r['zipf_ko_input'])) <= 1.0, (r['id'], w)

d.to_csv('/mnt/project-files/dataset_A/A_candidates_reviewed.csv', index=False, encoding='utf-8-sig')
print(d.shape)
for c in ['cedict_answer_ok','cedict_distractor_ok','ko_homograph_everyday']:
    print(c, d[c].value_counts().to_dict())
print('ctrl rows', (d.ctrl_suggest_note!='').sum(), 'no suggestion', ((d.ctrl_suggest=='')&(d.ctrl_suggest_note!='')).sum())
