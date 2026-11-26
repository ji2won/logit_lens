import math, wordfreq, pandas as pd
fd = wordfreq.get_frequency_dict('ko')
zf = lambda w: round(math.log10(fd[w]*1e9),2) if w in fd else 0.0
import os
P=os.path.dirname(os.path.abspath(__file__))+'/'
d = pd.read_csv(P+'A_candidates_reviewed.csv', keep_default_na=False)
OKS={'OK','COMPOSITIONAL'}
def reasons(r):
    out=[]
    if r.cedict_answer_ok not in OKS: out.append(f'CEDICT_ANSWER_{r.cedict_answer_ok}')
    if r.cedict_distractor_ok not in OKS: out.append(f'CEDICT_DISTRACTOR_{r.cedict_distractor_ok}')
    if r.ko_homograph_everyday=='Y': out.append('KO_HOMOGRAPH_EVERYDAY')
    if r.ko_homograph_everyday=='?': out.append('KO_HOMOGRAPH_BORDERLINE')
    if r.id in ('A125','A201'): out.append('CTRL_UNMATCHABLE')
    return ';'.join(out)
d['exclude_reason']=d.apply(reasons,axis=1)
d['keep']=d.exclude_reason.eq('').map({True:'Y',False:'N'})
# 대조어 최종
fin=[];finz=[];chg=[]
for _,r in d.iterrows():
    orig=r.ko_control.split(';'); oz=[float(x) for x in str(r.zipf_ko_control).split(';')]
    if r.ctrl_suggest:
        sug=r.ctrl_suggest.split(';'); new=[]
        for w,z in zip(orig,oz):
            if z and abs(z-float(r.zipf_ko_input))<=1.0: new.append(w)
            else: new.append(sug.pop(0) if sug else w)
        fin.append(';'.join(new)); finz.append(';'.join(str(zf(w)) for w in new)); chg.append('Y' if new!=orig else 'N')
    else:
        fin.append(r.ko_control); finz.append(str(r.zipf_ko_control)); chg.append('N')
d['ko_control_final']=fin; d['zipf_ko_control_final']=finz; d['ko_control_changed']=chg
d.to_csv(P+'A_candidates_reviewed.csv',index=False,encoding='utf-8-sig')
k=d[d.keep=='Y']
k.to_csv(P+'A_final.csv',index=False,encoding='utf-8-sig')
print(len(k),k.en.nunique(),k.direction.value_counts().to_dict(),'ctrl changed',(k.ko_control_changed=='Y').sum())
# 검증: 남은 행 대조어 빈도
bad=0
for _,r in k.iterrows():
    for z in r.zipf_ko_control_final.split(';'):
        if float(z)==0 or abs(float(z)-float(r.zipf_ko_input))>1: bad+=1
print('remaining gap/zero controls',bad)
x=d[d.keep=='N']
print(x.exclude_reason.str.split(';').explode().value_counts().to_dict())
print('A_dropped by answer', (x.exclude_reason.str.contains('ANSWER')).sum())
for _,r in x.iterrows(): print(r.id,r.en,r.direction,r.exclude_reason)
# 3단계 기준 밖이지만 빈도 확인이 안 된 행 표시(제외하지는 않음)
def fc(r):
    if float(r.zipf_ko_input)==0: return 'INPUT_ZERO'
    if any(float(z)==0 for z in r.zipf_ko_control_final.split(';')): return 'CTRL_ZERO'
    return ''
d['freq_unverified']=d.apply(fc,axis=1)
fu=d.freq_unverified!=''
d.loc[fu,'exclude_reason']=d.loc[fu,'exclude_reason'].map(lambda x: x+';' if x else '')+'FREQ_'+d.loc[fu,'freq_unverified']
d['keep']=d.exclude_reason.eq('').map({True:'Y',False:'N'})
d.to_csv(P+'A_candidates_reviewed.csv',index=False,encoding='utf-8-sig')
k=d[d.keep=='Y']; k.to_csv(P+'A_final.csv',index=False,encoding='utf-8-sig')
print(k.freq_unverified.value_counts().to_dict(), 'fully clean', (k.freq_unverified=='').sum(), k[k.freq_unverified==''].en.nunique())

x=d[d.keep=='N'].exclude_reason.str.split(';').explode().value_counts().to_dict(); print(x)
print('final',len(k),k.en.nunique(),k.direction.value_counts().to_dict(),'ctrl changed',(k.ko_control_changed=='Y').sum())
import itertools
r=d.exclude_reason
def has(p): return r.str.contains(p)
c1=has('CEDICT'); c2=has('KO_HOMOGRAPH'); c3=has('CTRL_UNMATCHABLE'); c4=has('FREQ_')
print('unique-by-step', 'c1',c1.sum(),'c2Y',has('EVERYDAY').sum(),'c2?',has('BORDERLINE').sum(),'c3',c3.sum(),'c4',c4.sum(), 'total drop',(d.keep=='N').sum())
