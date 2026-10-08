"""Conservative V1 English MCQ candidate construction; no splitting or clients.

Original source fields are copied unchanged. Transparent regex screens are
proxies, not gold knowledge labels. Independent sample review is a separate step.
"""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,re,unicodedata
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/bhashabench_clean'
REPORTS=ROOT/'reports'
EVIDENCE=REPORTS/'bhashabench_evidence'
DOMAINS=['ayur','legal','krishi']
OPTIONS=['option_a','option_b','option_c','option_d']
VERSION='conservative-content-v4'
SEED=42

# Every screen is content-based; native subject/topic never determines inclusion.
PATTERNS={
 'special_option':r'\b(?:all|none|both|neither)\b|\bany\s+(?:of\s+)?(?:the\s+)?(?:above|these|them|following)|\b(?:only\s+)?\(?(?:[a-e1-5]|i|ii|iii|iv|v)\)?\s*(?:and|&|,)\s*\(?(?:[a-e1-5]|i|ii|iii|iv|v)\)?|\b(?:a|b|c|d|i|ii|iii|iv)\s*[-–]\s*\(?[ivx]+\)?',
 'material_dependency':r'\b(?:passage|paragraph|diagram|graph|table|figure|image|picture|chart|illustration|case study|given information|following information|above information|factual situation)\b|<@Table>|\bcontext\s*:',
 'calculation':r'\b(?:calculate|compute|solve|simplify|evaluate|data sufficien\w*|number series|seating|arrangement|upstream|downstream|probability of|average of|median of|mean of|percentage more|percentage less|quantity i|quantity ii|sum of|find the (?:value|ratio|sum|average)|how much|cost price|selling price|simple interest|compound interest)\b|\b\d+\s*(?:days|hours|km|litres|liters)\b.{0,50}\b(?:work|speed|distance|mixture|ratio)\b|\b(?:2n|n)\s*=\s*\d+|[×÷√²³]|\b(?:statements?|conclusions?)\s*[:：].{0,100}[<>≥≤]',
 'application':r'\b(?:assertion|reason\s*\(?r\)?|conclusions?|infer\w*|justify|explain why|why\b|because|suppos\w*|assum\w*|hypothetical|scenario|case study|a patient|a farmer|a man|a woman|a boy|a girl|years old|aged\s+\d+|mr\.|mrs\.|example of|not an example|which.*example)\b|\b(?:shyam|ramesh|suresh|raman|dhawan|chaman)\b|\b[a-z]\b.{0,20}\b(?:kills|murders|robs|sells|deposits|borrows|lends|shoots|girlfriend)\b|\b(?:plaintiff|defendant|accused)\b.{0,70}\b(?:on\s+\d|despite|his servant|his friend|he did|she did)\b',
 'off_domain':r'\b(?:synonym of the word|antonym|synonyms? of (?:word|the word)|correctly spelt|spell\w*|idiom|grammatic\w*|sentence|vocabulary|odd one out|opposite in meaning|similar in meaning|dissimilar|ms[ -]?(?:excel|word)|microsoft|powerpoint|browser|webrowser|keyboard|shortcut|folder|cpu|booting|hard disk|computer|html|olympics?|mascot|film|movie|dadasaheb|padma vibhushan|nobel prize|award\w*|recently|current affairs|birthplace|birthday|born on|date of birth|present minister|appointed.*minister|brand ambassador|rail tunnel|longest coastline|satyajit|literary|anandamath)\b',
 'corrupted':r'\ufffd|\[ocr|missing\]|\b(?:edu ?tap|adda247|agriaddict|chosen option)\b|[\x00-\x08\x0b\x0c\x0e-\x1f]|Ã.|Â.|â€|ï¿',
 'referent_missing':r'^\s*(?:this|these|it|they|he|she)\s+(?:is|are|has|have)|\b(?:above|below)\s+(?:diagram|graph|table|figure|passage)|\b(?:as per|according to)\s+(?:the\s+)?passage|\b(?:write (?:a )?short note|discuss|essay)\b',
}
DOMAIN_PATTERNS={
 'ayur':r'\b(?:ayur\w*|charak\w*|sushrut\w*|vagbh\w*|vagbhat\w*|kashyap\w*|sharang\w*|samhita|acharya|ashtang\w*|yograt\w*|rasarat\w*|dosha\w*|dhatu\w*|dhatusar\w*|marma\w*|basti\w*|sneha\w*|nasya\w*|vaman\w*|virech\w*|rasa\w*|bhai\w*|prakr\w*|guna\w*|chikitsa\w*|shodhan\w*|sweda\w*|panchak\w*|oja\w*|vata\w*|pitta\w*|kapha\w*|aushadh\w*|dravya\w*|bhasma\w*|yantra\w*|agni\w*|shukra\w*|stanya\w*|garbhin\w*|garbha\w*|sutika\w*|anupa\w*|anjan\w*|twak|shastra|srot\w*|mutra\w*|jwara?\w*|unmad\w*|kamala|pandu\w*|atisara|madatyaya|ajeerna|shiro\w*|harita|trijata|trikarshik|pippali\w*|guduchi|haritaki|triphala|shunthi|must[a]?|kutaja|gokshura|hingula|parad|manjistha|chandras\w*|garden cress|eranda|vacha|nimb[a]?|brahmi|latakaranja|bhallatak|yashti\w*|shankh\w*|guggulu|amalaki|ashwagandha|shatavari|tulsi|ghee|ghrita|kwatha|kalka|asava|arishta|avala?eha|kshir\w*|tal[au]|netra\w*|sharir\w*|sira\w*|\w*roga\w*|\w*vyadhi\w*|\w*vyapad\w*|\w*yon\w*|\w*marma\w*|blood|leucocyte\w*|leukocyte\w*|erythro\w*|haem\w*|hem\w*|arter\w*|vein\w*|nerve\w*|brain|muscle\w*|humerus|femur|tibia|anatom\w*|bone\w*|heart|cardiac|renal|kidney|liver|lung\w*|spleen|thyroid|parathyroid|hormone\w*|aldosterone|insulin|pituitary|pancrea\w*|neuron\w*|neonat\w*|foet\w*|fetus|embryo\w*|pregnan\w*|placenta\w*|labour|labor|ovum|ova|uter\w*|ovari\w*|labia|vagina\w*|cervi\w*|cornea|iris|pupil|retina\w*|ophthalm\w*|eye|ocular|hearing|deafness|tonsil\w*|ear|tongue|skin|urticaria|immun\w*|vaccine\w*|infection|bacter\w*|virus|viral|fung\w*|mycosis|measles|tuberc\w*|leprosy|malaria|dengue|hiv|aids|diabet\w*|rheumatoid|ulcer|constipation|diarr\w*|hypocal\w*|hypercal\w*|goitre|mortem|poison\w*|toxic\w*|scorpion|snake|dapsone|rifampi\w*|erythropoietin|bilirubin|substantia|icds|pasteur\w*|vitamin|night blindness|infant|newborn|disease\w*|symptom\w*|syndrome|enzyme|physiol\w*)\b',
 'legal':r'\b(?:law\w*|legal|legislat\w*|constitution\w*|article|section|act|code|ipc|cpc|crpc|cr\.p\.c\.|i\.p\.c\.|evidence|court\w*|judge\w*|judicial|justice|magistrate|tribunal|juris\w*|contract\w*|tort\w*|penal|offen[cs]e\w*|crime|criminal|prosecut\w*|arrest\w*|bail\w*|acquitt\w*|convict\w*|imprison\w*|death sentence|warrant|summons|suit\w*|plaint\w*|defendan\w*|litigat\w*|appeal\w*|revision|injunction|decree\w*|judg?ment\w*|arbitrat\w*|mediat\w*|conciliation|plea bargaining|limitation|adverse possession|res judicata|estoppel|caveat|doctrine|jurisprudence|mortgage\w*|transfer of property|succession|inherit\w*|guardian\w*|divorce|marriage act|dower|mahr|writ\w*|certiorari|mandamus|habeas|prohibition|protection act|patent\w*|intellectual property|copyright|trademark\w*|compulsory licens\w*|specific relief|registration|stamp duty|compensation|damages|natural justice|mens rea|actus reus|nemo dat|force majeure|public international|un charter|u\.n\.|world intellectual|wipo|lok adalat|negotiable instruments|promissory|dishono\w*|tahsildar|land revenue|wazib|accommodation control|uniform civil|trade union|batna|immunity|immunities|accused|property right\w*)\b',
 'krishi':r'\b(?:agri\w*|crop\w*|plant\w*|seed\w*|soil\w*|fertili\w*|nitrogen\w*|nitrat\w*|nitrit\w*|phosph\w*|potass\w*|manure|compost|vermi\w*|irrigat\w*|drip|sprinkler|tillage|cultivat\w*|farming|farm\w*|horticul\w*|germinat\w*|pollinat\w*|propagat\w*|variet\w*|cultivar\w*|root\w*|shoot\w*|stem|leaf|leaves|flower\w*|fruit\w*|vegetable\w*|cereal\w*|oilseed\w*|legum\w*|pulse\w*|wheat|rice|paddy|maize|corn|barley|oat|cotton|sugar\w*|sorghum|jowar|bajra|millet\w*|groundnut|soybean|soyabean|mustard|sesam\w*|mango|banana|citrus|orange|guava|papaya|grape\w*|peach|plum|avocado|tomato|potato|cauliflower|cabbage|okra|onion|garlic|chilli|pepper|brinjal|celery|spinach|palak|pumpkin|gourd|jute|tea|coffee|cocoa|coconut|sunflower|linseed|alfalfa|lucerne|safflower|clover|rose|chrysanthem\w*|hollyhock|sweet pea|litchi|cherry|apple|pomegranate|pineapple|ber|pusa|arka|fhia|neelphonso|pjhm|grafting|budding|layering|pruning|greenhouse|green house|nursery|landscap\w*|hedge\w*|postharvest|post harvest|stored|storage|weeds?|herbicid\w*|pesticid\w*|insect\w*|insecticid\w*|nematod\w*|pathogen\w*|fung\w*|disease|pest\w*|larv\w*|entom\w*|aphid\w*|whitefly|beetle|thrips|moth|mite\w*|butterfly|honeybee|locust|borer|termit\w*|flies|fly|mosquito|hymenop\w*|pheromone|antenna\w*|mouthpart\w*|mouth parts|biotroph\w*|mycorr\w*|albugo|puccinia|blumeria|glomus|phytoplasma|phytophthora|fusarium|alternaria|rhizoctonia|taphrina|bordeaux|sclerot\w*|oospore|zoospore|ascocarp|apothecium|cleistothecium|synnema|perithecium|bacillus|rhizob\w*|nitrobacter|nitros\w*|nitrococcus|hered\w*|genet\w*|genom\w*|chromosom\w*|haploid|diploid|polyploid|meiosis|mitosis|dna|rna|t-rna|trna|rrna|mrna|rflp|aflp|ssr|snps?|marker\w*|recombinant|transgenic|totipoten\w*|photosynth\w*|chloroplast|chlorophyll|plastid\w*|chromoplast|peroxisome|phytoh\w*|phytoharmone|auxin|ethylene|gibberel\w*|cytokinin|abscisic|salicylic|stomata\w*|xylem|phloem|carotenoid|enzyme|protein|amino acid|cell|cells|embryo|tissue|plasmid|cosmid|shuttle vector|phage|epn|cryopreserv\w*|cryo\w*|gms|cgms|pgms|cattle|bull|cow|cows|bovine|poultry|broiler|chicken|layer|dairy|milk|goat|sheep|pig|swine|veterin\w*|livestock|fish\w*|aquacult\w*|semen|udder|rumin\w*|rumen|fodder|forage|septicaemia|poikilothermic|agroforest\w*|forestry|forest cover|silvicult\w*|pmfby|pmksy|pm kisan|e-chaupal|e-choupal|extension education|extension program|rural development|etawah|lab to land|icar|cimap|crida|green revolution|kharif|rabi|zaid|agroclimat\w*|agrometeoro\w*|albedo|rainfall|weathering|mineral|quartz|erosion|pedon|ecotype|ecological|arable|salin\w*|sodic\w*|acid soil|water potential|wien|radiation|windmill|watershed|soil conservation)\b',
}
DOMAIN_PATTERNS['legal']=DOMAIN_PATTERNS['legal'].replace('writ\\w*','writs?')
COMPILED={k:re.compile(v,re.I|re.S) for k,v in PATTERNS.items()}
DOMAIN_RX={k:re.compile(v,re.I) for k,v in DOMAIN_PATTERNS.items()}
PRIORITY=['non_mcq','corrupted','not_self_contained','non_english','special_option','calculation_aptitude','reasoning_application','off_domain','duplicate']

def norm(s):
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in str(s or '').lower()).split())

def assess(domain,r):
    reasons=[];rules=[]
    def flag(reason,rule):
        if reason not in reasons:reasons.append(reason)
        rules.append(rule)
    q=str(r.question or '');opts=[str(r[c] or '') for c in OPTIONS]
    a=str(r.correct_answer or '').strip()
    if r.question_type!='MCQ':flag('non_mcq','F01_native_type_exact_MCQ')
    if not q.strip() or any(not x.strip() for x in opts) or a not in list('ABCD'):
        flag('not_self_contained','F02_missing_or_invalid_Q_options_answer')
    if len(set(norm(x) for x in opts))<4:flag('not_self_contained','F03_duplicate_option_text')
    if any(norm('1 '+x) in [norm(v) for v in opts] for x in opts):flag('not_self_contained','F03b_unit_vs_one_unit_duplicate_option')
    if any(not re.search(r'[A-Za-z0-9]',x) or re.fullmatch(r'\s*\(?[a-d]\)?[).:]?\s*',x,re.I) for x in opts):
        flag('corrupted','F04_empty_label_or_unreadable_option')
    if sum(bool(re.fullmatch(r'\s*\d+\s*',x)) for x in opts)>=3 and any(re.fullmatch(r'\s*[lIoO]\s*',x) for x in opts):
        flag('corrupted','F04b_digit_vs_OCR_letter_option')
    if COMPILED['corrupted'].search(' '.join([q,*opts])):flag('corrupted','F05_OCR_control_mojibake_markers')
    if re.search(r'[\u0900-\u097f]',q+' '.join(opts)):flag('non_english','F06_Devanagari_in_native_English_file')
    words=re.findall(r'[A-Za-z]+',q)
    if len(words)<3 or len(''.join(words))<10:flag('not_self_contained','F07_too_little_question_content')
    if COMPILED['referent_missing'].search(q):flag('not_self_contained','F08_missing_referent_or_nonchoice_instruction')
    if re.search(r'even after it is\s*$|\band\s*[._…-]*\s*$',q,re.I):flag('not_self_contained','F08b_unfinished_dependent_clause')
    if re.search(r'following are all characteristics',q,re.I):flag('not_self_contained','F08c_ambiguous_all_characteristics_singlechoice')
    if not re.search(r'\b(?:is|are|was|were|has|have|had|will|shall|can|could|may|must|should|which|what|who|where|when|how|why|name|number|total|count|types?|quantity|praman\w*|matra|dose|weight|size|length|route|time|period|gana|prognosis|indication|anupana|content|sankhya|chapter|definition|meaning|synonym|location|family|symptom|cause|risk|treatment|function|preparation|dravyas?|composition|constitution|site|classification|structure)\b',q,re.I):
        flag('not_self_contained','F08d_no_explicit_question_relation_conservative')
    if any(re.search(r'[._…-]{3,}\s*'+re.escape(x.strip())+r'\s*$',q,re.I) for x in opts if len(x.strip())>=4):
        flag('not_self_contained','F08e_blank_followed_by_trailing_option_text_possible_leak')
    if re.search(r'(?:\bI\s*[-–]\s*[IVX]+|\(?[A-D]\)?\s*[-–]\s*\(?[IVX]+)',q,re.I) or re.search(r'\b(?:match list|list[- ]?i|statement\s*\(?[i12]\)?)\b',q,re.I):
        flag('special_option','F09_matching_multi_statement_wrapper')
    if any(COMPILED['special_option'].search(x) for x in opts):flag('special_option','F10_all_none_both_or_combination_options')
    if COMPILED['material_dependency'].search(q):flag('reasoning_application','C01_reject_all_material_context_wrappers_conservatively')
    if COMPILED['calculation'].search(q):flag('calculation_aptitude','C02_arithmetic_formal_aptitude_cues')
    if re.search(r'\b(?:work out|agronomic efficiency|croprotation intensity|crop rotation intensity|statistical test|data.{0,80}collected|significantly different|code language|coding|decoding)\b',q,re.I):flag('calculation_aptitude','C02b_quantitative_experiment_or_coding_cues')
    if COMPILED['application'].search(q):flag('reasoning_application','C03_case_causal_or_inference_cues')
    if re.search(r'\b(?:young lady|young woman|comes with|presents with|pap smear shows|if we take)\b',q,re.I):flag('reasoning_application','C03b_clinical_vignette_or_hypothetical_example')
    if domain=='legal' and (len(re.findall(r'(?<!\w)[A-DXYZ](?!\w)',q))>=2 or re.search(r"(?<!\w)[A-DXYZ]['’](?!\w)",q)):
        flag('reasoning_application','C03c_named_letter_legal_hypothetical_conservative')
    if COMPILED['off_domain'].search(q):flag('off_domain','C04_generic_English_computing_trivia_news_cues')
    if not DOMAIN_RX[domain].search(q):flag('off_domain','C05_no_positive_domain_evidence_in_question_conservative')
    if len(q.split())>60:flag('reasoning_application','C06_long_question_conservative_60_words')
    return reasons,rules

class UF:
    def __init__(self,n):self.p=list(range(n));self.size=[1]*n
    def find(self,a):
        while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.p[b]=a;self.size[a]+=self.size[b]

def near_edges(questions):
    """Exact set-Jaccard join at 17/20, with global-order prefix filtering.

    Prefix length |S|-ceil(.85|S|)+1 plus length filtering cannot miss pairs at
    this threshold. This replaces the exploratory MinHash approximation.
    Exact normalized copies are already unioned and excluded from near edges.
    """
    norms=[norm(q) for q in questions];grams=[];frequency=Counter()
    for text in norms:
        w=text.split();g={' '.join(w[j:j+3]) for j in range(len(w)-2)}
        grams.append(g);frequency.update(g)
    buckets=defaultdict(list);edges=[];proposals=0
    for i in sorted(range(len(grams)),key=lambda k:(len(grams[k]),k)):
        g=grams[i];n=len(g)
        if not n:continue
        prefix=sorted(g,key=lambda t:(frequency[t],t))[:n-(17*n+19)//20+1]
        matches=set()
        for token in prefix:
            for j in buckets[token]:
                if len(grams[j])*20>=n*17 and norms[j]!=norms[i]:matches.add(j)
        proposals+=len(matches)
        for j in sorted(matches):
            inter=len(g&grams[j]);union=n+len(grams[j])-inter
            if inter*20>=union*17:edges.append((min(i,j),max(i,j),inter/union))
        for token in prefix:buckets[token].append(i)
    return edges,proposals

def choose(indices,frame):
    # A deterministic source-ID hash does not prefer domain/difficulty/answer.
    return min(indices,key=lambda i:hashlib.sha256((frame.at[i,'domain']+'|'+str(frame.at[i,'id'])).encode()).hexdigest())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    snapshots=[];frames=[]
    for domain in DOMAINS:
        path=ROOT/f'data/bhashabench/{domain}/test.parquet'
        frame=pd.read_parquet(path)
        snapshots.append({'domain':domain,'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(frame)})
        frame.insert(0,'source_row_index',frame.index);frame.insert(0,'domain',domain)
        frames.append(frame)
    data=pd.concat(frames,ignore_index=True).fillna('')
    data.insert(0,'global_row_index',data.index)
    decisions=[assess(r.domain,r) for _,r in data.iterrows()]
    reasons=[v[0] for v in decisions];rules=[v[1] for v in decisions]
    stems=data.question.map(norm).tolist()
    qa=[json.dumps([stems[i],*[norm(r[c]) for c in OPTIONS],norm(r.correct_answer)],ensure_ascii=False) for i,r in data.iterrows()]
    exact=defaultdict(list);stemgroups=defaultdict(list)
    for i,key in enumerate(qa):exact[key].append(i)
    for i,key in enumerate(stems):
        if key:stemgroups[key].append(i)
    uf=UF(len(data));edges=[]
    for kind,groups in [('normalized_qa',exact),('normalized_stem',stemgroups)]:
        for group in groups.values():
            for i in group[1:]:uf.union(group[0],i);edges.append((group[0],i,kind,1.0))
    nears,proposals=near_edges(data.question.tolist())
    for a,b,score in nears:uf.union(a,b);edges.append((a,b,'near_trigram_jaccard',score))
    clusters=defaultdict(list)
    for i in range(len(data)):clusters[uf.find(i)].append(i)
    cluster_ids={root:'bbv1_'+hashlib.sha256('|'.join(sorted(data.at[i,'domain']+':'+str(data.at[i,'id']) for i in group)).encode()).hexdigest()[:20] for root,group in clusters.items()}
    # Different option orders are fine if the resolved correct option text agrees.
    conflicts=[]
    for key,group in stemgroups.items():
        targets={norm(str(data.at[i,OPTIONS['ABCD'.index(str(data.at[i,'correct_answer']))]])) for i in group if str(data.at[i,'correct_answer']) in 'ABCD' and len(str(data.at[i,'correct_answer']))==1}
        if len(targets)>1:
            conflicts.append({'stem':key,'rows':group,'resolved_answer_texts':sorted(targets)})
            for i in group:
                if 'not_self_contained' not in reasons[i]:reasons[i].append('not_self_contained')
                rules[i].append('D00_conflicting_targets_for_exact_normalized_stem_conservative')
    eligible=[i for i,r in enumerate(reasons) if not r]
    stageqa=[]
    byqa=defaultdict(list)
    for i in eligible:byqa[qa[i]].append(i)
    stageqa=[choose(g,data) for g in byqa.values()]
    bystem=defaultdict(list)
    for i in stageqa:bystem[stems[i]].append(i)
    stagestem=[choose(g,data) for g in bystem.values()]
    representatives={}
    for root,group in clusters.items():
        good=[i for i in group if not reasons[i]]
        if good:representatives[root]=choose(good,data)
    selected=set(representatives.values())
    for i in eligible:
        if i not in selected:
            reasons[i].append('duplicate')
            rules[i].append('D01_one_eligible_representative_per_global_duplicate_cluster')
    data['cluster_id']=[cluster_ids[uf.find(i)] for i in range(len(data))]
    data['primary_removal_reason']=[next((reason for reason in PRIORITY if reason in r),'') for r in reasons]
    data['all_removal_reasons']=[json.dumps(r,ensure_ascii=False) for r in reasons]
    data['removal_rule_ids']=[json.dumps(r,ensure_ascii=False) for r in rules]
    data['selected_candidate']=[i in selected for i in range(len(data))]
    data.to_parquet(OUT/'cleaning_decisions.parquet',index=False)
    removed=data[~data.selected_candidate].copy()
    removed.to_csv(REPORTS/'removed_examples.csv',index=False,encoding='utf-8-sig')
    stats=[];lengths={};difficulty=[]
    extra=['domain','source_row_index','cluster_id','cleaning_rule_version','source_repo','source_split']
    for domain in DOMAINS:
        native_columns=frames[DOMAINS.index(domain)].columns.drop(['domain','source_row_index']).tolist()
        candidate=data[(data.domain==domain)&data.selected_candidate].copy()
        candidate['cleaning_rule_version']=VERSION
        candidate['source_repo']=f'bharatgenai/BhashaBench-'+{'ayur':'Ayur','legal':'Legal','krishi':'Krishi'}[domain]
        candidate['source_split']='test'
        candidate=candidate[native_columns+extra].sort_values('source_row_index').reset_index(drop=True)
        assert len(candidate)>=300, f'{domain}: insufficient for requested sample300'
        assert candidate.cluster_id.is_unique
        candidate.to_parquet(OUT/f'{domain}_candidates.parquet',index=False)
        candidate.to_json(OUT/f'{domain}_candidates.jsonl',orient='records',lines=True,force_ascii=False)
        samples=candidate.sample(n=300,random_state=SEED).copy()
        samples.insert(0,'final_audit_index',range(300))
        samples.to_csv(REPORTS/f'final_audit_{domain}_300.csv',index=False,encoding='utf-8-sig')
        raw=data[data.domain==domain];rem=removed[removed.domain==domain]
        row={'domain':domain,'original':len(raw),'native_mcq':int(raw.question_type.eq('MCQ').sum()),
             'content_and_format_pass_before_dedup':sum(data.at[i,'domain']==domain for i in eligible),
             'after_normalized_qa_dedup':sum(data.at[i,'domain']==domain for i in stageqa),
             'after_normalized_stem_dedup':sum(data.at[i,'domain']==domain for i in stagestem),
             'after_cleaning':len(candidate),'retention_rate':len(candidate)/len(raw)}
        for reason in PRIORITY:row['removed_'+reason]=int(rem.primary_removal_reason.eq(reason).sum())
        assert row['original']==row['after_cleaning']+sum(row['removed_'+r] for r in PRIORITY)
        stats.append(row)
        lengths[domain]={}
        for kind,values in [('question_chars',candidate.question.str.len()),('question_words',candidate.question.str.split().str.len())]:
            lengths[domain][kind]={'mean':float(values.mean()),'median':float(values.median()),'p10':float(values.quantile(.1)),'p90':float(values.quantile(.9)),'max':int(values.max())}
        for level,n in candidate.question_level.value_counts(dropna=False).items():difficulty.append({'domain':domain,'difficulty':level,'count':int(n),'rate':n/len(candidate)})
        pd.testing.assert_frame_equal(candidate,pd.read_parquet(OUT/f'{domain}_candidates.parquet'))
        pd.testing.assert_frame_equal(candidate,pd.read_json(OUT/f'{domain}_candidates.jsonl',lines=True,dtype=False),check_dtype=False)
        # Original values and source row alignment must be preserved exactly.
        original=frames[DOMAINS.index(domain)].set_index('source_row_index')
        for c in native_columns:assert candidate[c].tolist()==original.loc[candidate.source_row_index,c].tolist(),(domain,c)
    cross=[]
    for root,group in clusters.items():
        ds={data.at[i,'domain'] for i in group}
        if len(ds)>1:cross.append({'cluster_id':cluster_ids[root],'domains':sorted(ds),'source_rows':group,'selected_row':representatives.get(root)})
    # Across all candidate files exactly one row may survive a global cluster.
    allselected=data[data.selected_candidate]
    assert allselected.cluster_id.is_unique and not allselected.question.map(norm).duplicated().any()
    pd.DataFrame(stats).to_csv(REPORTS/'cleaning_stats.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(difficulty).to_csv(REPORTS/'cleaning_difficulty.csv',index=False,encoding='utf-8-sig')
    (EVIDENCE/'cleaning_rules.json').write_text(json.dumps({'version':VERSION,'seed':SEED,'patterns':PATTERNS,'positive_domain_question_patterns':DOMAIN_PATTERNS,'primary_reason_priority':PRIORITY,'notes':'Native subject/topic never controls inclusion; no previous manual labels used; automatic screens are conservative proxies'},ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(edges,columns=['global_row_a','global_row_b','edge_type','similarity']).to_csv(EVIDENCE/'cleaning_duplicate_edges.csv',index=False)
    (EVIDENCE/'cleaning_conflicting_stems.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf8')
    (EVIDENCE/'cleaning_cross_domain_clusters.json').write_text(json.dumps(cross,ensure_ascii=False,indent=2),encoding='utf8')
    result={'version':VERSION,'source_files':snapshots,'stats':stats,'question_lengths':lengths,'difficulty':difficulty,
            'near_candidate_pairs':proposals,'confirmed_near_pairs':len(nears),'cluster_count':len(clusters),'largest_cluster':max(map(len,clusters.values())),
            'cross_domain_clusters':len(cross),'conflicting_exact_stem_groups':len(conflicts),
            'previous_manual_samples_used_for_labels':False,'purity_review_status':'PENDING_INDEPENDENT_300_PER_DOMAIN_REVIEW'}
    (EVIDENCE/'cleaning_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    for snapshot in snapshots:assert hashlib.sha256((ROOT/snapshot['path']).read_bytes()).hexdigest()==snapshot['sha256']
    print(pd.DataFrame(stats).to_string(index=False),flush=True)
    print('NEAR',len(nears),'CROSS-DOMAIN CLUSTERS',len(cross),'CONFLICTING EXACT STEMS',len(conflicts),flush=True)
    for p in [*OUT.glob('*'),REPORTS/'removed_examples.csv',REPORTS/'cleaning_stats.csv',*REPORTS.glob('final_audit_*_300.csv')]:print('OUTPUT',p,p.stat().st_size,'bytes',flush=True)

if __name__=='__main__':main()
