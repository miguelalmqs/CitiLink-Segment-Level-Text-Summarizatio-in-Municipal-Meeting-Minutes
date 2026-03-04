import os
import json
import pandas as pd
import nltk
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_calc

# Setup
nltk.download('wordnet')
nltk.download('punkt')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_overall_metrics(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    bleu_scorer = BLEU()

    # Storage for all scores to calculate global mean
    all_scores = {
        "ROUGE1": [], "ROUGE2": [], "ROUGEL": [], "ROUGELSUM": [],
        "METEOR": [], "BLEU": []
    }
    all_refs = []
    all_cands = []

    for entry in data:
        ref = entry['resumo_referencia']
        cand = entry['resumo_modelo']

        # ROUGE
        rs = r_scorer.score(ref, cand)
        all_scores["ROUGE1"].append(rs['rouge1'].fmeasure)
        all_scores["ROUGE2"].append(rs['rouge2'].fmeasure)
        all_scores["ROUGEL"].append(rs['rougeL'].fmeasure)
        all_scores["ROUGELSUM"].append(rs['rougeLsum'].fmeasure)

        # BLEU & METEOR
        all_scores["BLEU"].append(bleu_scorer.sentence_score(cand, [ref]).score)
        all_scores["METEOR"].append(meteor_score([nltk.word_tokenize(ref)], nltk.word_tokenize(cand)))

        all_refs.append(ref)
        all_cands.append(cand)

    # BERTScore (Vectorized for speed)
    P, R, F1 = bert_score_calc(all_cands, all_refs, lang="pt", verbose=False)

    # Combine everything into a final average
    overall = {
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum", "METEOR", "BLEU", "BS_P", "BS_R", "BS_F1"],
        "Value": [
            sum(all_scores["ROUGE1"]) / len(data),
            sum(all_scores["ROUGE2"]) / len(data),
            sum(all_scores["ROUGEL"]) / len(data),
            sum(all_scores["ROUGELSUM"]) / len(data),
            sum(all_scores["METEOR"]) / len(data),
            sum(all_scores["BLEU"]) / len(data),
            P.mean().item(),
            R.mean().item(),
            F1.mean().item()
        ]
    }

    return pd.DataFrame(overall)

# Execute
df_overall = calculate_overall_metrics(os.path.join(SCRIPT_DIR, 'summaries_and_results', 'generated_summaries.json'))
print(df_overall)
