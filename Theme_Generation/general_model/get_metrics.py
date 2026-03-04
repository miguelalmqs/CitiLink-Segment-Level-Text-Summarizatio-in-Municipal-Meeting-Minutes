import os
import json
import numpy as np
import evaluate
from bert_score import score

# --- CAMINHOS RELATIVOS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1️⃣ Carregar os dados
# ============================================================

RESULTS_PATH = os.path.join(SCRIPT_DIR, "summaries_and_results", "mbart_evaluation_results.json")
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

refs = [item["tema_anotado"] for item in data]
hyps = [item["tema_gerado"] for item in data]

# ============================================================
# 2️⃣ ROUGE
# ============================================================

rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=hyps, references=refs)

# extrair F1
rouge_1 = rouge_result['rouge1']
rouge_2 = rouge_result['rouge2']
rouge_L = rouge_result['rougeL']

print("=== ROUGE ===")
print(f"ROUGE-1 F1: {rouge_1:.4f}")
print(f"ROUGE-2 F1: {rouge_2:.4f}")
print(f"ROUGE-L F1: {rouge_L:.4f}")

# ============================================================
# 3️⃣ BLEU (corrigido para evaluate)
# ============================================================

bleu = evaluate.load("bleu")

# Apenas strings, não tokenizadas
bleu_result = bleu.compute(predictions=hyps, references=[[r] for r in refs])

print("\n=== BLEU ===")
print(f"BLEU score: {bleu_result['bleu']:.4f}")


# ============================================================
# 4️⃣ BERTScore
# ============================================================

P, R, F1 = score(hyps, refs, lang="pt", rescale_with_baseline=True)

print("\n=== BERTScore ===")
print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1:        {F1.mean().item():.4f}")
