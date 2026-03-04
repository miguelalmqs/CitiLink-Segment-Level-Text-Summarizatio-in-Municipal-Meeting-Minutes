import os
import json
from tqdm import tqdm
import evaluate

# ======================================================================
# 🚀 CONFIGURAÇÃO
# ======================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_JSON = os.path.join(SCRIPT_DIR, "summaries_and_results", "generated_summaries.json")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "summaries_and_results", "evaluation_results.json")

# ======================================================================
# 📥 CARREGAR RESULTADOS
# ======================================================================

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    results = json.load(f)

# ======================================================================
# 📊 INICIALIZAÇÃO DAS MÉTRICAS
# ======================================================================

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# ======================================================================
# 🔁 AVALIAÇÃO POR MUNICÍPIO
# ======================================================================

metrics_all = {}

for muni, segments in results.items():
    print(f"\n🏛️ Avaliando município: {muni} ({len(segments)} segmentos)")

    # Filtrar predições e referências vazias
    predictions = []
    references = []
    for seg in segments:
        pred = seg["resumo_pred"].strip()
        ref = seg["resumo_ref"].strip()
        if pred == "" or ref == "":
            continue
        predictions.append(pred)
        references.append([ref])  # BLEU HuggingFace espera lista de listas

    if not predictions:
        print(f"⚠️ Nenhuma predição válida para {muni}, a saltar.")
        continue

    # -------------------- ROUGE --------------------
    rouge_res = rouge.compute(predictions=predictions, references=[r[0] for r in references])

    # -------------------- BLEU --------------------
    bleu_res = bleu.compute(predictions=predictions, references=references)

    # -------------------- BERTScore --------------------
    bert_res = bertscore.compute(predictions=predictions, references=[r[0] for r in references], lang="pt")
    bert_avg = {
        "bertscore_f1": sum(bert_res["f1"])/len(bert_res["f1"]),
        "bertscore_precision": sum(bert_res["precision"])/len(bert_res["precision"]),
        "bertscore_recall": sum(bert_res["recall"])/len(bert_res["recall"])
    }

    # -------------------- Juntar métricas --------------------
    metrics_all[muni] = {
        "rouge1": rouge_res["rouge1"],
        "rouge2": rouge_res["rouge2"],
        "rougeL": rouge_res["rougeL"],
        "bleu": bleu_res["bleu"],
        **bert_avg
    }

# ======================================================================
# 💾 SALVAR RESULTADOS
# ======================================================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics_all, f, ensure_ascii=False, indent=2)

print(f"\n✅ Avaliação concluída. Resultados guardados em {OUTPUT_JSON}")
