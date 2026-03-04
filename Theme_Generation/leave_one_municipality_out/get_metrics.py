import os
import json
import torch
import gc
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

# --- CONFIGURAÇÃO ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

MODEL_BASE_DIR = os.path.join(SCRIPT_DIR, "results_mbart50_loo_temas")
JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ_v2.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
LANGUAGE_CODE = "pt_XX"
MAX_INPUT_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carregar métricas
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")

def run_evaluation():
    # 1. Re-carregar os dados originais para identificar os munis
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Carregar split pré-definido
    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    test_files = set(split_info["test_files"])

    # Build a lookup dict: {municipality_name: [minutes]}
    muni_data = {m["municipality"]: m["minutes"] for m in raw_data["municipalities"]}
    municipalities = sorted(list(muni_data.keys()))
    results_summary = []

    for test_muni in municipalities:
        muni_path_name = test_muni.replace(" ", "_")
        model_path = os.path.join(MODEL_BASE_DIR, f"loo_without_{muni_path_name}", "final_model")

        if not os.path.exists(model_path):
            print(f"⚠️ Modelo para {test_muni} não encontrado em {model_path}. Ignorando...")
            continue

        print(f"🔍 Avaliando modelo LOO (Excluiu {test_muni}) nos dados de: {test_muni}")

        # Carregar Modelo e Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)

        # Filtrar apenas os dados de teste deste município (usando split pré-definido)
        test_samples = []
        prefix = "Sumariza o segmento de ata num tema conciso (máx. 15 palavras), começando com nominalização (ex.: aprovação da, criação de) e sem pontuação final. Segmento: "

        for doc in muni_data[test_muni]:
            source_file = doc.get("minute_id", "") + ".json"
            if source_file not in test_files:
                continue
            for seg in doc.get("agenda_items", []):
                if seg.get("text") and seg.get("theme"):
                    test_samples.append({"input": prefix + seg["text"], "target": seg["theme"]})

        # --- INFERÊNCIA ---
        predictions = []
        references = []

        for sample in test_samples:
            inputs = tokenizer(sample["input"], return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)

            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODE],
                    max_new_tokens=50,
                    num_beams=4
                )

            pred = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            predictions.append(pred)
            references.append(sample["target"])

        # --- CÁLCULO DAS MÉTRICAS ---
        # ROUGE
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)

        # BLEU (espera lista de listas para referências)
        bleu_results = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])

        # BERTScore
        bs_results = bertscore_metric.compute(predictions=predictions, references=references, lang="pt")

        # Organizar métricas do município
        res = {
            "municipality": test_muni,
            "rouge1": round(rouge_results["rouge1"], 4),
            "rouge2": round(rouge_results["rouge2"], 4),
            "rougeL": round(rouge_results["rougeL"], 4),
            "bleu": round(bleu_results["bleu"], 4),
            "bs_precision": round(sum(bs_results["precision"]) / len(bs_results["precision"]), 4),
            "bs_recall": round(sum(bs_results["recall"]) / len(bs_results["recall"]), 4),
            "bs_f1": round(sum(bs_results["f1"]) / len(bs_results["f1"]), 4)
        }
        results_summary.append(res)
        print(f"✅ Concluído: R1: {res['rouge1']} | B-Score F1: {res['bs_f1']}")

        # Limpeza de memória
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- SALVAR E EXIBIR ---
    df = pd.DataFrame(results_summary)
    df.to_csv(os.path.join(SCRIPT_DIR, "summaries_and_results", "loo_evaluation_results.csv"), index=False)

    # Calcular média final (Average Performance Across All Folds)
    mean_results = df.mean(numeric_only=True).to_dict()
    mean_results["municipality"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([mean_results])], ignore_index=True)

    print("\n" + "="*30)
    print("📊 RESULTADOS FINAIS LOO")
    print("="*30)
    print(df)

if __name__ == "__main__":
    run_evaluation()
