import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

MODEL_BASE_DIR = os.path.join(SCRIPT_DIR, "results_mbart50_individual_muni")
JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ_v2.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
LANGUAGE_CODE = "pt_XX"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. CARREGAMENTO DAS MÉTRICAS (VIA EVALUATE) ---
# Nota: Requer 'pip install evaluate rouge_score bert_score'
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# --- 2. FUNÇÃO DE CARREGAMENTO DE DADOS ---
def load_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Ficheiro {json_path} não encontrado.")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    instruction_prefix = (
        "Sumariza o segmento de ata num tema conciso (máx. 15 palavras), começando com nominalização "
        "(ex.: aprovação da, criação de) e sem pontuação final. Segmento: "
    )

    for muni_obj in raw_data["municipalities"]:
        municipality = muni_obj["municipality"]
        for doc in muni_obj["minutes"]:
            source_file = doc.get("minute_id", "") + ".json"
            for segment in doc.get("agenda_items", []):
                text = segment.get("text", "")
                tema = segment.get("theme", "").strip()

                if text and tema:
                    samples.append({
                        "text": instruction_prefix + text,
                        "tema": tema,
                        "municipality": municipality,
                        "source_file": source_file
                    })
    return Dataset.from_list(samples)

# --- 3. EXECUÇÃO DA AVALIAÇÃO ---
full_dataset = load_data(JSON_PATH)
municipalities = sorted(list(set(full_dataset["municipality"])))

# Carregar split pré-definido
with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
    split_info = json.load(f)
test_files = set(split_info["test_files"])

all_results = []

print(f"Dispositivo detectado: {DEVICE.upper()}")

for muni in municipalities:
    muni_path = muni.replace(" ", "_")
    # Caminho para a pasta onde o script de treino guardou o modelo final
    model_path = os.path.join(MODEL_BASE_DIR, f"train_test_{muni_path}", "final_model")

    if not os.path.exists(model_path):
        print(f"⚠️ Modelo para {muni} não encontrado em {model_path}. A saltar...")
        continue

    print(f"\n🔍 Avaliando Câmara: {muni}")

    # Isolar o conjunto de TESTE usando o split pré-definido
    muni_ds = full_dataset.filter(lambda x: x["municipality"] == muni)
    test_ds = muni_ds.filter(lambda x: x["source_file"] in test_files)

    # Carregar Tokenizer e Modelo
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)

    predictions = []
    references = []

    # Geração de Inferência
    model.eval()
    for batch in tqdm(test_ds, desc=f"Gerando temas"):
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODE]
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(batch["tema"])

    # --- CÁLCULO DAS MÉTRICAS ---
    # ROUGE
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # BLEU (espera lista de strings para previsões e lista de listas para referências)
    bleu_results = bleu.compute(predictions=predictions, references=references)

    # BERTScore (específico para Português)
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="pt", device=DEVICE)

    # Compilar métricas num dicionário
    res = {
        "Câmara": muni,
        "Amostras": len(test_ds),
        "ROUGE-1": round(rouge_results["rouge1"], 4),
        "ROUGE-2": round(rouge_results["rouge2"], 4),
        "ROUGE-L": round(rouge_results["rougeL"], 4),
        "BLEU": round(bleu_results["bleu"], 4),
        "BERT_Prec": round(np.mean(bert_results["precision"]), 4),
        "BERT_Rec": round(np.mean(bert_results["recall"]), 4),
        "BERT_F1": round(np.mean(bert_results["f1"]), 4)
    }

    all_results.append(res)

    # Limpar memória GPU
    del model
    del tokenizer
    torch.cuda.empty_cache()

# --- 4. EXIBIÇÃO E EXPORTAÇÃO ---
if all_results:
    df = pd.DataFrame(all_results)
    print("\n" + "="*90)
    print("📊 RESULTADOS FINAIS POR CÂMARA (TEST SET)")
    print("="*90)
    print(df.to_string(index=False))

    # Salvar resultados
    output_file = os.path.join(SCRIPT_DIR, "summaries_and_results", "metricas_finais_municipios.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ Resultados exportados com sucesso para: {output_file}")
else:
    print("❌ Nenhum modelo foi processado. Verifica os caminhos das pastas.")
