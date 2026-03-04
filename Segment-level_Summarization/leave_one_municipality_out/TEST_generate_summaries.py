import os
import json
import torch
import numpy as np
import pandas as pd
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ======================================================================
# 🚀 CONFIGURATION
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

LOO_MODELS_BASE_DIR = os.path.join(SCRIPT_DIR, "loo_models_mbart")
JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ_v2.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "summaries_and-results", "generated_summaries.json")
LANGUAGE_CODE = "pt_XX"
BERT_MODEL = "neuralmind/bert-base-portuguese-cased"

# ----------------------------------------------------------------------
# 🛠️ DATA & GENERATION UTILS
# ----------------------------------------------------------------------

def load_all_data(json_path, split_json_path):
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    test_files = set(split_info["test_files"])

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments_by_muni = {}
    for muni_obj in data["municipalities"]:
        municipality = muni_obj["municipality"]
        muni_segments = []
        for doc in muni_obj["minutes"]:
            source_file = doc.get("minute_id", "") + ".json"
            if source_file not in test_files:
                continue
            for seg in doc.get("agenda_items", []):
                if seg.get("text") and seg.get("summary"):
                    muni_segments.append({
                        "municipio": municipality,
                        "source_file": source_file,
                        "text": seg["text"],
                        "tema": seg.get("theme", "Assunto Geral"),
                        "referencia": seg["summary"]
                    })
        if muni_segments:
            segments_by_muni[municipality] = muni_segments
    return segments_by_muni

def generate_recursive_summary(text, tema, model, tokenizer):
    inputs_tokens = tokenizer.encode(text, add_special_tokens=False)
    chunk_size = 700
    num_shards = (len(inputs_tokens) // chunk_size) + 1
    shards_summaries = []

    for i in range(num_shards):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_text = tokenizer.decode(inputs_tokens[start:end], skip_special_tokens=True)

        prompt = f"RESUMIR [TEMA: {tema}] [PARTE: {i+1}/{num_shards}] [TEXTO]: {chunk_text}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800).to(model.device)

        summary_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=1024,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODE]
        )
        output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        if output_text.strip():
            shards_summaries.append(output_text.strip())
    return " ".join(shards_summaries)

# ======================================================================
# ▶️ MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_by_muni = load_all_data(JSON_PATH, SPLIT_JSON_PATH)
    municipios = list(data_by_muni.keys())

    resultados_finais = []

    # --- STEP 1: GENERATION ---
    print(f"🤖 Starting LOO Inference for {len(municipios)} municipalities...")
    for held_out in municipios:
        muni_folder = held_out.replace(' ', '_')
        model_path = os.path.join(LOO_MODELS_BASE_DIR, f"train_without_{muni_folder}", "final_model")

        if not os.path.exists(model_path):
            print(f"⚠️ Model for {held_out} not found at {model_path}. Skipping.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer.src_lang, tokenizer.tgt_lang = LANGUAGE_CODE, LANGUAGE_CODE

        for item in tqdm(data_by_muni[held_out], desc=f"Muni: {held_out}"):
            gen_sum = generate_recursive_summary(item['text'], item['tema'], model, tokenizer)
            resultados_finais.append({
                "municipio": held_out,
                "preds": gen_sum,
                "refs": item['referencia']
            })

        del model
        torch.cuda.empty_cache()

    # Save to disk just in case
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, ensure_ascii=False, indent=4)

    # --- STEP 2: EVALUATION ---
    if not resultados_finais:
        print("❌ No summaries were generated. Check your model paths.")
    else:
        print("\n📊 Calculating Metrics...")
        preds = [r['preds'] for r in resultados_finais]
        refs = [r['refs'] for r in resultados_finais]

        metrics = {
            "rouge": evaluate.load("rouge"),
            "meteor": evaluate.load("meteor"),
            "sacrebleu": evaluate.load("sacrebleu"),
            "bertscore": evaluate.load("bertscore")
        }

        results = {}
        # Rouge
        r = metrics['rouge'].compute(predictions=preds, references=refs)
        results.update(r)
        # Meteor
        results['meteor'] = metrics['meteor'].compute(predictions=preds, references=refs)['meteor']
        # Sacrebleu
        results['bleu'] = metrics['sacrebleu'].compute(predictions=preds, references=[[r] for r in refs])['score']
        # BertScore
        bs = metrics['bertscore'].compute(predictions=preds, references=refs, lang="pt", model_type=BERT_MODEL)
        results['bs_p'], results['bs_r'], results['bs_f1'] = np.mean(bs['precision']), np.mean(bs['recall']), np.mean(bs['f1'])

        print("\n" + "="*40)
        print("          FINAL LOO RESULTS")
        print("="*40)
        for k, v in results.items():
            print(f"{k.upper():<15}: {v:.4f}")
