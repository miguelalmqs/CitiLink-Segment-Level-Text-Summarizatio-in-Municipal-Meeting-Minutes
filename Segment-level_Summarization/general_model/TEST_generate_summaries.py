import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

MODEL_PATH = os.path.join(SCRIPT_DIR, "results_mbart50_recursive_v1", "citilink_recursive_final")
JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ_v2.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "summaries_and_results", "generated_summaries.json")
LANGUAGE_CODE = "pt_XX"



def load_test_files(split_json_path):
    with open(split_json_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["test_files"])

def load_test_data(json_path, test_files):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_segments = []
    for muni_obj in data["municipalities"]:
        municipality = muni_obj["municipality"]
        for doc in muni_obj["minutes"]:
            source_file = doc.get("minute_id", "") + ".json"
            if source_file in test_files:
                for seg in doc.get("agenda_items", []):
                    if seg.get("text") and seg.get("summary"):
                        test_segments.append({
                            "municipio": municipality,
                            "source_file": source_file,
                            "text": seg["text"],
                            "tema": seg.get("theme", "Assunto Geral"),
                            "referencia": seg["summary"]
                        })
    return test_segments

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
            min_new_tokens=20,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            length_penalty=1.5,
            early_stopping=False,
            forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODE]
        )

        output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        if output_text.strip() and output_text not in shards_summaries:
            shards_summaries.append(output_text.strip())

    return " ".join(shards_summaries)


if __name__ == "__main__":
    test_files = load_test_files(SPLIT_JSON_PATH)
    test_data = load_test_data(JSON_PATH, test_files)

    print(f"📦 Documentos para processar: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


    tokenizer.src_lang = LANGUAGE_CODE
    tokenizer.tgt_lang = LANGUAGE_CODE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    resultados_finais = []

    print("🤖 A gerar resumos (Versão Anti-Corte)...")
    for item in tqdm(test_data):
        resumo_gerado = generate_recursive_summary(item['text'], item['tema'], model, tokenizer)

        resultados_finais.append({
            "municipio": item['municipio'],
            "ficheiro_origem": item['source_file'],
            "tema": item['tema'],
            "resumo_referencia": item['referencia'],
            "resumo_modelo": resumo_gerado
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Ficheiro guardado com sucesso: {OUTPUT_JSON}")
