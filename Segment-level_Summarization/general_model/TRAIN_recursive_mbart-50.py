import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ======================================================================
# ⚙️ CONFIGURAÇÃO DE LIMITES
# ======================================================================
CHECKPOINT = "facebook/mbart-large-50"
LANGUAGE_CODE = "pt_XX"

MAX_INPUT_LENGTH = 600
MAX_TARGET_LENGTH = 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ_v2.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_mbart50_recursive_v1")

# ======================================================================
# 🛠️ PROCESSAMENTO RECURSIVO
# ======================================================================

def load_split_file(split_json_path):
    with open(split_json_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_files"])

def load_segments_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_segments = []
    for muni_obj in data["municipalities"]:
        municipality = muni_obj["municipality"]
        for doc in muni_obj["minutes"]:
            source_file = doc.get("minute_id", "") + ".json"
            for seg in doc.get("agenda_items", []):
                if seg.get("text") and seg.get("summary"):
                    all_segments.append({
                        "source_file": source_file,
                        "text": seg["text"],
                        "tema": seg.get("theme", "Assunto Geral"),
                        "resumo": seg["summary"]
                    })
    return pd.DataFrame(all_segments)

def recursive_chunking_function(examples):
    # Definir as línguas explicitamente ANTES de qualquer encode
    tokenizer.src_lang = LANGUAGE_CODE
    tokenizer.tgt_lang = LANGUAGE_CODE

    inputs = []
    targets = []

    for text, tema, resumo in zip(examples["texto"], examples["tema"], examples["sumario"]):
        # Encode inicial sem truncagem
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        summary_tokens = tokenizer.encode(str(resumo), add_special_tokens=False)

        num_shards = max(
            (len(text_tokens) // (MAX_INPUT_LENGTH - 60)) + 1,
            (len(summary_tokens) // (MAX_TARGET_LENGTH - 20)) + 1
        )

        t_size = len(text_tokens) // num_shards
        s_size = len(summary_tokens) // num_shards

        for i in range(num_shards):
            t_chunk = text_tokens[i * t_size : (i + 1) * t_size]
            s_chunk = summary_tokens[i * s_size : (i + 1) * s_size]

            chunk_text = tokenizer.decode(t_chunk, skip_special_tokens=True)
            chunk_summary = tokenizer.decode(s_chunk, skip_special_tokens=True)

            prompt = f"RESUMIR [TEMA: {tema}] [PARTE: {i+1}/{num_shards}] [TEXTO]: {chunk_text}"

            inputs.append(prompt)
            targets.append(chunk_summary)

    # Nova forma de tokenizar (sem as_target_tokenizer)
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Tokeniza os labels usando text_target
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ======================================================================
# 🚀 TREINO
# ======================================================================

if __name__ == "__main__":
    df = load_segments_from_json(JSON_PATH)
    train_files = load_split_file(SPLIT_JSON_PATH)
    train_df = df[df["source_file"].isin(train_files)].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

    # Configuração vital para mBART
    tokenizer.src_lang = LANGUAGE_CODE
    tokenizer.tgt_lang = LANGUAGE_CODE
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id[LANGUAGE_CODE]

    train_ds = Dataset.from_pandas(train_df[['text', 'resumo', 'tema']].rename(
        columns={'text': 'texto', 'resumo': 'sumario'}
    ))

    print("--- Mapeando dataset com Correção de Tokenizer ---")
    tokenized_train = train_ds.map(
        recursive_chunking_function,
        batched=True,
        remove_columns=train_ds.column_names,
        batch_size=4
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    print(f"--- Iniciando Treino (Shards: {len(tokenized_train)}) ---")
    trainer.train()

    save_path = os.path.join(OUTPUT_DIR, "citilink_recursive_final")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Sucesso: {save_path}")
