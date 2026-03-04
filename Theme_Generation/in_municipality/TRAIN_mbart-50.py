import os
import json
import torch
import gc
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# --- CONFIGURAÇÃO ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ_v2.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
LANGUAGE_CODE = "pt_XX"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 150
BATCH_SIZE = 4
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "results_mbart50_individual_muni")

# --- 1. CARREGAMENTO DOS DADOS ---
def load_data(json_path):
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

# --- 2. LOOP DE TREINO POR MUNICÍPIO ---
full_dataset = load_data(JSON_PATH)
municipalities = sorted(list(set(full_dataset["municipality"])))

# Carregar split pré-definido
with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
    split_info = json.load(f)
train_files = set(split_info["train_files"])
val_files = set(split_info["val_files"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = LANGUAGE_CODE
tokenizer.tgt_lang = LANGUAGE_CODE

def tokenize_fn(examples):
    model_inputs = tokenizer(examples["text"], max_length=MAX_INPUT_LENGTH, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tema"], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

for target_muni in municipalities:
    print(f"\n" + "="*60)
    print(f"🚀 TREINANDO MODELO EXCLUSIVO: {target_muni}")
    print("="*60)

    muni_path = target_muni.replace(" ", "_")
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"train_test_{muni_path}")

    # Filtrar apenas dados deste município
    muni_ds = full_dataset.filter(lambda x: x["municipality"] == target_muni)

    # Usar split pré-definido para treino e validação
    train_ds = muni_ds.filter(lambda x: x["source_file"] in train_files)
    eval_ds = muni_ds.filter(lambda x: x["source_file"] in val_files)

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_test = eval_ds.map(tokenize_fn, batched=True)

    # Carregar Modelo (Reset para o base em cada iteração)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id[LANGUAGE_CODE]

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        num_train_epochs=5, # Aumentado ligeiramente pois o dataset por câmara é menor
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none",
        weight_decay=0.01,
        save_total_limit=1
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()

    # Salvar modelo final específico desta câmara
    final_model_save = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_save)
    tokenizer.save_pretrained(final_model_save)

    # Limpeza de memória RAM e VRAM
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

print("\n✅ Treino dos 6 modelos municipais concluído!")
