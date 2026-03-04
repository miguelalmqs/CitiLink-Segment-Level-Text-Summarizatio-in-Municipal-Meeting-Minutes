import os
import json
import torch
import pandas as pd
from typing import Set

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ======================================================================
# 🚀 CONFIGURAÇÃO
# ======================================================================

CHECKPOINT = "facebook/mbart-large-50"
LANGUAGE_CODE = "pt_XX"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ_v2.json")
SPLIT_PATH = os.path.join(ROOT_DIR, "split_info.json")

OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "in_muni_only")

CHUNK_MAX_LENGTH = 1024
CHUNK_STRIDE = 512
TARGET_MAX_LENGTH = 128

NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================
# 📂 SPLIT
# ======================================================================

def load_train_files(split_path: str) -> Set[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return set(split["train_files"])

# ======================================================================
# 💾 DATASET (FILTRADO PELO SPLIT)
# ======================================================================

def load_segments_from_json(path: str, train_files: Set[str]) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for muni_obj in data["municipalities"]:
        municipality = muni_obj["municipality"]
        for doc in muni_obj["minutes"]:
            source_file = doc.get("minute_id", "") + ".json"

            # 🔒 USAR APENAS DOCUMENTOS DE TREINO
            if source_file not in train_files:
                continue

            for seg in doc.get("agenda_items", []):
                if seg.get("text") and seg.get("summary"):
                    rows.append({
                        "municipio": municipality,
                        "source_file": source_file,
                        "text": seg["text"],
                        "resumo": seg["summary"],
                    })

    return pd.DataFrame(rows)

# ======================================================================
# ✂️ CHUNKING
# ======================================================================

def chunk_text(text: str, tokenizer):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_MAX_LENGTH, len(tokens))
        chunks.append(
            tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        )
        if end == len(tokens):
            break
        start += CHUNK_STRIDE
    return chunks

# ======================================================================
# 📦 DATASET
# ======================================================================

def prepare_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    records = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["text"], tokenizer)
        records.append({
            "texto": " ".join(chunks),
            "sumario": row["resumo"],
        })
    return Dataset.from_pandas(pd.DataFrame(records))

def preprocess_function(examples, tokenizer):
    tokenizer.src_lang = LANGUAGE_CODE
    tokenizer.tgt_lang = LANGUAGE_CODE

    inputs = tokenizer(
        examples["texto"],
        max_length=CHUNK_MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["sumario"],
            max_length=TARGET_MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

    inputs["labels"] = labels["input_ids"]
    return inputs

# ======================================================================
# 🔁 IN-MUNI-ONLY TRAINING
# ======================================================================

def train_in_muni_only(df: pd.DataFrame):

    municipios = sorted(df["municipio"].unique())
    print(f"🏛️ Municípios encontrados: {municipios}")

    for muni in municipios:
        print("\n" + "=" * 90)
        print(f"🏫 Treino APENAS com município: {muni}")
        print("=" * 90)

        train_df = df[df["municipio"] == muni].reset_index(drop=True)

        if train_df.empty:
            print("⚠️ Sem dados de treino, a saltar.")
            continue

        print(f"📊 Segmentos de treino: {len(train_df)}")

        output_dir = os.path.join(
            OUTPUT_BASE_DIR,
            muni.replace(" ", "_")
        )
        os.makedirs(output_dir, exist_ok=True)

        # ----------------------------
        # Modelo & Tokenizer
        # ----------------------------
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(DEVICE)

        tokenizer.src_lang = LANGUAGE_CODE
        tokenizer.tgt_lang = LANGUAGE_CODE
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[LANGUAGE_CODE]

        # ----------------------------
        # Dataset
        # ----------------------------
        train_dataset = prepare_dataset(train_df, tokenizer)
        tokenized_train = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=["texto", "sumario"],
        )

        # ----------------------------
        # Training args
        # ----------------------------
        total_steps = (
            len(tokenized_train) // BATCH_SIZE
        ) * NUM_EPOCHS

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            logging_steps=LOGGING_STEPS,
            save_total_limit=1,
            warmup_steps=int(WARMUP_RATIO * total_steps),
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )

        # ----------------------------
        # Train
        # ----------------------------
        print("🚀 A iniciar treino...")
        trainer.train()

        # ----------------------------
        # Save
        # ----------------------------
        final_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        print(f"✅ Modelo guardado em: {final_path}")

# ======================================================================
# ▶️ MAIN
# ======================================================================

if __name__ == "__main__":

    print("📥 A carregar split...")
    train_files = load_train_files(SPLIT_PATH)

    print("📥 A carregar dataset (APENAS TRAIN)...")
    df = load_segments_from_json(JSON_PATH, train_files)

    if df.empty:
        raise RuntimeError("❌ Dataset de treino vazio após aplicar split.")

    train_in_muni_only(df)
