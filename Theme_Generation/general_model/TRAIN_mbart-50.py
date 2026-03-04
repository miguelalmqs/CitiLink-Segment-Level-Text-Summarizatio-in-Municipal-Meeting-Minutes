import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# --- CAMINHOS RELATIVOS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

# --- CONFIGURAÇÃO ---
JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
# MUDANÇA CRÍTICA 1: Modelo mBART-50 da Facebook
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
# Código de Língua para Português (pt_XX)
LANGUAGE_CODE = "pt_XX"
MAX_INPUT_LENGTH = 1024
# O tema é muito curto, mas este tamanho assegura que a instrução e o tema cabem
MAX_TARGET_LENGTH = 150
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 8

# --- FUNÇÕES AUXILIARES ---

def load_and_prepare_data(json_path, split_json_path):
    """Carrega o JSON e transforma-o num formato Dataset do Hugging Face, usando o split pré-definido."""

    # 1. Carregar split
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    train_files = set(split_info["train_files"])
    val_files = set(split_info["val_files"])

    # 2. Carregar dados
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {json_path}. Certifique-se de que 'citilink_summ.json' está presente.")
        return None
    except json.JSONDecodeError:
        print(f"ERRO: Não foi possível decodificar o JSON em {json_path}. Verifique a sintaxe.")
        return None

    train_list = []
    val_list = []

    # 3. Iterar sobre o JSON para extrair pares Input/Target
    instruction_prefix = (
        "Sumariza o segmento de ata num tema conciso (máx. 15 palavras), começando com nominalização "
        "(ex.: aprovação da, criação de) e sem pontuação final. Segmento: "
    )

    for muni_obj in raw_data["municipalities"]:
        for doc in muni_obj["minutes"]:
            source_file = doc.get("minute_id", "") + ".json"
            for segment in doc.get("agenda_items", []):

                # Normalização e limpeza de texto (sanitização, adaptada do seu script original)
                text = segment.get("text", "")
                text = text.replace("Sr. Presidente", "um cidadão")
                text = text.replace("Vereador José Andrezo", "um cidadão")
                text = text.replace("Senhor Presidente", "um cidadão")
                text = text.replace("Município de Alandroal", "Município")

                # O Input (source) é o segmento completo + instrução
                source_text = instruction_prefix + text
                target_text = segment.get("theme", "").strip()

                if target_text and source_text:
                    sample = {"text": source_text, "tema": target_text}
                    if source_file in train_files:
                        train_list.append(sample)
                    elif source_file in val_files:
                        val_list.append(sample)

    if not train_list:
        print("AVISO: Nenhuma amostra de treino válida encontrada no JSON.")
        return None

    # 4. Converter para Hugging Face DatasetDict (usando o split pré-definido)
    return DatasetDict({
        'train': Dataset.from_list(train_list),
        'test': Dataset.from_list(val_list) if val_list else Dataset.from_list(train_list[:1])
    })


def tokenize_function(examples):
    """Função para tokenizar os dados de Input (text) e Target (tema) para mBART."""

    # CRÍTICO PARA MBART: Definir a língua de origem e destino
    # Isso é necessário para que o mBART adicione os tokens de idioma corretos.
    tokenizer.src_lang = LANGUAGE_CODE
    tokenizer.tgt_lang = LANGUAGE_CODE

    # Tokenização do Input
    model_inputs = tokenizer(examples["text"], max_length=MAX_INPUT_LENGTH, truncation=True)

    # Tokenização dos Labels (Targets)
    # mBART usa 'labels' para o target, assim como T5
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tema"], max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- INICIALIZAÇÃO ---

print(f"A carregar o tokenizador e o modelo: {MODEL_NAME}...")
# AutoTokenizer & AutoModelForSeq2SeqLM: Classes genéricas para Encoder-Decoder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --- CARREGAMENTO E PROCESSAMENTO DE DADOS ---
print("A carregar e pré-processar o dataset...")

datasets = load_and_prepare_data(JSON_PATH, SPLIT_JSON_PATH)

if datasets is None:
    print("O processamento não pode continuar devido a um erro de carregamento de dados.")
    exit()

# Aplicar a tokenização aos datasets
tokenized_datasets = datasets.map(tokenize_function, batched=True)

print(f"Total de amostras de Treino: {len(tokenized_datasets['train'])}")
print(f"Total de amostras de Teste: {len(tokenized_datasets['test'])}")


# --- DEFINIÇÃO DOS ARGUMENTOS DE TREINAMENTO ---

# Diretório para guardar o modelo treinado
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_mbart50_ata_summarization")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=os.path.join(SCRIPT_DIR, 'logs'),
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    # CRÍTICO PARA MBART: Definir a língua de destino para a geração
    # O token de língua será adicionado no início de cada geração
    generation_max_length=MAX_TARGET_LENGTH,
    predict_with_generate=True, # Usar geração durante a avaliação
    report_to="none", # Desativa a integração com Weights & Biases
)

# Data Collator: junta amostras em batches (preenchendo-as com padding)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# --- INICIALIZAÇÃO E INÍCIO DO TREINAMENTO ---

# Inicializar o Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("-" * 80)
print("INICIANDO O AJUSTE FINO (FINE-TUNING) DO mBART-50...")
print("-" * 80)

# Iniciar o treinamento
trainer.train()

# --- GUARDA E INFERÊNCIA ---

# Guardar o modelo final treinado
final_model_path = f"{OUTPUT_DIR}/final_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Treinamento concluído. Modelo guardado em: {final_model_path}")

print("-" * 80)
print("EXEMPLO DE INFERÊNCIA COM O MODELO TREINADO:")

# Para testar, carrega-se o modelo guardado
trained_tokenizer = AutoTokenizer.from_pretrained(final_model_path)
trained_model = AutoModelForSeq2SeqLM.from_pretrained(final_model_path)

# CRÍTICO PARA INFERÊNCIA DO MBART: Definir a língua de destino
trained_tokenizer.tgt_lang = LANGUAGE_CODE

# Usar a primeira amostra de teste para inferência
sample_input = datasets['test'][0]
input_text = sample_input["text"]
target_text = sample_input["tema"]

# Tokenizar o input
input_ids = trained_tokenizer(
    input_text,
    return_tensors="pt",
    max_length=MAX_INPUT_LENGTH,
    truncation=True
).input_ids

# Gerar o tema (usando beam search para melhor qualidade)
output_ids = trained_model.generate(
    input_ids,
    max_length=MAX_TARGET_LENGTH,
    num_beams=5, # Usar 5 beams para maior qualidade
    early_stopping=True,
)
predicted_tema = trained_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"Texto Original (Input para o modelo): {input_text}")
print(f"Tema Alvo (Target): {target_text}")
print(f"Tema Predito pelo Modelo Treinado: {predicted_tema}")
print("-" * 80)
