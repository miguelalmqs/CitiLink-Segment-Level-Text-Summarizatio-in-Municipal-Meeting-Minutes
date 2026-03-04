import os
import json
import torch
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# --- CAMINHOS RELATIVOS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..", "..")

# --- CONFIGURAÇÃO ---
# O caminho fornecido onde o modelo e o tokenizador treinado estão guardados
MODEL_DIR = os.path.join(SCRIPT_DIR, "results_mbart50_ata_summarization", "final_model")
JSON_PATH = os.path.join(ROOT_DIR, "dataset", "citilink_summ.json")
SPLIT_JSON_PATH = os.path.join(ROOT_DIR, "split_info.json")
LANGUAGE_CODE = "pt_XX"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 150
BATCH_SIZE = 8
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_mbart50_ata_summarization")


# --- FUNÇÕES AUXILIARES DE CARREGAMENTO E PROCESSAMENTO ---

def load_and_prepare_data(json_path, split_json_path):
    """Carrega o JSON e extrai as amostras de teste usando o split pré-definido."""

    # 1. Carregar split
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    test_files = set(split_info["test_files"])

    # 2. Carregar dados
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {json_path}.")
        return None
    except json.JSONDecodeError:
        print(f"ERRO: Não foi possível decodificar o JSON em {json_path}.")
        return None

    test_list = []

    instruction_prefix = (
        "Sumariza o segmento de ata num tema conciso (máx. 15 palavras), começando com nominalização "
        "(ex.: aprovação da, criação de) e sem pontuação final. Segmento: "
    )

    for muni_obj in raw_data["municipalities"]:
        for doc in muni_obj["minutes"]:
            source_file = doc.get("minute_id", "") + ".json"
            if source_file not in test_files:
                continue
            for segment in doc.get("agenda_items", []):

                # Normalização e limpeza de texto (igual à usada no treino)
                text = segment.get("text", "")

                source_text = instruction_prefix + text
                target_text = segment.get("theme", "").strip()

                if target_text and source_text:
                    test_list.append({
                        "text": source_text,
                        "tema": target_text
                    })

    if not test_list:
        print("AVISO: Nenhuma amostra de teste encontrada no JSON.")
        return None

    return DatasetDict({
        'test': Dataset.from_list(test_list)
    })


# --- FUNÇÃO DE MÉTRICAS (ROUGE, METEOR, BLEU, BERTScore) ---

# Carregar objetos de métricas (requerem a instalação de 'evaluate')
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")


def compute_metrics(eval_preds):
    """
    Função para calcular ROUGE, METEOR, BLEU e BERTScore durante a avaliação.

    CORREÇÃO: Adiciona-se a descodificação dos tokens para resolver o NameError.
    """
    predictions, labels = eval_preds

    # --- DESCODIFICAÇÃO E PRÉ-PROCESSAMENTO (CORREÇÃO DO NameError) ---

    # 1. Substituir todos os -100 (padding/máscara) pelo ID de padding real do tokenizer
    global tokenizer
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 2. Descodificar as previsões (tokens para texto)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 3. Descodificar os rótulos (tokens para texto)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 4. Pré-processamento: Formatar decoded_labels como lista de listas (necessário para SacreBLEU)
    decoded_labels = [[label.strip()] for label in decoded_labels]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    # --- FIM DA CORREÇÃO ---

    # --- CÁLCULO DAS MÉTRICAS ---

    # ROUGE
    rouge_results = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    # METEOR (Espera lista de strings para references)
    meteor_result = meteor_metric.compute(
        predictions=decoded_preds,
        references=[ref[0] for ref in decoded_labels]
    )

    # BLEU (SacreBLEU) (Espera lista de listas para references)
    bleu_result = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    # BERTScore
    # Nota: BERTScore é mais lento.
    bert_results = bertscore_metric.compute(
        predictions=decoded_preds,
        references=[ref[0] for ref in decoded_labels], # Passamos o primeiro ref
        lang="pt",
    )
    bert_f1 = np.mean(bert_results['f1'])

    # --- AGRUPAR OS RESULTADOS ---
    final_metrics = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "meteor": meteor_result["meteor"],
        "sacrebleu": bleu_result["score"],
        "bertscore_f1": bert_f1
    }

    # Arredondar e formatar (ROUGE/METEOR/BERTScore * 100, BLEU direto)
    final_metrics = {k: round(v * 100, 4) if k not in ["sacrebleu"] else round(v, 4) for k, v in final_metrics.items()}

    return final_metrics


# --- EXECUÇÃO DE CARREGAMENTO E AVALIAÇÃO ---

print("-" * 80)
print(f"1. A carregar Tokenizador e Modelo de: {MODEL_DIR}...")
print("-" * 80)

# 1. Carregar o Tokenizador e o Modelo Treinado
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    # Definir a língua de origem e destino para o tokenizador (mBART-CRÍTICO)
    tokenizer.src_lang = LANGUAGE_CODE
    tokenizer.tgt_lang = LANGUAGE_CODE
except Exception as e:
    print(f"ERRO ao carregar o modelo ou tokenizador do diretório: {e}")
    exit()

# 2. Carregar e Pré-processar o Dataset de Teste
print("\n2. A carregar e tokenizar o dataset de teste (split pré-definido)...")

datasets = load_and_prepare_data(JSON_PATH, SPLIT_JSON_PATH)

if datasets is None:
    print("O processamento não pode continuar devido a um erro de carregamento de dados.")
    exit()

def tokenize_function_load(examples):
    """Função para tokenizar o Input e o Target (usa o tokenizador já carregado)."""
    tokenizer.src_lang = LANGUAGE_CODE
    tokenizer.tgt_lang = LANGUAGE_CODE

    model_inputs = tokenizer(examples["text"], max_length=MAX_INPUT_LENGTH, truncation=True)
    # O uso de 'as_target_tokenizer' está obsoleto mas funciona com transformers < 4.41.
    # No entanto, para modelos Seq2Seq, o 'tokenizer' é usado implicitamente como target.
    # Vamos manter o padrão para maior compatibilidade.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tema"], max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Aplicar a tokenização APENAS ao dataset de teste
tokenized_test_dataset = datasets['test'].map(tokenize_function_load, batched=True)

print(f"Total de amostras de Teste para avaliação: {len(tokenized_test_dataset)}")


# 3. Inicializar o Trainer (Apenas para avaliação)
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True, # Usar geração durante a avaliação
    generation_max_length=MAX_TARGET_LENGTH,
    report_to="none",
    # Opcional: Para gerar previsões ligeiramente melhores, podemos aumentar o beam size
    # generation_num_beams=6,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=None, # Não é necessário para avaliação
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics, # A função corrigida será usada aqui
)


# 4. Executar a Avaliação (para obter os scores)
print("-" * 80)
print("3. INICIANDO A AVALIAÇÃO COMPLETA NO DATASET DE TESTE...")
print("-" * 80)

# Se estiver a usar CUDA, o modelo deve ser transferido para lá
if torch.cuda.is_available():
    trainer.model.to("cuda")

# Esta chamada de avaliação obtém os scores que viu
evaluation_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)


# --- 5. RE-EXECUTAR GERAÇÃO E GUARDAR RESULTADOS EM JSON ---
print("-" * 80)
print("4. GERANDO PREVISÕES FINAIS E GUARDANDO JSON DE RESULTADOS...")
print("-" * 80)

# Usamos o método .predict para obter as previsões e os labels para o JSON
results = trainer.predict(
    test_dataset=tokenized_test_dataset,
    metric_key_prefix="predict"
)

# Os resultados de .predict contêm os IDs de token (predictions e labels)
predictions_ids = results.predictions
label_ids = results.label_ids

# 1. Pré-processamento dos labels
# Substituir todos os -100 (padding) pelo ID de padding real
label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

# 2. Descodificar as previsões e os rótulos (Temas)
temas_gerados = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
temas_anotados = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

# 3. Recuperar o texto original (Input)
# O dataset tokenizado ainda contém a coluna 'text' (texto original com prefixo de instrução)
textos_originais_com_prefixo = tokenized_test_dataset['text']
# Define o prefixo exato para o remover do texto original
instruction_prefix = "Sumariza o segmento de ata num tema conciso (máx. 15 palavras), começando com nominalização (ex.: aprovação da, criação de) e sem pontuação final. Segmento: "
prefix_len = len(instruction_prefix)
textos_originais = [t[prefix_len:].strip() for t in textos_originais_com_prefixo]


# 4. Combinar os resultados
results_list = []
for original, annotated, generated in zip(textos_originais, temas_anotados, temas_gerados):
    results_list.append({
        "texto_original": original.strip(),
        "tema_anotado": annotated.strip(),
        "tema_gerado": generated.strip()
    })

OUTPUT_JSON_FILE = os.path.join(SCRIPT_DIR, "summaries_and_results", "mbart_evaluation_results.json")
try:
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Resultados salvos com sucesso em: {OUTPUT_JSON_FILE}")
except Exception as e:
    print(f"\n❌ ERRO ao salvar o JSON: {e}")


# --- IMPRIMIR RESULTADOS FINAIS ---
print("\n✨ Resultados (Scores em %) ✨")
print("-----------------------------------------------------")
for key, value in evaluation_results.items():
    if key.startswith("eval_"):
        metric_name = key.replace("eval_", "")
        print(f"| {metric_name.upper():<15} | {value:>15.4f} |")
print("-----------------------------------------------------")
