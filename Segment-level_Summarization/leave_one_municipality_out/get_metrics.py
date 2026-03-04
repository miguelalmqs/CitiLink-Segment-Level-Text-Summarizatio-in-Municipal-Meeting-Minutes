import os
import json
import numpy as np
import evaluate
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_JSON = os.path.join(SCRIPT_DIR, "summaries_and-results", "generated_summaries.json")
BERT_MODEL = "bert-base-multilingual-cased"

def run_detailed_evaluation():
    # 1. Carregar dados
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    municipios = df['municipio'].unique()

    # 2. Carregar métricas
    print("🚀 A carregar motores de avaliação...")
    metrics = {
        "rouge": evaluate.load("rouge"),
        "meteor": evaluate.load("meteor"),
        "sacrebleu": evaluate.load("sacrebleu"),
        "bertscore": evaluate.load("bertscore")
    }

    all_muni_results = []

    # 3. Calcular métricas para cada município
    for muni in municipios:
        print(f"📊 A avaliar: {muni}")
        muni_df = df[df['municipio'] == muni]

        preds = muni_df['preds'].tolist()
        refs = muni_df['refs'].tolist()

        # Cálculo individual
        res = {}
        res['municipio'] = muni
        res['n_segmentos'] = len(muni_df)

        # ROUGE
        rouge = metrics['rouge'].compute(predictions=preds, references=refs)
        res.update({k.upper(): round(v, 4) for k, v in rouge.items()})

        # METEOR
        res['METEOR'] = round(metrics['meteor'].compute(predictions=preds, references=refs)['meteor'], 4)

        # BLEU
        res['BLEU'] = round(metrics['sacrebleu'].compute(predictions=preds, references=[[r] for r in refs])['score'], 4)

        # BERTScore
        bs = metrics['bertscore'].compute(predictions=preds, references=refs, lang="pt", model_type=BERT_MODEL)
        res['BS_P'] = round(np.mean(bs['precision']), 4)
        res['BS_R'] = round(np.mean(bs['recall']), 4)
        res['BS_F1'] = round(np.mean(bs['f1']), 4)

        all_muni_results.append(res)

    # 4. Criar Tabela Comparativa
    final_df = pd.DataFrame(all_muni_results)

    print("\n" + "="*80)
    print(f"{'MUNICÍPIO':<15} | {'ROUGE-L':<10} | {'BLEU':<10} | {'BS-F1':<10} | {'SEGS':<5}")
    print("-"*80)
    for _, row in final_df.iterrows():
        print(f"{row['municipio']:<15} | {row['ROUGEL']:<10} | {row['BLEU']:<10} | {row['BS_F1']:<10} | {row['n_segmentos']:<5}")
    print("="*80)

    # Guardar em CSV para usares na tese/relatório
    output_csv = os.path.join(SCRIPT_DIR, "summaries_and-results", "evaluation_results.csv")
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ Relatório detalhado guardado em: {output_csv}")

if __name__ == "__main__":
    run_detailed_evaluation()
