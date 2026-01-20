#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Completo de Scoring
=============================
Este script faz:
1. Pré-processa os novos dados (mesma transformação do treino)
2. Carrega o modelo do MLflow
3. Faz as predições
4. Salva os resultados

USO:
    python pipeline_scoring.py --input ../data/base_clientes_inadimplencia_2.csv
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import mlflow

# Adiciona o diretório src ao path para importar os módulos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Importa funções de pré-processamento
from pre_processamento import (
    tratar_valores_nulos,
    tratar_data_nascimento,
    converter_colunas_data,
    calcular_tempo_assinatura,
    calcular_tempo_atraso_fatura,
    codificar_variaveis_categoricas,
    escalar_variaveis
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("pipeline-scoring")

# Configurações
MLFLOW_TRACKING_URI = os.path.join(PROJECT_DIR, "mlruns")


def preprocessar_para_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o mesmo pré-processamento usado no treino.
    
    IMPORTANTE: Em produção, você salvaria o scaler/encoder do treino
    e reutilizaria aqui para garantir consistência.
    """
    log.info("Iniciando pré-processamento para scoring...")
    
    # Colunas de data e colunas para remover
    colunas_data = ["Data_Contratacao", "Data_Vencimento_Fatura", "Data_Ingestao", "Data_Atualizacao"]
    drop_cols = ["Telefone", "Nome", "Email", "Data_Nascimento", 
                 "Data_Contratacao", "Data_Vencimento_Fatura", 
                 "Data_Ingestao", "Data_Atualizacao"]
    
    # Guarda o ID antes de processar
    id_cliente = df["ID_Cliente"].copy()
    
    # Pipeline de pré-processamento
    df = tratar_valores_nulos(df)
    df = tratar_data_nascimento(df)
    df = df.set_index('ID_Cliente', drop=True)
    df = converter_colunas_data(df, colunas_data)
    df = calcular_tempo_assinatura(df)
    df = calcular_tempo_atraso_fatura(df)
    
    # Remove colunas que não serão usadas
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    
    # Codifica categóricas
    df = codificar_variaveis_categoricas(df)
    
    # Remove target se existir (dados de scoring não têm target)
    if "Status_Pagamento" in df.columns:
        df = df.drop(columns=["Status_Pagamento"])
    
    # Escala variáveis numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = escalar_variaveis(df, numeric_cols)
    
    log.info(f"Pré-processamento concluído. Shape: {df.shape}")
    return df


def carregar_modelo(model_name: str, model_version: str):
    """Carrega modelo do MLflow Registry local."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    uri = f"models:/{model_name}/{model_version}"
    log.info(f"Carregando modelo: {uri}")
    
    model = mlflow.pyfunc.load_model(uri)
    return model


def fazer_predicoes(model, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o modelo e retorna predições com probabilidades."""
    log.info("Gerando predições...")
    
    # Predições
    predictions = model.predict(df)
    
    # Monta DataFrame de resultado
    resultado = pd.DataFrame(index=df.index)
    resultado["prediction"] = predictions
    
    # Tenta obter probabilidades
    try:
        # Para modelos sklearn
        unwrapped_model = model._model_impl.python_model
        if hasattr(unwrapped_model, "predict_proba"):
            probas = unwrapped_model.predict_proba(df)
            if probas.shape[1] == 2:
                resultado["prob_adimplente"] = probas[:, 0]
                resultado["prob_inadimplente"] = probas[:, 1]
    except Exception as e:
        log.warning(f"Não foi possível obter probabilidades: {e}")
    
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de scoring")
    parser.add_argument("--input", type=str, required=True, help="Caminho do CSV de entrada")
    parser.add_argument("--model-name", type=str, default="ModelRFC-GCP", help="Nome do modelo")
    parser.add_argument("--model-version", type=str, default="1", help="Versão do modelo")
    parser.add_argument("--output", type=str, default=None, help="Caminho do CSV de saída")
    args = parser.parse_args()
    
    # 1. Carrega dados brutos
    log.info(f"Carregando dados de: {args.input}")
    df_raw = pd.read_csv(args.input)
    log.info(f"Dados carregados: {df_raw.shape}")
    
    # Guarda IDs originais
    ids_originais = df_raw["ID_Cliente"].copy()
    
    # 2. Pré-processa
    df_processed = preprocessar_para_scoring(df_raw)
    
    # 3. Carrega modelo
    model = carregar_modelo(args.model_name, args.model_version)
    
    # 4. Faz predições
    resultado = fazer_predicoes(model, df_processed)
    
    # 5. Adiciona ID de volta
    resultado = resultado.reset_index()
    
    # 6. Salva resultado
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PROJECT_DIR, f"predicoes_{ts}.csv")
    
    resultado.to_csv(output_path, index=False)
    log.info(f"Predições salvas em: {output_path}")
    
    # 7. Mostra resumo
    print("\n" + "="*50)
    print("RESUMO DAS PREDIÇÕES")
    print("="*50)
    print(f"Total de clientes: {len(resultado)}")
    print(f"Preditos como Adimplentes (0): {(resultado['prediction'] == 0).sum()}")
    print(f"Preditos como Inadimplentes (1): {(resultado['prediction'] == 1).sum()}")
    if "prob_inadimplente" in resultado.columns:
        print(f"\nProbabilidade média de inadimplência: {resultado['prob_inadimplente'].mean():.2%}")
    print("="*50)
    
    return resultado


if __name__ == "__main__":
    main()
