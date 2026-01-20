#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring em Batch - Google Cloud Platform
==========================================
Este script realiza predições em lote usando um modelo treinado.

Fluxo MLOps:
1. Carrega modelo do MLflow (registry)
2. Lê dados do GCS
3. Faz predições
4. Salva resultados no GCS

Este é um padrão comum em produção para processar grandes volumes de dados.
"""

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import mlflow
from mlflow.exceptions import RestException

from google.cloud import storage
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("batch-scoring")

# ==================== CONFIGURAÇÕES GCP ====================
PROJECT_ID = "mlops-484912"
REGION = "us-central1"
BUCKET_NAME = "meu-bucket-29061999"

# Detecta diretório do projeto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# MLflow: usa tracking local (mesma pasta do model_registry.py)
MLFLOW_TRACKING_URI = os.path.join(PROJECT_DIR, "mlruns")

# -------------------- INICIALIZAÇÃO GCP --------------------
def init_gcp():
    """Inicializa conexão com GCP e configura MLflow."""
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}"
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    log.info(f"GCP inicializado - Projeto: {PROJECT_ID}")


def get_gcs_client():
    """Retorna cliente do Google Cloud Storage."""
    return storage.Client(project=PROJECT_ID)

# -------------------- LOAD DO MODELO --------------------
def _load_from_registry(model_name: str, stage_or_version: str):
    """
    Carrega modelo do MLflow Registry.
    
    Padrões de URI:
    - models:/ModelName/1  -> versão específica
    - models:/ModelName/Production  -> stage de produção
    """
    uri = f"models:/{model_name}/{stage_or_version}"
    log.info(f"Carregando modelo do MLflow Registry: {uri}")
    return mlflow.pyfunc.load_model(uri)


def _load_from_gcs(model_path: str):
    """
    Carrega modelo diretamente do GCS.
    Útil quando o modelo foi salvo como artifact.
    """
    log.info(f"Carregando modelo do GCS: {model_path}")
    return mlflow.pyfunc.load_model(model_path)


def load_model_resiliente(model_name: str, stage_or_version: str):
    """
    Carrega modelo com fallback.
    Primeiro tenta o Registry, depois tenta GCS direto.
    """
    try:
        return _load_from_registry(model_name, stage_or_version)
    except RestException as e:
        log.warning(f"Registry falhou ({e}). Tentando GCS direto...")
    except Exception as e:
        log.warning(f"Registry falhou ({e}). Tentando GCS direto...")

    # Fallback: carrega do GCS
    gcs_path = f"gs://{BUCKET_NAME}/mlflow/models/{model_name}/{stage_or_version}"
    return _load_from_gcs(gcs_path)

# -------------------- DADOS / SCORING --------------------
def load_dataframe_from_gcs(gcs_path: str) -> pd.DataFrame:
    """
    Carrega DataFrame diretamente do GCS.
    Suporta paths como: gs://bucket/path/file.csv
    """
    log.info(f"Lendo CSV do GCS: {gcs_path}")
    df = pd.read_csv(gcs_path)
    log.info(f"Shape: {df.shape}")
    return df


def load_dataframe_from_local_csv(csv_path: str | Path) -> pd.DataFrame:
    """Carrega DataFrame de arquivo local."""
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    log.info(f"Lendo CSV local: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Shape: {df.shape}")
    return df

def _expected_feature_names_from_signature(pyfunc_model) -> list[str] | None:
    try:
        sig = getattr(pyfunc_model, "metadata", None)
        if sig and getattr(sig, "signature", None) and sig.signature.inputs:
            return [inp.name for inp in sig.signature.inputs.inputs if inp.name]
    except Exception:
        pass
    return None

def _align_dataframe_to_features(df: pd.DataFrame, expected_cols: list[str], id_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids_out = pd.DataFrame(index=df.index)
    for c in id_cols:
        if c not in df.columns:
            raise KeyError(f"Coluna de ID '{c}' não encontrada.")
        ids_out[c] = df[c]

    work = df.drop(columns=id_cols, errors="ignore").copy()
    for col in ["target", "label", "inadimplente", "Status_Pagamento"]:
        if col in work.columns:
            work = work.drop(columns=[col])

    extras = [c for c in work.columns if c not in expected_cols]
    if extras:
        log.warning(f"Removendo colunas não vistas no treino: {extras}")
        work = work.drop(columns=extras)

    missing = [c for c in expected_cols if c not in work.columns]
    if missing:
        log.warning(f"Criando colunas faltantes com 0: {missing}")
        for c in missing:
            work[c] = 0

    work = work[expected_cols]

    # Tentativa leve de coerção numérica
    for c in work.columns:
        if work[c].dtype == "object":
            try:
                work[c] = pd.to_numeric(work[c], errors="raise")
            except Exception:
                work[c] = work[c].astype(str)
        if np.issubdtype(work[c].dtype, np.integer):
            work[c] = work[c].astype("float64")

    return work, ids_out

def score_dataframe(model, df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """
    Aplica o modelo nos dados e retorna predições.
    
    IMPORTANTE: Os dados de entrada precisam passar pelo mesmo
    pré-processamento usado no treino!
    """
    # Tenta obter features esperadas do modelo
    expected = _expected_feature_names_from_signature(model)
    
    if not expected:
        # Tenta do modelo sklearn subjacente
        native_model = getattr(getattr(model, "_model_impl", None), "sklearn_model", None)
        if native_model is not None and hasattr(native_model, "feature_names_in_"):
            expected = list(native_model.feature_names_in_)
    
    if not expected:
        # Tenta acessar o modelo diretamente (para XGBoost/RandomForest)
        try:
            unwrapped = model._model_impl.python_model
            if hasattr(unwrapped, "feature_names_in_"):
                expected = list(unwrapped.feature_names_in_)
        except:
            pass
    
    if not expected:
        # Último recurso: usa todas as colunas numéricas exceto ID e target
        log.warning("Não foi possível identificar features do modelo. Usando todas as colunas numéricas.")
        work = df.drop(columns=id_cols, errors="ignore").copy()
        for col in ["target", "label", "inadimplente", "Status_Pagamento"]:
            if col in work.columns:
                work = work.drop(columns=[col])
        
        # Guarda IDs
        ids_out = pd.DataFrame(index=df.index)
        for c in id_cols:
            if c in df.columns:
                ids_out[c] = df[c]
        
        # Usa apenas colunas numéricas
        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        X = work[numeric_cols]
        
        preds = model.predict(X)
        out = ids_out.reset_index(drop=True)
        out["prediction"] = preds
        
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    out["probability_inadimplente"] = proba[:, 1]
        except:
            pass
        
        return out

    # Fluxo normal com features conhecidas
    X, ids_out = _align_dataframe_to_features(df, expected_cols=expected, id_cols=id_cols)

    preds = model.predict(X)
    out = ids_out.reset_index(drop=True)

    if isinstance(preds, pd.DataFrame):
        out = pd.concat([out, preds.add_prefix("pred_").reset_index(drop=True)], axis=1)
    else:
        out["prediction"] = preds

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba_df = pd.DataFrame(proba)
            proba_df.columns = [f"proba_class_{i}" for i in range(proba_df.shape[1])]
            out = pd.concat([out, proba_df], axis=1)
    except Exception:
        pass

    return out

# -------------------- SALVAR CSV --------------------
def save_predictions_csv(df_out: pd.DataFrame, input_csv_path: str, output_prefix: str) -> Path:
    """
    Salva o CSV de predições localmente.
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    input_dir = Path(input_csv_path).resolve().parent
    candidate_path = input_dir / f"{output_prefix}_{ts}.csv"

    try:
        df_out.to_csv(candidate_path, index=False, encoding="utf-8")
        log.info(f"CSV salvo localmente: {candidate_path}")
        return candidate_path
    except Exception as e:
        log.warning(f"Não consegui salvar em {input_dir} ({e}). Usando ./outputs/ ...")
        Path("outputs").mkdir(exist_ok=True)
        fallback = Path("outputs") / f"{output_prefix}_{ts}.csv"
        df_out.to_csv(fallback, index=False, encoding="utf-8")
        log.info(f"CSV salvo em fallback: {fallback}")
        return fallback


def upload_predictions_to_gcs(local_path: str, gcs_folder: str = "predictions") -> str:
    """
    Faz upload das predições para o GCS.
    
    Este é um padrão comum em MLOps:
    - Predições ficam centralizadas no bucket
    - Fácil de integrar com BigQuery ou outros serviços
    """
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    
    filename = Path(local_path).name
    gcs_path = f"{gcs_folder}/{filename}"
    
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    
    full_path = f"gs://{BUCKET_NAME}/{gcs_path}"
    log.info(f"Upload para GCS concluído: {full_path}")
    return full_path

# -------------------- MAIN --------------------
def main():
    """
    Pipeline de Scoring em Batch.
    
    Fluxo típico de MLOps em produção:
    1. Inicializa conexão com cloud
    2. Carrega modelo versionado
    3. Processa dados de entrada
    4. Gera predições
    5. Salva resultados no storage
    """
    parser = argparse.ArgumentParser(description="Scoring batch com GCP.")
    parser.add_argument("--model-name", type=str, default="ModelRFC-GCP")
    parser.add_argument("--model-version", type=str, default="1")
    parser.add_argument("--registry-stage", type=str, default=None, help="Ex.: Production")
    parser.add_argument("--input-csv", type=str, required=True, help="Caminho local ou gs://")
    parser.add_argument("--id-cols", nargs="*", default=["ID_Cliente"])
    parser.add_argument("--output-prefix", type=str, default="predicoes_inadimplencia")
    parser.add_argument("--upload-output", type=str, default="false", help="true/false")
    args = parser.parse_args()

    # 1. Inicializa GCP
    init_gcp()

    # 2. Carrega modelo
    stage_or_version = args.registry_stage if args.registry_stage else args.model_version
    model = load_model_resiliente(args.model_name, stage_or_version)

    # 3. Carrega dados
    if args.input_csv.startswith("gs://"):
        df_in = load_dataframe_from_gcs(args.input_csv)
    else:
        df_in = load_dataframe_from_local_csv(args.input_csv)
    
    # 4. Gera predições
    df_out = score_dataframe(model, df_in, id_cols=args.id_cols)

    # 5. Salva resultados
    saved_path = save_predictions_csv(df_out, args.input_csv, args.output_prefix)

    # 6. (Opcional) Upload para GCS
    if args.upload_output.lower() == "true":
        upload_predictions_to_gcs(str(saved_path))

    log.info("Scoring concluído com sucesso!")


if __name__ == "__main__":
    main()
