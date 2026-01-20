"""
TESTE LOCAL DA API (sem deploy)
================================
Roda a API localmente para testar o modelo.
Funciona no Colab/Vertex AI Workbench.
"""

import os
import sys
import joblib
import numpy as np

# Adicionar path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)


def find_and_load_model():
    """Encontra e carrega o modelo treinado"""
    import mlflow
    
    mlruns_path = os.path.join(PROJECT_DIR, "mlruns")
    
    if not os.path.exists(mlruns_path):
        # Tentar path do Colab
        mlruns_path = "/content/MLFlow-GCP/mlruns"
    
    print(f"📂 Buscando modelo em: {mlruns_path}")
    mlflow.set_tracking_uri(mlruns_path)
    
    # Buscar em todos os experiments
    experiments = mlflow.search_experiments()
    print(f"   Encontrados {len(experiments)} experiments")
    
    for exp in experiments:
        if exp.name in ['.trash', 'models']:
            continue
            
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        
        if runs.empty:
            continue
            
        # Ordenar por accuracy
        if 'metrics.accuracy' in runs.columns:
            runs = runs.sort_values("metrics.accuracy", ascending=False)
        
        best_run = runs.iloc[0]
        run_id = best_run["run_id"]
        print(f"   Tentando run: {run_id} do experiment '{exp.name}'")
        
        # Tentar carregar modelo
        for artifact_name in ["model_rfc", "model_xgb", "model"]:
            try:
                model_uri = f"runs:/{run_id}/{artifact_name}"
                model = mlflow.sklearn.load_model(model_uri)
                print(f"✅ Modelo sklearn carregado: {artifact_name}")
                return model
            except:
                try:
                    model = mlflow.xgboost.load_model(model_uri)
                    print(f"✅ Modelo XGBoost carregado: {artifact_name}")
                    return model
                except:
                    continue
    
    raise Exception("Modelo não encontrado!")


def test_prediction(model):
    """Testa predição com dados de exemplo"""
    
    # Carregar dados de teste (usar df_transformado)
    df_path = os.path.join(PROJECT_DIR, "df_transformado.csv")
    
    if not os.path.exists(df_path):
        df_path = "/content/MLFlow-GCP/df_transformado.csv"
    
    if os.path.exists(df_path):
        import pandas as pd
        df = pd.read_csv(df_path)
        
        # Colunas que NÃO são features (remover antes de prever)
        cols_to_drop = ["Inadimplente", "ID_Cliente", "Status_Pagamento"]
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        
        # Separar target se existir
        target = "Inadimplente"
        if target in df.columns:
            y_real = df[target].head(5).values
        else:
            y_real = None
        
        # Features para predição (remover colunas não-features)
        X_test = df.drop(columns=cols_to_drop, errors='ignore').head(5)
        
        print(f"   Features usadas ({len(X_test.columns)}): {list(X_test.columns)[:5]}...")
        
        # Fazer predições
        predictions = model.predict(X_test)
        
        # Probabilidades
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_test)
        else:
            probas = None
        
        print("\n" + "=" * 60)
        print("📊 TESTE DE PREDIÇÕES")
        print("=" * 60)
        
        for i in range(len(predictions)):
            pred = predictions[i]
            label = "⚠️ Inadimplente" if pred == 1 else "✅ Adimplente"
            
            print(f"\nCliente {i+1}:")
            print(f"   Predição: {label}")
            
            if probas is not None:
                print(f"   Confiança: {max(probas[i]):.2%}")
            
            if y_real is not None:
                real = "Inadimplente" if y_real[i] == 1 else "Adimplente"
                match = "✓" if pred == y_real[i] else "✗"
                print(f"   Real: {real} {match}")
        
        return predictions
    else:
        print("⚠️ Arquivo df_transformado.csv não encontrado")
        print("   Testando com dados fictícios...")
        
        # Criar dados fictícios (ajustar conforme features reais)
        X_test = np.random.randn(3, 10)  # 3 amostras, 10 features
        predictions = model.predict(X_test)
        
        for i, pred in enumerate(predictions):
            label = "⚠️ Inadimplente" if pred == 1 else "✅ Adimplente"
            print(f"Cliente {i+1}: {label}")
        
        return predictions


def main():
    print("=" * 60)
    print("🧪 TESTE LOCAL DO MODELO")
    print("=" * 60)
    
    # 1. Carregar modelo
    print("\n📌 Carregando modelo...")
    model = find_and_load_model()
    
    # 2. Testar predições
    print("\n📌 Testando predições...")
    test_prediction(model)
    
    # 3. Salvar modelo para uso posterior
    model_path = os.path.join(SCRIPT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo salvo em: {model_path}")
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO!")
    print("=" * 60)
    print(f"\nPara usar o modelo:")
    print(f"   import joblib")
    print(f"   model = joblib.load('{model_path}')")
    print(f"   predictions = model.predict(X)")


if __name__ == "__main__":
    main()
