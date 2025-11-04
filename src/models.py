from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def train_linear_regression(df: pd.DataFrame, independent_vars: list, dependent_var: str):
    """
    Treina um modelo de Regressão Linear Múltipla.

    Args:
        df (pd.DataFrame): O DataFrame com os dados.
        independent_vars (list): Lista de colunas das variáveis independentes (X).
        dependent_var (str): A coluna da variável dependente (y).

    Returns:
        dict: Um dicionário contendo o modelo, métricas e dados de predição.
    """
    # Remove linhas onde a variável dependente é NaN
    df_clean = df.dropna(subset=[dependent_var]).copy()
    
    if df_clean.empty:
        raise ValueError(f"Nenhum dado válido encontrado para a variável dependente '{dependent_var}'")
    
    X = df_clean[independent_vars]
    y = df_clean[dependent_var]
    
    # Verifica se há NaN nas variáveis independentes
    if X.isnull().any().any():
        print(f"⚠️ Valores faltantes detectados em {X.isnull().sum().sum()} células. Aplicando imputação...")
        
        # Imputa valores faltantes com a mediana
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=independent_vars, index=X.index)
        
        print(f"✓ Imputação concluída. Usando mediana para preencher valores faltantes.")
    
    # Divide os dados em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realiza predições no conjunto de teste
    y_pred = model.predict(X_test)

    # Calcula as métricas de avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Coleta os coeficientes para interpretação
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    intercept = model.intercept_

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mse": mse,
        "r2": r2,
        "coefficients": coefficients,
        "intercept": intercept
    }

def train_logistic_regression(df: pd.DataFrame, independent_vars: list):
    """
    Treina um modelo de Regressão Logística para prever vitória/derrota.

    Args:
        df (pd.DataFrame): O DataFrame com os dados.
        independent_vars (list): Lista de colunas das variáveis independentes (X).

    Returns:
        dict: Um dicionário contendo o modelo, métricas e dados de predição.
    """
    dependent_var = 'WIN'
    
    # Remove linhas onde WIN é NaN
    df_clean = df.dropna(subset=[dependent_var]).copy()
    
    if df_clean.empty:
        raise ValueError("Nenhum dado válido encontrado para a variável 'WIN'")
    
    # Verifica se há pelo menos 2 classes (vitória E derrota)
    unique_classes = df_clean[dependent_var].nunique()
    if unique_classes < 2:
        raise ValueError(
            f"Dados insuficientes para classificação binária. "
            f"Encontradas {unique_classes} classe(s), mas são necessárias 2 (vitória E derrota). "
            f"Experimente selecionar mais jogadores ou verificar os dados."
        )
    
    # Verifica quantidade mínima de amostras por classe
    class_counts = df_clean[dependent_var].value_counts()
    min_samples = class_counts.min()
    
    if min_samples < 2:
        raise ValueError(
            f"Dados insuficientes: a classe minoritária tem apenas {min_samples} amostra(s). "
            f"São necessárias pelo menos 2 amostras de cada classe. "
            f"Distribuição: {class_counts.to_dict()}"
        )
    
    if len(df_clean) < 10:
        raise ValueError(
            f"Dados insuficientes para análise confiável. "
            f"Total de registros: {len(df_clean)}. Mínimo recomendado: 10. "
            f"Selecione mais jogadores ou verifique os dados."
        )
    
    X = df_clean[independent_vars]
    y = df_clean[dependent_var]
    
    # Verifica se há NaN nas variáveis independentes
    if X.isnull().any().any():
        print(f"⚠️ Valores faltantes detectados em {X.isnull().sum().sum()} células. Aplicando imputação...")
        
        # Imputa valores faltantes com a mediana
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=independent_vars, index=X.index)
        
        print(f"✓ Imputação concluída. Usando mediana para preencher valores faltantes.")

    # Ajusta test_size se necessário
    test_size = 0.2
    if len(df_clean) < 20:
        test_size = 0.3  # Usa 30% para teste se dataset for pequeno
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
    except ValueError as e:
        # Se stratify falhar (muito poucos dados), tenta sem stratify
        print(f"⚠️ Não foi possível usar stratify: {e}")
        print("   Tentando divisão sem stratificação...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42
        )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilidade da classe 1 (Vitória)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Flatten the 2D coefficient array and create DataFrame correctly
    coefficients = pd.DataFrame(model.coef_.flatten(), index=X.columns, columns=['Coefficient'])

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "coefficients": coefficients
    }