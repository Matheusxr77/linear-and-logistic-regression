# Trecho de src/models.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import pandas as pd

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
    X = df[independent_vars]
    y = df[dependent_var]

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

# Trecho de src/models.py (continuação)

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
    X = df[independent_vars]
    y = df[dependent_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilidade da classe 1 (Vitória)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Fix: Flatten the 2D coefficient array and create DataFrame correctly
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