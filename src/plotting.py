# Trecho de src/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

def plot_regression_scatter(y_test, y_pred, x_test_col, x_label, y_label):
    """
    Gera um Diagrama de Dispersão com Linha de Regressão.
    """
    fig, ax = plt.subplots()
    sns.regplot(x=x_test_col, y=y_test, ax=ax, scatter_kws={'alpha':0.5}, label='Dados Reais')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Regressão Linear: {y_label} vs. {x_label}')
    ax.legend()
    return fig

def plot_predicted_vs_actual(y_test, y_pred, y_label):
    """
    Gera um gráfico de Previsão vs. Realidade.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha Ideal (y=x)')
    ax.set_xlabel(f'Valores Reais de {y_label}')
    ax.set_ylabel(f'Valores Previstos de {y_label}')
    ax.set_title('Previsão vs. Realidade')
    ax.legend()
    return fig

def plot_regression_confidence_interval(df, x_var, y_var):
    """
    Gera gráficos de tendência com intervalo de confiança de 95%.
    Se houver mais de uma variável independente, gera um gráfico para cada uma.
    """

    # Se for apenas uma variável, converte para lista para simplificar o loop
    if isinstance(x_var, str):
        x_var = [x_var]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(x_var),
        figsize=(6 * len(x_var), 5)
    )

    # Garante que axes seja iterável
    if len(x_var) == 1:
        axes = [axes]

    for ax, x in zip(axes, x_var):
        sns.regplot(x=x, y=y_var, data=df, ax=ax, ci=95, scatter_kws={'alpha': 0.3})
        ax.set_title(f'Tendência de {y_var} por {x} (IC 95%)')
        ax.set_xlabel(x)
        ax.set_ylabel(y_var)

    plt.tight_layout()
    return fig


def plot_regression_confusion_matrix(y_test, y_pred):
    """
    Gera uma Matriz de Confusão adaptada para o problema de regressão.
    """
    threshold = y_test.mean()
    y_test_class = (y_test > threshold).astype(int)
    y_pred_class = (y_pred > threshold).astype(int)
    
    cm = confusion_matrix(y_test_class, y_pred_class)
    
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Abaixo da Média', 'Acima da Média'])
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f'Matriz de Confusão (Limiar = {threshold:.2f})')
    return fig

# Trecho de src/plotting.py (continuação)

def plot_roc_curve(y_test, y_pred_proba):
    """
    Gera a Curva ROC e calcula a AUC.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    ax.plot([0.0, 1.0], [0.0, 1.0], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC (Receiver Operating Characteristic)')
    ax.legend(loc="lower right")
    return fig

def plot_predicted_probabilities(y_pred_proba):
    """
    Gera um histograma das probabilidades de vitória previstas.
    """
    fig, ax = plt.subplots()
    sns.histplot(y_pred_proba, bins=20, kde=True, ax=ax)
    ax.set_title('Distribuição das Probabilidades de Vitória Previstas')
    ax.set_xlabel('Probabilidade de Vitória Prevista')
    ax.set_ylabel('Frequência')
    return fig

def plot_feature_importance(coefficients, title):
    """
    Gera um gráfico de barras da importância (coeficientes) das features.
    """
    fig, ax = plt.subplots(figsize=(10, len(coefficients) * 0.5))
    coefficients_sorted = coefficients.reindex(coefficients.Coefficient.abs().sort_values(ascending=False).index)
    sns.barplot(x='Coefficient', y=coefficients_sorted.index, data=coefficients_sorted, ax=ax, palette='viridis')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('Impacto no Log-Odds da Vitória')
    ax.set_ylabel('Features')
    return fig

def plot_logistic_regression_curve(df, x_var, y_var):
    """
    Gera um diagrama de dispersão com a curva de regressão logística (sigmoide).
    """
    fig, ax = plt.subplots()
    sns.regplot(x=x_var, y=y_var, data=df, logistic=True, ci=95, 
                y_jitter=.03, scatter_kws={'alpha': 0.2})
    ax.set_title(f'Probabilidade de Vitória vs. {x_var}')
    ax.set_ylabel('Probabilidade de Vitória')
    ax.set_xlabel(x_var)
    return fig