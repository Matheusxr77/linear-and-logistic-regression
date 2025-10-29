# pages/2_Análise_de_Regressão_Logística.py

import streamlit as st
import pandas as pd
from src.models import train_logistic_regression
from src.plotting import (
    plot_roc_curve,
    plot_predicted_probabilities,
    plot_feature_importance,
    plot_logistic_regression_curve
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Regressão Logística", layout="wide")

st.title("Parte 2: Análise de Regressão Logística")
st.markdown("""
A Regressão Logística é usada para prever um resultado binário. Neste caso, preveremos se os Lakers irão **Vencer (1)** ou **Perder (0)** um jogo. O modelo produz uma curva em forma de "S" (sigmoide) que estima a probabilidade de vitória.

**Instruções:**
1.  **Escolha as Variáveis Independentes (X):** Selecione as estatísticas que você acredita serem importantes para prever o resultado de um jogo.
2.  Clique em **'Executar Análise'** para treinar o modelo e ver a probabilidade de vitória e outras métricas.
""")

# Verifica se os dados foram carregados
if 'team_data' not in st.session_state or st.session_state['team_data'].empty:
    st.error("Os dados não foram carregados. Por favor, volte para a página principal (app.py) para iniciar o carregamento.")
else:
    df = st.session_state['team_data']

    # ✅ Garante que exista uma coluna WIN
    if 'WL' in df.columns and 'WIN' not in df.columns:
        df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)

    # Define as colunas numéricas que podem ser usadas como variáveis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'WIN' in numeric_cols:
        numeric_cols.remove('WIN')


    # --- Interface do Usuário ---
    independent_vars = st.multiselect(
        "1. Escolha as Variáveis Independentes (X) para prever Vitória/Derrota:",
        options=numeric_cols,
        default=numeric_cols[:1],
        help="Estas são as variáveis que o modelo usará para calcular a probabilidade de vitória."
    )

    # --- Execução do Modelo ---
    if st.button("Executar Análise de Regressão Logística", type="primary"):
        if not independent_vars:
            st.warning("Por favor, selecione pelo menos uma variável independente.")
        else:
            with st.spinner("Treinando o modelo de Regressão Logística e gerando gráficos..."):
                # Treina o modelo
                results = train_logistic_regression(df, independent_vars)

                st.success("Análise concluída!")

                # --- Seção de Resultados ---
                st.subheader("Resultados do Modelo")

                # Exibe a probabilidade média de vitória
                avg_win_prob = results['y_pred_proba'].mean()
                st.metric(
                    label="Probabilidade Média de Vitória (no conjunto de teste)",
                    value=f"{avg_win_prob:.1%}"
                )
                st.info(f"Com base nas variáveis selecionadas, o modelo estima que, em média, os Lakers têm uma chance de vitória de **{avg_win_prob:.1%}** nos jogos do conjunto de teste.")

                # Relatório de Classificação e Matriz de Confusão
                col_report, col_matrix = st.columns(2)
                with col_report:
                    st.text("Relatório de Classificação:")
                    st.text(results['classification_report'])

                with col_matrix:
                    st.markdown("Matriz de Confusão:")
                    fig, ax = plt.subplots()
                    sns.heatmap(
                        results['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['Derrota (0)', 'Vitória (1)'],
                        yticklabels=['Derrota (0)', 'Vitória (1)']
                    )
                    ax.set_xlabel("Previsto")
                    ax.set_ylabel("Real")
                    st.pyplot(fig)

                # --- Seção de Gráficos ---
                st.subheader("Visualizações Gráficas")

                # Gráfico 1: Curva ROC
                st.markdown("#### 1. Curva ROC (Receiver Operating Characteristic)")
                st.pyplot(plot_roc_curve(results['y_test'], results['y_pred_proba']))
                st.caption("A Curva ROC avalia a performance do classificador. Um modelo ideal se aproxima do canto superior esquerdo. A Área Sob a Curva (AUC) quantifica a capacidade do modelo de distinguir entre as classes (vitória/derrota).")

                # Gráfico 2: Probabilidades Previstas
                st.markdown("#### 2. Gráfico de Distribuição das Probabilidades Previstas")
                st.pyplot(plot_predicted_probabilities(results['y_pred_proba']))
                st.caption("Este histograma mostra a distribuição das probabilidades de vitória previstas pelo modelo. Um bom modelo tende a mostrar picos perto de 0 (derrotas) e 1 (vitórias).")

                # Gráfico 3: Importância de Variáveis
                st.markdown("#### 3. Gráfico de Importância de Variáveis")
                st.pyplot(plot_feature_importance(results['coefficients'], "Impacto das Variáveis na Probabilidade de Vitória"))
                st.caption("Este gráfico exibe os coeficientes do modelo. Coeficientes positivos aumentam a probabilidade de vitória, enquanto negativos a diminuem. A magnitude indica a força do impacto.")

                # Gráfico 4: Diagrama de Dispersão com Curva de Regressão Logística
                st.markdown("#### 4. Diagrama de Dispersão com Curva de Regressão Logística")

                if len(independent_vars) > 0:

                    x_col = independent_vars[0]
                    X = df[[x_col]].values
                    y = df['WIN'].values

                    # Treina o modelo logístico
                    model = LogisticRegression()
                    model.fit(X, y)

                    # Gera curva sigmoide
                    X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    fig, ax = plt.subplots()
                    ax.scatter(X, y, alpha=0.3, label="Dados reais")
                    ax.plot(X_test, y_prob, color="orange", linewidth=2, label="Curva logística (sigmoide)")
                    ax.set_title(f"Probabilidade de Vitória vs. {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Probabilidade de Vitória (WIN)")
                    ax.legend()

                    st.pyplot(fig)
                    st.caption(f"Este gráfico mostra como a probabilidade de vitória muda com base na variável {x_col}. A curva em 'S' é característica da regressão logística.")
                else:
                    st.info("Selecione pelo menos uma variável independente para gerar o gráfico.")

                # Gráfico 5: Gráfico de Tendência com Intervalo de Confiança
                st.markdown("#### 5. Gráfico de Tendência com Intervalo de Confiança de 95%")
                st.info("O gráfico anterior ('Diagrama de Dispersão com Curva de Regressão Logística') já inclui o intervalo de confiança de 95% para a curva sigmoide, representado pela área sombreada. Isso visualiza a incerteza na estimativa da probabilidade de vitória.")