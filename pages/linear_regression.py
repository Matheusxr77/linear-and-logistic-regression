# pages/1_Análise_de_Regressão_Linear.py

import streamlit as st
import pandas as pd
from src.models import train_linear_regression
from src.plotting import (
    plot_regression_scatter,
    plot_predicted_vs_actual,
    plot_regression_confidence_interval,
    plot_regression_confusion_matrix
)

st.set_page_config(page_title="Regressão Linear", layout="wide")

st.title("Parte 1: Análise de Regressão Linear")
st.markdown("""
Nesta seção, utilizamos a Regressão Linear Múltipla para modelar a relação entre uma variável de desempenho (dependente) e uma ou mais variáveis estatísticas (independentes).

**Instruções:**
1.  **Escolha a Variável Dependente (Y):** Selecione a estatística que você deseja prever.
2.  **Escolha as Variáveis Independentes (X):** Selecione uma ou mais estatísticas que você acredita que influenciam a variável dependente.
3.  Clique em **'Executar Análise'** para treinar o modelo e visualizar os resultados.
""")

# Verifica se os dados foram carregados e estão no estado da sessão
if 'team_data' not in st.session_state or st.session_state['team_data'].empty:
    st.error("Os dados não foram carregados. Por favor, volte para a página principal (app.py) para iniciar o carregamento.")
else:
    df = st.session_state['team_data']

    # Define as colunas numéricas que podem ser usadas como variáveis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Remove a variável 'WIN' que é categórica por natureza
    if 'WIN' in numeric_cols:
        numeric_cols.remove('WIN')

    # --- Interface do Usuário (Seleção de Variáveis) ---
    col1, col2 = st.columns(2)
    with col1:
        dependent_var = st.selectbox(
            "1. Escolha a Variável Dependente (Y) para prever:",
            options=numeric_cols,
            index=numeric_cols.index('PTS') if 'PTS' in numeric_cols else 0,
            help="Esta é a variável que o modelo tentará prever."
        )

    # Garante que a variável dependente não possa ser selecionada como independente
    available_independent_vars = [v for v in numeric_cols if v!= dependent_var]

    with col2:
        independent_vars = st.multiselect(
            "2. Escolha as Variáveis Independentes (X):",
            options=available_independent_vars,
            default=[available_independent_vars[0]] if available_independent_vars else [],
            help="Estas são as variáveis que o modelo usará para fazer a previsão."
        )

    # --- Execução do Modelo e Exibição dos Resultados ---
    if st.button("Executar Análise de Regressão Linear", type="primary"):
        if not independent_vars:
            st.warning("Por favor, selecione pelo menos uma variável independente.")
        else:
            with st.spinner("Treinando o modelo de Regressão Linear e gerando gráficos..."):
                # Treina o modelo
                results = train_linear_regression(df, independent_vars, dependent_var)

                st.success("Análise concluída!")

                # --- Seção de Resultados ---
                st.subheader("Resultados do Modelo")

                # Exibe a equação da regressão
                coef_str = " + ".join([f"({results['coefficients'].loc[var, 'Coefficient']:.2f} * {var})" for var in independent_vars])
                st.latex(f"{dependent_var} = {results['intercept']:.2f} + {coef_str}")

                # Exibe métricas e coeficientes
                col_metric1, col_metric2 = st.columns(2)
                col_metric1.metric(label="Coeficiente de Determinação (R²)", value=f"{results['r2']:.4f}")
                col_metric2.metric(label="Erro Quadrático Médio (MSE)", value=f"{results['mse']:.4f}")

                st.write("**Coeficientes do Modelo:**")
                st.dataframe(results['coefficients'])
                st.info(
                    "**Interpretação dos Coeficientes:** Cada coeficiente representa o quanto a variável dependente "
                    f"({dependent_var}) muda, em média, para cada aumento de uma unidade na variável independente, "
                    "mantendo todas as outras variáveis constantes."
                )

                # --- Seção de Gráficos ---
                st.subheader("Visualizações Gráficas")

                # Gráfico 1: Diagrama de Dispersão com Linha de Regressão
                st.markdown("#### 1. Diagrama de Dispersão com Linha de Regressão")
                st.pyplot(plot_regression_scatter(
                    y_test=results['y_test'],
                    y_pred=results['y_pred'],
                    x_test_col=results['X_test'][independent_vars],
                    x_label=independent_vars,
                    y_label=dependent_var
                ))
                st.caption(f"Este gráfico mostra a relação entre a variável dependente ({dependent_var}) e a primeira variável independente selecionada ({independent_vars}), com a linha de regressão ajustada pelo modelo.")

                # Gráfico 2: Previsão vs. Realidade
                st.markdown("#### 2. Gráfico de Previsão vs. Realidade")
                st.pyplot(plot_predicted_vs_actual(
                    y_test=results['y_test'],
                    y_pred=results['y_pred'],
                    y_label=dependent_var
                ))
                st.caption("Este gráfico compara os valores reais com os valores previstos pelo modelo. Pontos próximos à linha tracejada vermelha indicam predições precisas.")

                # Gráfico 3: Gráfico de Tendência com Intervalo de Confiança
                st.markdown("#### 3. Gráfico de Tendência com Intervalo de Confiança de 95%")
                st.pyplot(plot_regression_confidence_interval(
                    df=df,
                    x_var=independent_vars,
                    y_var=dependent_var
                ))
                st.caption(f"Visualiza a tendência entre {dependent_var} e {independent_vars}. A área sombreada representa o intervalo de confiança de 95% para a linha de regressão, indicando a incerteza da estimativa.")

                # Gráfico 4: Matriz de Confusão (Adaptada)
                st.markdown("#### 4. Matriz de Confusão (Adaptada para Regressão)")
                st.pyplot(plot_regression_confusion_matrix(
                    y_test=results['y_test'],
                    y_pred=results['y_pred']
                ))
                st.caption("Como a matriz de confusão é para modelos de classificação, adaptamos a análise: os valores foram classificados como 'Acima da Média' ou 'Abaixo da Média' para avaliar a capacidade do modelo de prever a magnitude do resultado.")