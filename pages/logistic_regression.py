import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from src.models import train_logistic_regression
from src.plotting import (
    plot_roc_curve,
    plot_predicted_probabilities,
    plot_feature_importance,
    plot_logistic_regression_curve
)

st.set_page_config(page_title="Regressão Logística", layout="wide")

st.title("Parte 2: Análise de Regressão Logística")
st.markdown("""
A Regressão Logística é usada para prever um resultado binário. Neste caso, preveremos se os Lakers irão **Vencer (1)** ou **Perder (0)** um jogo. O modelo produz uma curva em forma de "S" (sigmoide) que estima a probabilidade de vitória.

**Equação da Regressão Logística:**
""")

st.latex(r"p = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n)}}")

st.markdown("""
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
        st.session_state['team_data'] = df

    # Define as colunas numéricas que podem ser usadas como variáveis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Remove a coluna WIN da lista de variáveis independentes
    if 'WIN' in numeric_cols:
        numeric_cols.remove('WIN')

    # --- Abas de Navegação ---
    tabs = st.tabs(["📊 Análise Principal", "📈 Exploração de Dados"])

    # ============================================================================
    # ABA 1: ANÁLISE PRINCIPAL
    # ============================================================================
    with tabs[0]:
        st.subheader("Modelo de Regressão Logística")

        independent_vars = st.multiselect(
            "1. Escolha as Variáveis Independentes (X) para prever Vitória/Derrota:",
            options=numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
            help="Estas são as variáveis que o modelo usará para calcular a probabilidade de vitória."
        )

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

                    # Exibe a equação da regressão logística
                    coef_str = " + ".join([f"({results['coefficients'].loc[var, 'Coefficient']:.4f} × {var})" for var in independent_vars])
                    st.markdown("**Equação do Modelo:**")
                    st.latex(f"p(Vitória) = \\frac{{1}}{{1 + e^{{-({results['model'].intercept_[0]:.4f} + {coef_str})}}}}")

                    # Exibe a probabilidade média de vitória
                    avg_win_prob = results['y_pred_proba'].mean()
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        label="Probabilidade Média de Vitória",
                        value=f"{avg_win_prob:.1%}",
                        help="Probabilidade média de vitória baseada no conjunto de teste"
                    )
                    col2.metric(
                        label="Acurácia do Modelo",
                        value=f"{results['accuracy']:.1%}",
                        help="Percentual de predições corretas"
                    )
                    
                    # Adiciona exemplo de predição específica
                    col3.metric(
                        label="Confiança Média",
                        value=f"{np.mean(np.maximum(results['y_pred_proba'], 1 - results['y_pred_proba'])):.1%}",
                        help="Confiança média do modelo nas predições"
                    )
                    
                    st.info(f"""
                    **Interpretação:** Com base nas variáveis selecionadas ({', '.join(independent_vars)}), 
                    o modelo estima que os Lakers têm uma probabilidade média de **{avg_win_prob:.1%}** de vencer 
                    os jogos do conjunto de teste.
                    
                    **Exemplo:** "Os Lakers têm {avg_win_prob:.0%} de chance de vencer baseado no desempenho atual."
                    """)

                    # Coeficientes do modelo
                    st.write("**Coeficientes do Modelo (Log-Odds):**")
                    st.dataframe(results['coefficients'])
                    st.caption("Coeficientes positivos aumentam a probabilidade de vitória; negativos a diminuem.")

                    # Relatório de Classificação e Matriz de Confusão
                    col_report, col_matrix = st.columns(2)
                    with col_report:
                        st.markdown("**Relatório de Classificação:**")
                        st.text(results['classification_report'])

                    with col_matrix:
                        st.markdown("**Matriz de Confusão:**")
                        fig, ax = plt.subplots()
                        sns.heatmap(
                            results['confusion_matrix'],
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            xticklabels=['Derrota (0)', 'Vitória (1)'],
                            yticklabels=['Derrota (0)', 'Vitória (1)'],
                            ax=ax
                        )
                        ax.set_xlabel("Previsto")
                        ax.set_ylabel("Real")
                        st.pyplot(fig)

                    # --- Seção de Gráficos ---
                    st.subheader("Visualizações Gráficas")

                    # Gráfico 1: Curva ROC
                    st.markdown("#### 1. Curva ROC (Receiver Operating Characteristic)")
                    st.pyplot(plot_roc_curve(results['y_test'], results['y_pred_proba']))
                    st.caption("A Curva ROC avalia a performance do classificador. Um modelo ideal se aproxima do canto superior esquerdo. A Área Sob a Curva (AUC) quantifica a capacidade do modelo.")

                    # Gráfico 2: Probabilidades Previstas
                    st.markdown("#### 2. Gráfico de Distribuição das Probabilidades Previstas")
                    st.pyplot(plot_predicted_probabilities(results['y_pred_proba']))
                    st.caption("Este histograma mostra a distribuição das probabilidades de vitória previstas. Um bom modelo tende a mostrar picos perto de 0 (derrotas) e 1 (vitórias).")

                    # Gráfico 3: Importância de Variáveis
                    st.markdown("#### 3. Gráfico de Importância de Variáveis")
                    st.pyplot(plot_feature_importance(results['coefficients'], "Impacto das Variáveis na Probabilidade de Vitória"))
                    st.caption("Coeficientes positivos aumentam a probabilidade de vitória, negativos a diminuem. A magnitude indica força do impacto.")

                    # Gráfico 4: Diagrama de Dispersão com Curva Sigmoide
                    st.markdown("#### 4. Diagrama de Dispersão com Curva de Regressão Logística (Sigmoide)")

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

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(X, y, alpha=0.3, label="Dados reais", s=50, edgecolors='k', linewidth=0.5)
                        ax.plot(X_test, y_prob, color="red", linewidth=3, label="Curva Sigmoide")
                        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Limiar de Decisão (0.5)')
                        ax.set_title(f"Probabilidade de Vitória vs. {x_col}", fontsize=14, fontweight='bold')
                        ax.set_xlabel(x_col, fontsize=12)
                        ax.set_ylabel("Probabilidade de Vitória (WIN)", fontsize=12)
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)
                        st.caption(f"Este gráfico mostra como a probabilidade de vitória muda com base na variável {x_col}. A curva em 'S' (sigmoide) é característica da regressão logística. Valores acima de 0.5 indicam predição de vitória.")
                    else:
                        st.info("Selecione pelo menos uma variável independente para gerar o gráfico.")

                    # Gráfico 5: Matriz de Confusão (já incluída acima, mas adicionar versão gráfica maior)
                    st.markdown("#### 5. Matriz de Confusão Detalhada")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        results['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='YlGnBu',
                        xticklabels=['Derrota (0)', 'Vitória (1)'],
                        yticklabels=['Derrota (0)', 'Vitória (1)'],
                        ax=ax,
                        cbar_kws={'label': 'Número de Predições'}
                    )
                    ax.set_xlabel("Previsto", fontsize=12, fontweight='bold')
                    ax.set_ylabel("Real", fontsize=12, fontweight='bold')
                    ax.set_title("Matriz de Confusão - Desempenho do Classificador", fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    st.caption("A matriz de confusão mostra os acertos e erros do modelo. Diagonal principal = predições corretas.")

                    # Gráfico 6: Gráfico de Tendência com Intervalo de Confiança
                    st.markdown("#### 6. Gráfico de Tendência com Intervalo de Confiança de 95%")
                    
                    fig_trend, axes_trend = plt.subplots(
                        nrows=1,
                        ncols=len(independent_vars),
                        figsize=(6 * len(independent_vars), 5)
                    )
                    
                    # Garante que axes seja iterável
                    if len(independent_vars) == 1:
                        axes_trend = [axes_trend]
                    
                    for ax, x_col in zip(axes_trend, independent_vars):
                        # Ordena os dados por x_col para plotagem suave
                        sorted_indices = df[x_col].argsort()
                        x_sorted = df[x_col].iloc[sorted_indices].values
                        y_sorted = df['WIN'].iloc[sorted_indices].values
                        
                        # Scatter plot dos dados reais
                        ax.scatter(x_sorted, y_sorted, alpha=0.3, s=30, color='blue', label='Dados Reais')
                        
                        # Treina modelo logístico para esta variável
                        X_var = df[[x_col]].values
                        y_var = df['WIN'].values
                        log_model = LogisticRegression(max_iter=1000)
                        log_model.fit(X_var, y_var)
                        
                        # Gera pontos para a curva suave
                        x_range = np.linspace(X_var.min(), X_var.max(), 300).reshape(-1, 1)
                        y_prob = log_model.predict_proba(x_range)[:, 1]
                        
                        # Calcula intervalo de confiança usando bootstrap
                        n_bootstrap = 100
                        predictions = np.zeros((len(x_range), n_bootstrap))
                        
                        for i in range(n_bootstrap):
                            # Amostragem bootstrap
                            indices = np.random.choice(len(X_var), size=len(X_var), replace=True)
                            X_boot = X_var[indices]
                            y_boot = y_var[indices]
                            
                            # Treina modelo no bootstrap
                            boot_model = LogisticRegression(max_iter=1000)
                            boot_model.fit(X_boot, y_boot)
                            predictions[:, i] = boot_model.predict_proba(x_range)[:, 1]
                        
                        # Calcula percentis para IC 95%
                        lower_bound = np.percentile(predictions, 2.5, axis=1)
                        upper_bound = np.percentile(predictions, 97.5, axis=1)
                        
                        # Plota curva e IC
                        ax.plot(x_range, y_prob, color='red', linewidth=2, label='Curva Logística')
                        ax.fill_between(x_range.ravel(), lower_bound, upper_bound, 
                                       alpha=0.3, color='red', label='IC 95%')
                        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1, 
                                  alpha=0.7, label='Limiar 0.5')
                        
                        ax.set_title(f'Probabilidade de Vitória vs. {x_col} (IC 95%)', 
                                    fontsize=12, fontweight='bold')
                        ax.set_xlabel(x_col, fontsize=11)
                        ax.set_ylabel('Probabilidade de Vitória', fontsize=11)
                        ax.set_ylim(-0.05, 1.05)
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig_trend)
                    st.caption("Visualiza a relação entre cada variável independente e a probabilidade de vitória. A área sombreada representa o intervalo de confiança de 95% calculado via bootstrap.")

    # ============================================================================
    # ABA 2: EXPLORAÇÃO DE DADOS
    # ============================================================================
    with tabs[1]:
        st.subheader("Exploração e Análise Exploratória de Dados")

        # Estatísticas Descritivas
        st.markdown("#### 📊 Estatísticas Descritivas")
        stats_df = df[numeric_cols].describe().T
        stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]
        st.dataframe(stats_df)

        # Distribuição das Variáveis
        st.markdown("#### 📈 Distribuição das Variáveis")
        fig, axes = plt.subplots(nrows=(len(numeric_cols) + 3) // 4, ncols=4, figsize=(16, 3 * ((len(numeric_cols) + 3) // 4)))
        fig.suptitle('Distribuição das Variáveis', fontsize=16, fontweight='bold')

        for ax, column in zip(axes.flatten(), numeric_cols):
            sns.histplot(df[column], kde=True, ax=ax, color='skyblue', bins=30)
            ax.set_title(column, fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

        # Matriz de Correlação
        st.markdown("#### 🔗 Correlações entre Variáveis")
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5, ax=ax)
        ax.set_title('Mapa de Calor da Correlação entre Variáveis', fontsize=14, fontweight='bold')
        st.pyplot(fig)

        # Distribuição de Vitórias e Derrotas
        st.markdown("#### 🏀 Distribuição de Vitórias e Derrotas")
        if 'WIN' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            win_counts = df['WIN'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4']
            ax.bar(['Derrotas (0)', 'Vitórias (1)'], [win_counts.get(0, 0), win_counts.get(1, 0)], color=colors, edgecolor='black', linewidth=2)
            ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
            ax.set_title('Distribuição de Resultados dos Jogos', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate([win_counts.get(0, 0), win_counts.get(1, 0)]):
                ax.text(i, v + 1, str(v), ha='center', fontsize=12, fontweight='bold')
            
            st.pyplot(fig)