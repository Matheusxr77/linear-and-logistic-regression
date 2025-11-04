import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
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

**Equação da Regressão Linear:**
""")

st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \varepsilon")

st.markdown("""
**Hipóteses que podemos testar:**
- Um determinado Jogador fará Y pontos?
- Um determinado Jogador fará Y rebotes?
- Um determinado Jogador fará Y assistências?
- O time fará X Pontos no jogo?
- O time fará X Rebotes no jogo?
- O time fará X Assistências no jogo?

**Instruções:**
1.  **Escolha o(s) Jogador(es):** Selecione um ou mais jogadores para análise.
2.  **Escolha a Variável Dependente (Y):** Selecione a estatística que você deseja prever.
3.  **Escolha as Variáveis Independentes (X):** Selecione uma ou mais estatísticas que você acredita que influenciam a variável dependente.
4.  Clique em **'Executar Análise'** para treinar o modelo e visualizar os resultados.
""")

# Verifica se os dados foram carregados e estão no estado da sessão
if 'player_data' not in st.session_state or st.session_state['player_data'].empty:
    st.error("Os dados dos jogadores não foram carregados. Por favor, volte para a página principal (app.py) para iniciar o carregamento.")
else:
    player_df = st.session_state['player_data']

    # --- SELEÇÃO DE JOGADORES ---
    st.subheader("🏀 Seleção de Jogadores")
    
    # Identifica a coluna de nome do jogador
    name_column = None
    for col in ['PLAYER_NAME', 'Player_Name', 'PLAYER', 'Player']:
        if col in player_df.columns:
            name_column = col
            break
    
    if not name_column:
        st.error("Não foi possível identificar a coluna com os nomes dos jogadores nos dados carregados.")
        st.info(f"Colunas disponíveis: {', '.join(player_df.columns.tolist())}")
        st.stop()
    
    # Obtém lista única de jogadores
    available_players = sorted(player_df[name_column].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_players = st.multiselect(
            "Selecione um ou mais jogadores para análise:",
            options=available_players,
            default=[available_players[0]] if available_players else [],
            help="Você pode selecionar múltiplos jogadores para análise comparativa"
        )
    
    with col2:
        st.metric("Jogadores disponíveis", len(available_players))
        st.metric("Jogadores selecionados", len(selected_players))
    
    if not selected_players:
        st.warning("⚠️ Por favor, selecione pelo menos um jogador para continuar com a análise.")
        st.stop()
    
    # Filtra dados pelos jogadores selecionados
    df = player_df[player_df[name_column].isin(selected_players)].copy()
    
    st.success(f"✓ {len(selected_players)} jogador(es) selecionado(s): {', '.join(selected_players)}")
    
    # Mostra estatísticas resumidas dos jogadores selecionados
    with st.expander("📊 Estatísticas dos Jogadores Selecionados"):
        stats_cols = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'MIN']
        available_stats = [col for col in stats_cols if col in df.columns]
        
        if available_stats:
            summary_stats = df.groupby(name_column)[available_stats].agg(['mean', 'std', 'min', 'max'])
            st.dataframe(summary_stats.round(2), use_container_width=True)
        else:
            st.warning("Estatísticas básicas não encontradas nos dados.")

    # Lista detalhada de jogadores (similar à regressão logística)
    with st.expander("📋 Lista Detalhada de Jogadores Selecionados"):
        for player in selected_players:
            player_data = df[df[name_column] == player]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {player}")
            
            with col2:
                st.metric("Jogos", len(player_data))
            
            with col3:
                if 'MIN' in player_data.columns:
                    avg_min = player_data['MIN'].mean()
                    st.metric("Min/Jogo", f"{avg_min:.1f}")
            
            # Métricas principais
            metrics_row = st.columns(5)
            
            metric_configs = [
                ('PTS', 'Pontos', '🏀'),
                ('REB', 'Rebotes', '🔄'),
                ('AST', 'Assistências', '🎯'),
                ('FG_PCT', 'FG%', '📊'),
                ('FG3_PCT', '3P%', '🎪')
            ]
            
            for idx, (col_name, label, icon) in enumerate(metric_configs):
                if col_name in player_data.columns:
                    avg_value = player_data[col_name].mean()
                    if 'PCT' in col_name:
                        metrics_row[idx].metric(f"{icon} {label}", f"{avg_value:.1%}")
                    else:
                        metrics_row[idx].metric(f"{icon} {label}", f"{avg_value:.1f}")
            
            st.divider()

    # Define as colunas numéricas que podem ser usadas como variáveis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Remove colunas de ID que não são úteis para análise
    exclude_cols = ['PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'WIN']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # --- Abas de Navegação ---
    tabs = st.tabs(["📊 Análise Principal", "📈 Exploração de Dados", "🎁 Análise de Resíduos"])

    # ============================================================================
    # ABA 1: ANÁLISE PRINCIPAL
    # ============================================================================
    with tabs[0]:
        st.subheader("Modelo de Regressão Linear")

        col1, col2 = st.columns(2)
        with col1:
            dependent_var = st.selectbox(
                "1. Escolha a Variável Dependente (Y) para prever:",
                options=numeric_cols,
                index=numeric_cols.index('PTS') if 'PTS' in numeric_cols else 0,
                help="Esta é a variável que o modelo tentará prever."
            )

        available_independent_vars = [v for v in numeric_cols if v != dependent_var]

        with col2:
            independent_vars = st.multiselect(
                "2. Escolha as Variáveis Independentes (X):",
                options=available_independent_vars,
                default=[available_independent_vars[0]] if available_independent_vars else [],
                help="Estas são as variáveis que o modelo usará para fazer a previsão."
            )

        if st.button("Executar Análise de Regressão Linear", type="primary"):
            if not independent_vars:
                st.warning("Por favor, selecione pelo menos uma variável independente.")
            else:
                # Verifica quantidade de dados válidos antes de treinar
                df_valid = df.dropna(subset=[dependent_var] + independent_vars)
                
                if len(df_valid) < 10:
                    st.error(f"""
                    ⚠️ Dados insuficientes para análise!
                    
                    - Total de registros: {len(df)}
                    - Registros válidos (sem NaN): {len(df_valid)}
                    - Mínimo necessário: 10
                    
                    Sugestões:
                    1. Selecione outras variáveis com menos valores faltantes
                    2. Selecione mais jogadores
                    3. Verifique se os dados foram carregados corretamente
                    """)
                    
                    # Mostra diagnóstico de valores faltantes
                    with st.expander("📊 Diagnóstico de Valores Faltantes"):
                        missing_stats = pd.DataFrame({
                            'Valores Faltantes': df[independent_vars + [dependent_var]].isnull().sum(),
                            'Percentual (%)': (df[independent_vars + [dependent_var]].isnull().sum() / len(df) * 100).round(2)
                        })
                        st.dataframe(missing_stats)
                else:
                    with st.spinner("Treinando o modelo de Regressão Linear e gerando gráficos..."):
                        results = train_linear_regression(df, independent_vars, dependent_var)
                        
                        st.success("Análise concluída!")
                        
                        # Mostra aviso se houve imputação
                        if df[independent_vars].isnull().any().any():
                            st.info(f"""
                            ℹ️ **Nota sobre valores faltantes:**
                            Alguns valores faltantes foram detectados e preenchidos automaticamente 
                            com a mediana das respectivas variáveis para permitir a análise.
                            
                            - Registros originais: {len(df)}
                            - Registros após limpeza: {len(df_valid)}
                            """)

                        # --- Seção de Resultados ---
                        st.subheader("Resultados do Modelo")

                        # Exibe a equação da regressão
                        coef_str = " + ".join([f"({results['coefficients'].loc[var, 'Coefficient']:.4f} × {var})" for var in independent_vars])
                        st.markdown("**Equação de Regressão Ajustada:**")
                        st.latex(f"{dependent_var} = {results['intercept']:.4f} + {coef_str} + \\varepsilon")

                        # Exibe métricas e coeficientes
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        col_metric1.metric(label="Coeficiente de Determinação (R²)", value=f"{results['r2']:.4f}",
                                          help="Quanto da variação em Y é explicada por X. Varia de 0 a 1.")
                        col_metric2.metric(label="Erro Quadrático Médio (MSE)", value=f"{results['mse']:.4f}",
                                          help="Média dos erros ao quadrado. Quanto menor, melhor.")
                        col_metric3.metric(label="Raiz do MSE (RMSE)", value=f"{np.sqrt(results['mse']):.4f}",
                                          help="Erro médio em unidades da variável Y.")

                        st.write("**Coeficientes do Modelo:**")
                        st.dataframe(results['coefficients'])
                        st.info(
                            f"""
                            **Interpretação dos Coeficientes:** 
                            
                            Cada coeficiente (β) representa o quanto a variável dependente ({dependent_var}) 
                            muda, em média, para cada aumento de **uma unidade** na variável independente correspondente, 
                            **mantendo todas as outras variáveis constantes** (ceteris paribus).
                            
                            **Exemplo:** Se o coeficiente de 'FG%' for 2.5, significa que para cada aumento de 1% 
                            na porcentagem de arremessos convertidos, espera-se um aumento de 2.5 pontos em {dependent_var}.
                            """
                        )

                        # --- Seção de Gráficos ---
                        st.subheader("Visualizações Gráficas")

                        # Gráfico 1: Diagrama de Dispersão com Linha de Regressão
                        st.markdown("#### 1. Diagrama de Dispersão com Linha de Regressão")
                        st.pyplot(plot_regression_scatter(
                            y_test=results['y_test'],
                            y_pred=results['y_pred'],
                            x_test_col=results['X_test'].iloc[:, 0],
                            x_label=independent_vars[0],
                            y_label=dependent_var
                        ))
                        st.caption(f"Este gráfico mostra a relação entre a variável dependente ({dependent_var}) e a primeira variável independente selecionada ({independent_vars[0]}), com a linha de regressão ajustada pelo modelo.")

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
        num_plots = len(numeric_cols)
        ncols = 4
        nrows = (num_plots + ncols - 1) // ncols
        
        # Fix: Adjust figure size to prevent tight_layout warnings
        fig, axes = plt.subplots(
            nrows=nrows, 
            ncols=ncols, 
            figsize=(18, 4 * nrows)  # Increased width from 16 to 18
        )
        fig.suptitle('Distribuição das Variáveis', fontsize=16, fontweight='bold')

        for ax, column in zip(axes.flatten(), numeric_cols):
            sns.histplot(df[column], kde=True, ax=ax, color='skyblue', bins=30)
            ax.set_title(column, fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Hide unused subplots
        for ax in axes.flatten()[len(numeric_cols):]:
            ax.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        # Matriz de Correlação
        st.markdown("#### 🔗 Correlações entre Variáveis")
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5, ax=ax)
        ax.set_title('Mapa de Calor da Correlação entre Variáveis', fontsize=14, fontweight='bold')
        st.pyplot(fig)

    # ============================================================================
    # ABA 3: ANÁLISE DE RESÍDUOS
    # ============================================================================
    with tabs[2]:
        st.subheader("🎁 Análise de Resíduos - Validação das Premissas")
        st.markdown("""
        A análise de resíduos valida as premissas fundamentais da regressão linear:
        1. **Linearidade**: Relação linear entre X e y
        2. **Homocedasticidade**: Variância constante dos erros
        3. **Normalidade**: Resíduos seguem distribuição normal
        4. **Independência**: Ausência de padrões nos resíduos
        
        **Resíduos**: Diferenças entre valores reais e preditos: $e_i = y_i - \\hat{y}_i$
        """)

        col1, col2 = st.columns(2)
        with col1:
            dependent_var_res = st.selectbox(
                "Escolha a Variável Dependente (Y):",
                options=numeric_cols,
                index=numeric_cols.index('PTS') if 'PTS' in numeric_cols else 0,
                key="res_dependent"
            )

        available_independent_vars_res = [v for v in numeric_cols if v != dependent_var_res]

        with col2:
            independent_vars_res = st.multiselect(
                "Escolha as Variáveis Independentes (X):",
                options=available_independent_vars_res,
                default=[available_independent_vars_res[0]] if available_independent_vars_res else [],
                key="res_independent"
            )

        if st.button("Gerar Análise de Resíduos", type="primary", key="residuals_btn"):
            if not independent_vars_res:
                st.warning("Selecione pelo menos uma variável independente.")
            else:
                with st.spinner("Gerando análise de resíduos..."):
                    results_res = train_linear_regression(df, independent_vars_res, dependent_var_res)
                    
                    residuals = results_res['y_test'] - results_res['y_pred']
                    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle('Análise de Resíduos - Validação das Premissas da Regressão Linear', 
                                fontsize=14, fontweight='bold', y=1.00)

                    # 1. Resíduos vs Valores Preditos
                    axes[0, 0].scatter(results_res['y_pred'], residuals, alpha=0.5, edgecolors='k', linewidth=0.5, s=30)
                    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Resíduo = 0')
                    axes[0, 0].set_xlabel('Valores Preditos', fontsize=12, fontweight='bold')
                    axes[0, 0].set_ylabel('Resíduos', fontsize=12, fontweight='bold')
                    axes[0, 0].set_title('1. Resíduos vs Predições\n✓ Padrão aleatório indica homocedasticidade', fontsize=12, fontweight='bold')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)

                    # 2. Histograma dos Resíduos
                    axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Resíduos')
                    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Média = 0')
                    mu, sigma = residuals.mean(), residuals.std()
                    x = np.linspace(residuals.min(), residuals.max(), 100)
                    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
                    axes[0, 1].set_xlabel('Resíduos', fontsize=12, fontweight='bold')
                    axes[0, 1].set_ylabel('Densidade', fontsize=12, fontweight='bold')
                    axes[0, 1].set_title('2. Distribuição dos Resíduos\n✓ Deve seguir distribuição normal', fontsize=12, fontweight='bold')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)

                    # 3. Q-Q Plot
                    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
                    axes[1, 0].get_lines()[0].set_markerfacecolor('blue')
                    axes[1, 0].get_lines()[0].set_markeredgecolor('black')
                    axes[1, 0].get_lines()[0].set_markersize(5)
                    axes[1, 0].get_lines()[1].set_color('red')
                    axes[1, 0].get_lines()[1].set_linewidth(2)
                    axes[1, 0].set_title('3. Q-Q Plot\n✓ Pontos na linha diagonal indicam normalidade', fontsize=12, fontweight='bold')
                    axes[1, 0].grid(True, alpha=0.3)

                    # 4. Scale-Location Plot
                    axes[1, 1].scatter(results_res['y_pred'], np.abs(standardized_residuals), alpha=0.5, edgecolors='k', linewidth=0.5, s=30)
                    axes[1, 1].axhline(y=2, color='orange', linestyle=':', linewidth=2, label='±2σ (95%)')
                    axes[1, 1].axhline(y=3, color='red', linestyle=':', linewidth=2, label='±3σ (99.7%)')
                    axes[1, 1].set_xlabel('Valores Preditos', fontsize=12, fontweight='bold')
                    axes[1, 1].set_ylabel('|Resíduos Padronizados|', fontsize=12, fontweight='bold')
                    axes[1, 1].set_title('4. Scale-Location Plot\n✓ Linha horizontal indica variância constante', fontsize=12, fontweight='bold')
                    axes[1, 1].legend(loc='upper right')
                    axes[1, 1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Estatísticas dos resíduos
                    st.markdown("#### 📊 Estatísticas dos Resíduos")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Média", f"{np.mean(residuals):.6f}")
                    col2.metric("Desvio Padrão", f"{np.std(residuals):.4f}")
                    col3.metric("Mínimo", f"{np.min(residuals):.4f}")
                    col4.metric("Máximo", f"{np.max(residuals):.4f}")

                    # Teste de Normalidade
                    st.markdown("#### 🔬 Teste de Normalidade (Shapiro-Wilk)")
                    sample_size = min(5000, len(residuals))
                    sample_residuals = np.random.choice(residuals, sample_size, replace=False)
                    statistic, p_value = stats.shapiro(sample_residuals)

                    col1, col2 = st.columns(2)
                    col1.metric("Estatística W", f"{statistic:.6f}")
                    col2.metric("p-valor", f"{p_value:.6f}")

                    if p_value > 0.05:
                        st.success("✓ Resíduos são normais (p > 0.05)")
                    else:
                        st.warning("✗ Há desvios da normalidade (p ≤ 0.05)")

                    # Detecção de Outliers
                    st.markdown("#### ⚠️ Detecção de Outliers")
                    outliers_2sigma = np.sum(np.abs(standardized_residuals) > 2)
                    outliers_3sigma = np.sum(np.abs(standardized_residuals) > 3)
                    total = len(residuals)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total de observações", total)
                    col2.metric("Além de ±2σ", f"{outliers_2sigma} ({outliers_2sigma/total*100:.2f}%)")
                    col3.metric("Além de ±3σ", f"{outliers_3sigma} ({outliers_3sigma/total*100:.2f}%)")