import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from src.models import train_logistic_regression
from src.plotting import (
    plot_roc_curve,
    plot_predicted_probabilities,
    plot_feature_importance,
    plot_logistic_regression_curve,
    plot_regression_confidence_interval
)

st.set_page_config(page_title="Regressão Logística", layout="wide")

st.title("Parte 2: Análise de Regressão Logística")
st.markdown("""
Nesta seção, utilizamos a **Regressão Logística** para prever a **probabilidade de vitória ou derrota**, com base em estatísticas de desempenho.

**Equação da Regressão Logística (Função Sigmoide):**
""")

st.latex(r"P(\text{Vitória}|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n)}}")

st.markdown("""
**Interpretação:**
- Se $P(\text{Vitória}) > 0.5$: O modelo prevê **vitória**
- Se $P(\text{Vitória}) < 0.5$: O modelo prevê **derrota**

**Você pode escolher analisar:**
1. **Jogadores Individuais**: Probabilidade de vitória do time quando o jogador tem bom desempenho
2. **Time (Lakers)**: Probabilidade de vitória do time baseado nas estatísticas coletivas
""")

# --- SELEÇÃO DE MODO DE ANÁLISE ---
st.subheader("🎯 Escolha o Tipo de Análise")

analysis_mode = st.radio(
    "Selecione o modo de análise:",
    options=["🏀 Por Jogador", "🏆 Por Time (Lakers)"],
    horizontal=True,
    help="Escolha se quer analisar jogadores individuais ou o desempenho geral do time"
)

# ============================================================================
# MODO: POR JOGADOR
# ============================================================================
if analysis_mode == "🏀 Por Jogador":
    if 'player_data' not in st.session_state or st.session_state['player_data'].empty:
        st.error("Os dados dos jogadores não foram carregados. Por favor, volte para a página principal (app.py) para iniciar o carregamento.")
        st.stop()
    
    player_df = st.session_state['player_data'].copy()
    
    # Identifica a coluna de nome do jogador
    name_column = None
    for col in ['PLAYER_NAME', 'Player_Name', 'PLAYER', 'Player']:
        if col in player_df.columns:
            name_column = col
            break
    
    if not name_column:
        st.error("Não foi possível identificar a coluna com os nomes dos jogadores nos dados carregados.")
        st.stop()
    
    # Seleção de jogadores
    st.subheader("🏀 Seleção de Jogadores")
    available_players = sorted(player_df[name_column].unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_players = st.multiselect(
            "Selecione um ou mais jogadores para análise:",
            options=available_players,
            default=[available_players[0]] if available_players else [],
            help="Análise da probabilidade de vitória do time baseado no desempenho desses jogadores"
        )
    
    with col2:
        st.metric("Jogadores disponíveis", len(available_players))
        st.metric("Jogadores selecionados", len(selected_players))
    
    if not selected_players:
        st.warning("⚠️ Por favor, selecione pelo menos um jogador para continuar com a análise.")
        st.stop()
    
    # Filtra dados pelos jogadores selecionados
    df = player_df[player_df[name_column].isin(selected_players)].copy()
    
    # Verifica/cria a coluna WIN
    if 'WL' in df.columns:
        df['WIN'] = (df['WL'] == 'W').astype(int)
    elif 'WIN' not in df.columns:
        st.error("Coluna de resultado do jogo (WIN ou WL) não encontrada nos dados.")
        st.stop()
    
    st.success(f"✓ {len(selected_players)} jogador(es) selecionado(s): {', '.join(selected_players)}")
    
    # Mostra estatísticas dos jogadores selecionados
    with st.expander("📊 Estatísticas dos Jogadores Selecionados"):
        stats_cols = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'MIN']
        available_stats = [col for col in stats_cols if col in df.columns]
        
        if available_stats:
            summary_stats = df.groupby(name_column)[available_stats].agg(['mean', 'std', 'min', 'max'])
            st.dataframe(summary_stats.round(2))

# ============================================================================
# MODO: POR TIME
# ============================================================================
else:  # analysis_mode == "🏆 Por Time (Lakers)"
    if 'team_data' not in st.session_state or st.session_state['team_data'].empty:
        st.error("Os dados do time não foram carregados. Por favor, volte para a página principal (app.py) para iniciar o carregamento.")
        st.stop()

    df = st.session_state['team_data'].copy()
    
    # Verifica/cria a coluna WIN
    if 'WL' in df.columns:
        df['WIN'] = (df['WL'] == 'W').astype(int)
    elif 'WIN' not in df.columns:
        st.error("Coluna de resultado do jogo (WIN ou WL) não encontrada nos dados.")
        st.stop()

# ============================================================================
# ANÁLISE COMUM (AMBOS OS MODOS)
# ============================================================================

# Mostra informações do dataset
st.info(f"""
📊 **Dataset Carregado:**
- Modo de análise: **{analysis_mode}**
- Total de registros: {len(df)}
- Vitórias: {df['WIN'].sum()} ({df['WIN'].mean():.1%})
- Derrotas: {(df['WIN'] == 0).sum()} ({(1 - df['WIN'].mean()):.1%})
""")

# Define as colunas numéricas disponíveis
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
exclude_cols = ['PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'WIN', 'SEASON_ID']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Remove colunas que podem ter apenas um valor
for col in numeric_cols.copy():
    if df[col].nunique() <= 1:
        numeric_cols.remove(col)

# --- Abas de Navegação ---
tabs = st.tabs(["📊 Análise Principal", "📈 Exploração de Dados", "🎯 Simulador de Probabilidade"])

# ============================================================================
# ABA 1: ANÁLISE PRINCIPAL
# ============================================================================
with tabs[0]:
    if analysis_mode == "🏀 Por Jogador":
        st.subheader(f"🏀 Modelo de Regressão Logística - {', '.join(selected_players)}")
        st.info("""
        **Hipótese:** Qual a probabilidade do time vencer quando este(s) jogador(es) tem/têm bom desempenho?
        
        O modelo analisará como as estatísticas individuais dos jogadores selecionados 
        impactam na probabilidade de vitória do time.
        """)
    else:
        st.subheader("🏀 Modelo de Regressão Logística - Los Angeles Lakers")
        st.info("""
        **Hipótese:** Qual a probabilidade do time Los Angeles Lakers vencer uma partida 
        baseado em suas estatísticas de desempenho?
        """)

    # Seleção de variáveis com sugestões
    st.markdown("#### Seleção de Variáveis Independentes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sugestões de variáveis importantes
        suggested_vars = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        default_vars = [var for var in suggested_vars if var in numeric_cols][:3]
        
        independent_vars = st.multiselect(
            "Escolha as Variáveis Independentes (X) para prever vitória:",
            options=numeric_cols,
            default=default_vars if default_vars else numeric_cols[:3],
            help="Estas variáveis serão usadas para calcular a probabilidade de vitória."
        )
    
    with col2:
        st.markdown("**💡 Sugestões:**")
        st.markdown("""
        - **PTS**: Pontos marcados
        - **FG_PCT**: % Arremessos
        - **REB**: Rebotes
        - **AST**: Assistências
        - **TOV**: Turnovers (-)
        """)

    if st.button("🚀 Executar Análise de Regressão Logística", type="primary"):
        if not independent_vars:
            st.warning("⚠️ Por favor, selecione pelo menos uma variável independente.")
        elif df['WIN'].nunique() < 2:
            st.error("""
            ❌ **Dados insuficientes para análise de classificação!**
            
            Os dados selecionados contêm apenas **um tipo de resultado** (apenas vitórias OU apenas derrotas).
            
            **Soluções:**
            1. Selecione **mais jogadores** para ter uma amostra maior
            2. Verifique se os jogadores selecionados têm jogos com ambos os resultados (vitórias E derrotas)
            3. No modo "Por Time", verifique se há dados suficientes da temporada
            """)
            
            # Mostra diagnóstico
            with st.expander("📊 Diagnóstico dos Dados"):
                win_count = df['WIN'].sum()
                loss_count = (df['WIN'] == 0).sum()
                st.write(f"**Total de jogos:** {len(df)}")
                st.write(f"**Vitórias:** {win_count}")
                st.write(f"**Derrotas:** {loss_count}")
                
                if analysis_mode == "🏀 Por Jogador":
                    st.write("**Distribuição por jogador:**")
                    player_stats = df.groupby(name_column)['WIN'].agg(['count', 'sum'])
                    player_stats.columns = ['Total de Jogos', 'Vitórias']
                    player_stats['Derrotas'] = player_stats['Total de Jogos'] - player_stats['Vitórias']
                    st.dataframe(player_stats)
        else:
            with st.spinner("🔄 Treinando o modelo de Regressão Logística e gerando gráficos..."):
                try:
                    results = train_logistic_regression(df, independent_vars)

                    st.success("✅ Análise concluída!")

                    # --- Seção de Resultados ---
                    st.subheader("📋 Resultados do Modelo")

                    # Exibe a equação logística
                    coef_str = " + ".join([f"({results['coefficients'].loc[var, 'Coefficient']:.4f} × {var})" for var in independent_vars])
                    st.markdown("**Equação Logística (Log-Odds):**")
                    st.latex(f"\\text{{log-odds}}(\\text{{Vitória}}) = {coef_str}")
                    
                    st.markdown("**Equação de Probabilidade (Sigmoide):**")
                    st.latex(f"P(\\text{{Vitória}}) = \\frac{{1}}{{1 + e^{{-({coef_str})}}}}")

                    # Métricas principais
                    st.markdown("#### 🎯 Métricas de Performance")
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    
                    # Parse classification report
                    report_dict = classification_report(results['y_test'], results['y_pred'], output_dict=True, zero_division=0)
                    
                    col_metric1.metric("Acurácia", f"{results['accuracy']:.2%}", 
                                      help="Percentual de predições corretas")
                    col_metric2.metric("Precisão", f"{report_dict['1']['precision']:.2%}",
                                      help="Das vitórias previstas, quantas foram corretas")
                    col_metric3.metric("Recall", f"{report_dict['1']['recall']:.2%}",
                                      help="Das vitórias reais, quantas foram previstas")
                    col_metric4.metric("F1-Score", f"{report_dict['1']['f1-score']:.2%}",
                                      help="Média harmônica entre Precisão e Recall")

                    # Coeficientes com interpretação
                    st.markdown("#### 📊 Coeficientes do Modelo (Impacto no Log-Odds)")
                    coef_display = results['coefficients'].copy()
                    coef_display['Interpretação'] = coef_display['Coefficient'].apply(
                        lambda x: f"{'↑' if x > 0 else '↓'} {'Aumenta' if x > 0 else 'Diminui'} chance de vitória"
                    )
                    st.dataframe(coef_display.style.background_gradient(subset=['Coefficient'], cmap='RdYlGn'))
                    
                    st.info("""
                    **📖 Como interpretar:**
                    - **Coeficiente positivo (+)**: Aumenta a probabilidade de vitória
                    - **Coeficiente negativo (-)**: Diminui a probabilidade de vitória
                    - **Magnitude**: Quanto maior o valor absoluto, maior o impacto
                    """)

                    # Exemplo de predição
                    st.markdown("#### 🎲 Exemplo de Predição")
                    sample_idx = np.random.randint(0, len(results['y_test']))
                    sample_prob = results['y_pred_proba'][sample_idx]
                    sample_real = "Vitória" if results['y_test'].iloc[sample_idx] == 1 else "Derrota"
                    sample_pred = "Vitória" if sample_prob > 0.5 else "Derrota"
                    
                    st.success(f"""
                    **Jogo Exemplo #{sample_idx + 1}:**
                    - **Probabilidade de Vitória:** {sample_prob:.1%}
                    - **Predição:** {sample_pred}
                    - **Resultado Real:** {sample_real}
                    - **Status:** {'✅ Acertou!' if sample_pred == sample_real else '❌ Errou'}
                    """)

                    # Relatório de classificação completo
                    with st.expander("📋 Relatório de Classificação Detalhado"):
                        st.text(results['classification_report'])

                    # --- Seção de Gráficos ---
                    st.subheader("📈 Visualizações Gráficas")

                    # Layout em 2 colunas para os gráficos
                    col1, col2 = st.columns(2)

                    with col1:
                        # Gráfico 1: Curva ROC
                        st.markdown("#### 1. Curva ROC")
                        st.pyplot(plot_roc_curve(results['y_test'], results['y_pred_proba']))
                        st.caption("📊 A curva ROC mostra o trade-off entre taxa de verdadeiros positivos e falsos positivos. AUC próximo de 1.0 indica excelente performance.")

                    with col2:
                        # Gráfico 2: Distribuição de Probabilidades
                        st.markdown("#### 2. Distribuição de Probabilidades")
                        st.pyplot(plot_predicted_probabilities(results['y_pred_proba']))
                        st.caption("📊 Mostra como o modelo distribui as probabilidades de vitória nas predições.")

                    # Gráfico 3: Importância das Features (largura completa)
                    st.markdown("#### 3. Importância das Variáveis")
                    st.pyplot(plot_feature_importance(results['coefficients'], "Impacto das Variáveis na Probabilidade de Vitória"))
                    st.caption("📊 Coeficientes positivos (verde) aumentam a probabilidade de vitória; negativos (vermelho) a diminuem.")

                    col3, col4 = st.columns(2)

                    with col3:
                        # Gráfico 4: Curva Sigmoide
                        if len(independent_vars) > 0:
                            st.markdown("#### 4. Curva Logística (Sigmoide)")
                            st.pyplot(plot_logistic_regression_curve(df, independent_vars[0], 'WIN'))
                            st.caption(f"📊 Mostra como a probabilidade de vitória varia com {independent_vars[0]}.")

                    with col4:
                        # Gráfico 5: Matriz de Confusão
                        st.markdown("#### 5. Matriz de Confusão")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        disp = ConfusionMatrixDisplay(confusion_matrix=results['confusion_matrix'], 
                                                      display_labels=['Derrota', 'Vitória'])
                        disp.plot(ax=ax, cmap=plt.cm.Blues)
                        ax.set_title('Matriz de Confusão - Predição de Vitória/Derrota')
                        st.pyplot(fig)
                        st.caption("📊 Mostra quantas predições foram corretas (diagonal) e incorretas (fora da diagonal).")

                    # Gráfico 6: Tendência com Intervalo de Confiança
                    st.markdown("#### 6. Gráfico de Tendência com Intervalo de Confiança")
                    st.pyplot(plot_regression_confidence_interval(df, independent_vars, 'WIN'))
                    st.caption("📊 Visualiza a tendência entre a probabilidade de vitória e as variáveis independentes com intervalo de confiança de 95%.")

                except ValueError as ve:
                    st.error(f"❌ {str(ve)}")
                    
                    # Mostra sugestões específicas
                    with st.expander("💡 Sugestões para Resolver"):
                        st.markdown("""
                        **Possíveis soluções:**
                        
                        1. **Selecione mais jogadores**: 
                           - Aumenta a quantidade de dados
                           - Melhora a distribuição entre vitórias e derrotas
                        
                        2. **Verifique a distribuição dos dados**:
                           - Certifique-se de que há jogos com vitória E derrota
                           - Evite jogadores com poucos jogos registrados
                        
                        3. **Use o modo "Por Time"**:
                           - Analisa o time completo ao invés de jogadores individuais
                           - Geralmente tem mais dados disponíveis
                        
                        4. **Atualize os dados**:
                           - Execute `python fetch_data.py` para buscar dados mais recentes
                        """)
                    
                    # Diagnóstico detalhado
                    with st.expander("📊 Diagnóstico Detalhado"):
                        st.write(f"**Total de registros:** {len(df)}")
                        st.write(f"**Classes únicas em WIN:** {df['WIN'].nunique()}")
                        st.write(f"**Distribuição de WIN:**")
                        st.write(df['WIN'].value_counts())
                        
                        if analysis_mode == "🏀 Por Jogador":
                            st.write("**Estatísticas por jogador:**")
                            for player in selected_players:
                                player_data = df[df[name_column] == player]
                                wins = player_data['WIN'].sum()
                                losses = player_data['WIN'].count() - wins
                                st.write(f"- **{player}**: {len(player_data)} jogos ({wins} vitórias, {losses} derrotas)")
                
                except Exception as e:
                    st.error(f"❌ Erro inesperado ao treinar o modelo: {str(e)}")
                    import traceback
                    with st.expander("🔍 Detalhes técnicos do erro"):
                        st.code(traceback.format_exc())

# ============================================================================
# ABA 2: EXPLORAÇÃO DE DADOS
# ============================================================================
with tabs[1]:
    st.subheader("🔍 Exploração e Análise Exploratória de Dados")

    # Estatísticas por resultado
    st.markdown("#### 📊 Estatísticas por Resultado (Vitória/Derrota)")
    
    stats_by_result = df.groupby('WIN')[numeric_cols].mean()
    stats_by_result.index = ['Derrota', 'Derrota']
    stats_by_result.index = ['Derrota', 'Vitória']
    
    st.dataframe(stats_by_result.round(2).style.background_gradient(cmap='RdYlGn', axis=0))

    # Comparação visual
    st.markdown("#### 📈 Comparação de Estatísticas: Vitória vs Derrota")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_stat = st.selectbox(
            "Selecione uma estatística para comparar:",
            options=numeric_cols,
            index=numeric_cols.index('PTS') if 'PTS' in numeric_cols else 0
        )
    
    with col2:
        st.metric("Diferença Média", 
                 f"{stats_by_result.loc['Vitória', selected_stat] - stats_by_result.loc['Derrota', selected_stat]:.2f}",
                 delta=f"{((stats_by_result.loc['Vitória', selected_stat] / stats_by_result.loc['Derrota', selected_stat]) - 1) * 100:.1f}%")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df.copy()
    df_plot['Resultado'] = df_plot['WIN'].map({0: 'Derrota', 1: 'Vitória'})
    sns.boxplot(
        data=df_plot, 
        x='Resultado', 
        y=selected_stat, 
        ax=ax, 
        hue='Resultado',
        palette=['#d32f2f', '#388e3c'],
        legend=False
    )
    ax.set_title(f'Distribuição de {selected_stat} por Resultado', fontsize=14, fontweight='bold')
    ax.set_ylabel(selected_stat, fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Matriz de Correlação com WIN
    st.markdown("#### 🔗 Correlação das Variáveis com Vitória")
    
    correlations = df[numeric_cols + ['WIN']].corr()['WIN'].drop('WIN').sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(correlations) * 0.3)))
    colors = ['#388e3c' if x > 0 else '#d32f2f' for x in correlations.values]
    correlations.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Correlação das Variáveis com Vitória', fontsize=14, fontweight='bold')
    ax.set_xlabel('Correlação de Pearson', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig)
    
    st.caption("📊 Valores positivos indicam que maiores valores da variável estão associados a mais vitórias.")

# ============================================================================
# ABA 3: SIMULADOR DE PROBABILIDADE
# ============================================================================
with tabs[2]:
    st.subheader("🎯 Simulador de Probabilidade de Vitória")
    
    st.markdown("""
    Configure as estatísticas do time para **simular a probabilidade de vitória** em um jogo hipotético.
    """)
    
    # Treina um modelo completo para usar no simulador
    if len(numeric_cols) >= 3:
        sim_vars = st.multiselect(
            "Selecione as variáveis para o simulador:",
            options=numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
            key="sim_vars"
        )
        
        if sim_vars and st.button("🔧 Preparar Simulador", type="secondary"):
            with st.spinner("Treinando modelo para simulação..."):
                try:
                    sim_results = train_logistic_regression(df, sim_vars)
                    st.session_state['sim_model'] = sim_results['model']
                    st.session_state['sim_vars'] = sim_vars
                    st.success("✅ Simulador preparado!")
                except Exception as e:
                    st.error(f"Erro ao preparar simulador: {e}")
        
        if 'sim_model' in st.session_state and 'sim_vars' in st.session_state:
            st.markdown("#### 📝 Configure as Estatísticas do Jogo")
            
            # Cria inputs para cada variável
            sim_values = {}
            cols = st.columns(min(3, len(st.session_state['sim_vars'])))
            
            for idx, var in enumerate(st.session_state['sim_vars']):
                col_idx = idx % len(cols)
                with cols[col_idx]:
                    var_mean = df[var].mean()
                    var_std = df[var].std()
                    var_min = df[var].min()
                    var_max = df[var].max()
                    
                    sim_values[var] = st.number_input(
                        f"{var}",
                        min_value=float(var_min),
                        max_value=float(var_max),
                        value=float(var_mean),
                        step=float(var_std / 10),
                        help=f"Média: {var_mean:.2f} | Desvio: {var_std:.2f}"
                    )
            
            if st.button("🎲 Calcular Probabilidade", type="primary"):
                # Prepara dados para predição
                X_sim = pd.DataFrame([sim_values])
                
                # Faz a predição
                prob = st.session_state['sim_model'].predict_proba(X_sim)[0, 1]
                prediction = "Vitória" if prob > 0.5 else "Derrota"
                
                # Exibe resultado com destaque
                st.markdown("---")
                st.markdown("### 🎯 Resultado da Simulação")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Probabilidade de Vitória", f"{prob:.1%}")
                with col2:
                    st.metric("Probabilidade de Derrota", f"{(1-prob):.1%}")
                with col3:
                    st.metric("Predição", prediction, 
                             delta="Favorito" if prob > 0.6 else ("Equilibrado" if prob > 0.4 else "Azarão"))
                
                # Visualização de probabilidade
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh([0], [prob], color='#388e3c', height=0.5)
                ax.barh([0], [1-prob], left=[prob], color='#d32f2f', height=0.5)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_yticks([])
                ax.axvline(0.5, color='black', linestyle='--', linewidth=2)
                ax.set_title(f'Los Angeles Lakers têm {prob:.1%} de chance de vencer', fontsize=14, fontweight='bold')
                ax.text(prob/2, 0, f'Vitória\n{prob:.1%}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                ax.text(prob + (1-prob)/2, 0, f'Derrota\n{(1-prob):.1%}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                st.pyplot(fig)
                
                st.success(f"""
                **💬 Interpretação:**
                Com as estatísticas configuradas, o modelo prevê que os Lakers têm **{prob:.1%}** de chance de vencer.
                Isso significa que em 100 jogos com características similares, esperaríamos aproximadamente **{int(prob*100)} vitórias**.
                """)