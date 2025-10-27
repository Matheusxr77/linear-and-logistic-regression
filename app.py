# app.py

import streamlit as st
import pandas as pd
from src.data_preprocessing import clean_data, feature_engineering, preprocess_player_data
from src.data_saver import load_team_data, load_all_games_data, load_player_data, data_exists

# Configuração da página principal do Streamlit
st.set_page_config(
    page_title="Análise Preditiva NBA - LA Lakers",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cache de Dados ---
# O decorator @st.cache_data garante que os dados sejam carregados da API apenas uma vez
# e armazenados em cache para execuções subsequentes, acelerando a aplicação.

@st.cache_data
def load_and_preprocess_data(team_name, season):
    """
    Função completa para carregar e pré-processar todos os dados necessários.
    Encapsula a lógica de carregamento e processamento para ser cacheada.
    """
    team_id = get_team_id(team_name)
    if team_id is None:
        st.error(f"Time '{team_name}' não encontrado. Verifique o nome.")
        return None, None

    # Carrega os logs de jogos dos Lakers e de toda a liga
    lakers_games_raw = get_team_game_logs(team_id, season)
    all_games_raw = get_all_games_for_season(season)

    if lakers_games_raw.empty or all_games_raw.empty:
        st.error("Falha ao carregar os dados dos jogos da API. Tente novamente mais tarde.")
        return None, None

    # Limpa os dados
    lakers_games_clean = clean_data(lakers_games_raw)
    all_games_clean = clean_data(all_games_raw)

    # Realiza a engenharia de features para criar o dataset mestre
    processed_team_data = feature_engineering(lakers_games_clean, all_games_clean)

    # Carrega e processa dados dos jogadores
    player_logs_raw = get_player_game_logs(team_id, season)
    if not player_logs_raw.empty:
        processed_player_data = preprocess_player_data(player_logs_raw)
    else:
        st.warning("Não foi possível carregar os dados dos jogadores.")
        processed_player_data = pd.DataFrame()

    return processed_team_data, processed_player_data

@st.cache_data
def load_preprocessed_data(team_name, season):
    """
    Carrega dados pré-processados dos arquivos CSV.
    """
    if not data_exists(team_name, season):
        st.error(
            f"Dados não encontrados. Execute o script 'fetch_data.py' para carregar os dados da API:\n"
            f"```\npython fetch_data.py\n```"
        )
        return None, None

    # Carrega dos CSVs
    team_data = load_team_data(team_name, season)
    player_data = load_player_data(team_name, season)
    
    if team_data.empty:
        st.error("Falha ao carregar os dados do time.")
        return None, None

    return team_data, player_data

# --- Interface Principal ---

st.title("🏀 Análise Preditiva de Desempenho na NBA")
st.header("Los Angeles Lakers - Temporada 2024-25")

st.markdown("""
Esta aplicação implementa modelos de **Regressão Linear** e **Regressão Logística** para analisar e prever o desempenho do time Los Angeles Lakers, conforme especificado na Atividade 1 da disciplina de Redes Neurais.

**Navegue pelas análises usando o menu na barra lateral à esquerda.**

### Sobre o Projeto
O projeto utiliza a biblioteca `nba_api` para coletar dados da temporada 2024-25 e `scikit-learn` para a modelagem. A interface foi construída com Streamlit para permitir uma exploração interativa dos modelos.

- **Análise de Regressão Linear:** Permite prever estatísticas numéricas (como pontos, rebotes, assistências) com base em outras variáveis do jogo.
- **Análise de Regressão Logística:** Permite prever a probabilidade de vitória ou derrota em um jogo.
""")

# Carrega os dados do CSV
if 'team_data' not in st.session_state or 'player_data' not in st.session_state:
    with st.spinner("Carregando dados pré-processados..."):
        team_data, player_data = load_preprocessed_data(team_name="Los Angeles Lakers", season="2024-25")
        if team_data is not None:
            st.session_state['team_data'] = team_data
            st.session_state['player_data'] = player_data
            st.success("Dados carregados com sucesso!")
            st.write("Pré-visualização dos dados do time:")
            st.dataframe(team_data.head())
        else:
            st.error("Não foi possível carregar os dados. A aplicação não pode continuar.")