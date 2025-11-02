"""
Script para carregar dados da NBA API e salvar em CSV.
Execute este script quando quiser atualizar os dados da aplicação.

Uso: python fetch_data.py
"""

from src.data_loader import get_team_id, get_team_game_logs, get_all_games_for_season, get_player_game_logs
from src.data_preprocessing import clean_data, feature_engineering, preprocess_player_data
from src.data_saver import save_team_data, save_all_games_data, save_player_data
import sys

def fetch_and_save_data(team_name: str = "Los Angeles Lakers", season: str = "2024-25"):
    """Busca dados da API e salva em CSV."""
    
    print(f"Iniciando coleta de dados para {team_name} - Temporada {season}...")
    
    # Obtém ID do time
    team_id = get_team_id(team_name)
    if team_id is None:
        print(f"Erro: Time '{team_name}' não encontrado.")
        return False
    
    print(f"Team ID: {team_id}")
    
    # Carrega dados brutos
    print("\n1. Carregando logs de jogos do time...")
    team_games_raw = get_team_game_logs(team_id, season)
    
    print("2. Carregando todos os jogos da temporada...")
    all_games_raw = get_all_games_for_season(season)
    
    if team_games_raw.empty or all_games_raw.empty:
        print("Erro: Falha ao carregar dados da API.")
        return False
    
    # Limpa dados
    print("\n3. Limpando dados...")
    print("Colunas recebidas em team_games_raw:", list(team_games_raw.columns))

    try:
        team_games_clean = clean_data(team_games_raw)
    except ValueError as e:
        print("Erro ao limpar dados:", e)
        # salva CSV de debug para inspeção manual
        try:
            team_games_raw.to_csv("debug_team_games_raw.csv", index=False)
            print("Arquivo debug_team_games_raw.csv salvo para inspeção.")
        except Exception as save_e:
            print("Falha ao salvar arquivo de debug:", save_e)
        raise
    
    all_games_clean = clean_data(all_games_raw)
    
    # Feature engineering
    print("4. Realizando engenharia de features...")
    team_games_processed = feature_engineering(team_games_clean, all_games_clean)
    
    # Carrega dados dos jogadores
    print("5. Carregando dados dos jogadores...")
    player_logs_raw = get_player_game_logs(team_id, season)
    
    if not player_logs_raw.empty:
        player_logs_processed = preprocess_player_data(player_logs_raw)
    else:
        print("Aviso: Não foi possível carregar dados dos jogadores.")
        player_logs_processed = player_logs_raw
    
    # Salva em CSV
    print("\n6. Salvando dados em CSV...")
    save_team_data(team_games_processed, team_name, season)
    save_all_games_data(all_games_clean, season)
    save_player_data(player_logs_processed, team_name, season)
    
    print(f"\n✓ Dados salvos com sucesso!")
    return True

if __name__ == "__main__":
    team_name = "Los Angeles Lakers"
    season = "2024-25"
    
    success = fetch_and_save_data(team_name, season)
    sys.exit(0 if success else 1)