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
    
    # Limpa dados do time
    print("\n3. Limpando dados do time...")
    print(f"   Colunas recebidas em team_games_raw: {list(team_games_raw.columns)}")
    print(f"   Total de jogos do time: {len(team_games_raw)}")

    try:
        team_games_clean = clean_data(team_games_raw)
        print(f"   ✓ Dados do time limpos: {len(team_games_clean)} registros")
    except ValueError as e:
        print(f"   ✗ Erro ao limpar dados: {e}")
        try:
            team_games_raw.to_csv("debug_team_games_raw.csv", index=False)
            print("   Arquivo debug_team_games_raw.csv salvo para inspeção.")
        except Exception as save_e:
            print(f"   Falha ao salvar arquivo de debug: {save_e}")
        raise
    
    all_games_clean = clean_data(all_games_raw)
    
    # Feature engineering
    print("\n4. Realizando engenharia de features...")
    team_games_processed = feature_engineering(team_games_clean, all_games_clean)
    print(f"   ✓ Features criadas: {len(team_games_processed)} registros")
    
    # Carrega dados dos jogadores
    print("\n5. Carregando dados dos jogadores...")
    print("   ATENÇÃO: Este processo pode levar vários minutos devido ao rate limiting da API...")
    player_logs_raw = get_player_game_logs(team_id, season)
    
    if not player_logs_raw.empty:
        print(f"   ✓ Dados brutos carregados: {len(player_logs_raw)} registros")
        print(f"   Colunas disponíveis: {list(player_logs_raw.columns)}")
        
        # Verifica se PLAYER_NAME foi adicionado
        if 'PLAYER_NAME' in player_logs_raw.columns:
            unique_players = player_logs_raw['PLAYER_NAME'].nunique()
            print(f"   Jogadores únicos: {unique_players}")
            print(f"   Exemplos: {list(player_logs_raw['PLAYER_NAME'].unique()[:5])}")
        else:
            print(f"   ⚠️ Coluna PLAYER_NAME não foi criada")
        
        print("\n6. Processando dados dos jogadores...")
        player_logs_processed = preprocess_player_data(player_logs_raw)
        
        if player_logs_processed.empty:
            print("   ⚠️ ATENÇÃO: DataFrame de jogadores ficou vazio após processamento!")
            print("   Salvando dados brutos para análise...")
            player_logs_raw.to_csv("debug_player_logs_raw.csv", index=False)
            player_logs_processed = player_logs_raw
        else:
            print(f"   ✓ Dados processados: {len(player_logs_processed)} registros")
    else:
        print("   ⚠️ Aviso: Não foi possível carregar dados dos jogadores.")
        player_logs_processed = player_logs_raw
    
    # Salva em CSV
    print("\n7. Salvando dados em CSV...")
    
    save_team_data(team_games_processed, team_name, season)
    print(f"   ✓ Dados do time salvos: {len(team_games_processed)} registros")
    
    save_all_games_data(all_games_clean, season)
    print(f"   ✓ Todos os jogos salvos: {len(all_games_clean)} registros")
    
    if not player_logs_processed.empty:
        save_player_data(player_logs_processed, team_name, season)
        print(f"   ✓ Dados de jogadores salvos: {len(player_logs_processed)} registros")
        
        import os
        from src.data_saver import get_data_path
        player_file = get_data_path(f"{team_name.replace(' ', '_')}_{season}_players.csv")
        if os.path.exists(player_file):
            file_size = os.path.getsize(player_file)
            print(f"   ✓ Arquivo criado: {player_file} ({file_size} bytes)")
    else:
        print("   ⚠️ Nenhum dado de jogador para salvar")
    
    print(f"\n{'='*60}")
    print(f"✓ PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}")
    print(f"Resumo:")
    print(f"  - Jogos do time: {len(team_games_processed)}")
    print(f"  - Todos os jogos: {len(all_games_clean)}")
    print(f"  - Logs de jogadores: {len(player_logs_processed)}")
    
    return True

if __name__ == "__main__":
    team_name = "Los Angeles Lakers"
    season = "2024-25"
    
    try:
        success = fetch_and_save_data(team_name, season)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERRO FATAL: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)