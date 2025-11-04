from nba_api.stats.endpoints import teamgamelog, leaguegamefinder, playergamelog, commonplayerinfo
from nba_api.stats.static import teams
import pandas as pd
import time

# Adicionar um cabeçalho de User-Agent é uma boa prática para evitar problemas de conexão.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# novo helper para retry ao chamar .get_data_frames()
def _get_data_frames_with_retry(endpoint, max_retries: int = 5, backoff_factor: float = 1.0):
    """
    Tenta endpoint.get_data_frames() várias vezes com backoff exponencial.
    Retorna a lista de DataFrames em caso de sucesso ou [] em caso de falha.
    """
    for attempt in range(1, max_retries + 1):
        try:
            dfs = endpoint.get_data_frames()
            return dfs or []
        except Exception as e:
            print(f"Tentativa {attempt} falhou: {e}")
            if attempt == max_retries:
                print("Máximo de tentativas atingido.")
                return []
            time.sleep(backoff_factor * attempt)

def get_team_id(team_name: str) -> int:
    """
    Obtém o ID de um time da NBA pelo seu nome completo.

    Args:
        team_name (str): O nome completo do time (ex: 'Los Angeles Lakers').

    Returns:
        int: O ID do time.
    """
    try:
        team_list = teams.find_teams_by_full_name(team_name)
        if not team_list:
            print(f"Erro: Time '{team_name}' não encontrado.")
            return None
        return team_list[0]['id']
    except (IndexError, KeyError, TypeError) as e:
        print(f"Erro ao buscar o time '{team_name}': {e}")
        return None

def get_team_game_logs(team_id: int, season: str) -> pd.DataFrame:
    """
    Obtém os logs de todos os jogos de um time em uma determinada temporada.
    Utiliza LeagueGameFinder para maior robustez.

    Args:
        team_id (int): O ID do time.
        season (str): A temporada no formato 'YYYY-YY' (ex: '2024-25').

    Returns:
        pd.DataFrame: DataFrame com os logs de jogos do time.
    """
    print(f"Buscando logs de jogos para o time ID {team_id} na temporada {season}...")
    try:
        game_finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        time.sleep(0.6)  # Pausa para evitar sobrecarregar a API
        games_dfs = _get_data_frames_with_retry(game_finder, max_retries=5, backoff_factor=1)
        if not games_dfs:
            return pd.DataFrame()
        games_df = games_dfs[0]
        print("Logs de jogos do time obtidos com sucesso.")
        return games_df
    except Exception as e:
        print(f"Erro ao buscar logs de jogos do time: {e}")
        return pd.DataFrame()

def get_all_games_for_season(season: str) -> pd.DataFrame:
    """
    Obtém os logs de TODOS os jogos da liga em uma temporada.
    Isso é necessário para extrair estatísticas dos oponentes.

    Args:
        season (str): A temporada no formato 'YYYY-YY' (ex: '2024-25').

    Returns:
        pd.DataFrame: DataFrame com os logs de todos os jogos da temporada.
    """
    print(f"Buscando todos os jogos da temporada {season} para análise de oponentes...")
    try:
        game_finder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='00', # '00' para NBA
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        time.sleep(0.6)
        all_games_dfs = _get_data_frames_with_retry(game_finder, max_retries=5, backoff_factor=1)
        if not all_games_dfs:
            return pd.DataFrame()
        all_games_df = all_games_dfs[0]
        print("Todos os jogos da temporada obtidos com sucesso.")
        return all_games_df
    except Exception as e:
        print(f"Erro ao buscar todos os jogos da temporada: {e}")
        return pd.DataFrame()

def get_player_info(player_id: int) -> dict:
    """Busca informações do jogador incluindo o nome."""
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = player_info.get_data_frames()[0]
        if not df.empty:
            return {
                'PLAYER_ID': player_id,
                'PLAYER_NAME': df['DISPLAY_FIRST_LAST'].iloc[0] if 'DISPLAY_FIRST_LAST' in df.columns else None,
                'FIRST_NAME': df['FIRST_NAME'].iloc[0] if 'FIRST_NAME' in df.columns else None,
                'LAST_NAME': df['LAST_NAME'].iloc[0] if 'LAST_NAME' in df.columns else None
            }
    except Exception as e:
        print(f"Erro ao buscar info do jogador {player_id}: {e}")
    return None

def get_player_game_logs(team_id: int, season: str = "2024-25") -> pd.DataFrame:
    """
    Carrega logs de jogos de todos os jogadores do time.
    Adiciona o nome do jogador usando commonplayerinfo.
    """
    # Primeiro, pega os logs do time para identificar os jogadores
    team_games = get_team_game_logs(team_id, season)
    
    if team_games.empty:
        return pd.DataFrame()
    
    # Usa leaguegamefinder para pegar dados de todos os jogadores do time
    print(f"   Buscando jogadores do time {team_id}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        player_or_team_abbreviation='P'  # 'P' para jogadores
    )
    
    all_player_games = gamefinder.get_data_frames()[0]
    
    if all_player_games.empty:
        print("   Nenhum dado de jogador encontrado via LeagueGameFinder")
        return pd.DataFrame()
    
    # Identifica jogadores únicos
    unique_players = all_player_games['PLAYER_ID'].unique()
    print(f"   {len(unique_players)} jogadores encontrados")
    
    # Busca informações de cada jogador (com rate limiting)
    player_names = {}
    for idx, player_id in enumerate(unique_players):
        if idx > 0 and idx % 10 == 0:
            print(f"   Processando jogador {idx}/{len(unique_players)}...")
            time.sleep(1)  # Rate limiting
        
        info = get_player_info(player_id)
        if info and info['PLAYER_NAME']:
            player_names[player_id] = info['PLAYER_NAME']
        else:
            player_names[player_id] = f"Player_{player_id}"
        
        time.sleep(0.6)  # Rate limiting entre chamadas
    
    # Adiciona nomes aos dados
    all_player_games['PLAYER_NAME'] = all_player_games['PLAYER_ID'].map(player_names)
    
    print(f"   Logs de todos os jogadores combinados com sucesso.")
    return all_player_games