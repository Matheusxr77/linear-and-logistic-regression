import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, commonteamroster, playergamelog

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

def get_player_game_logs(team_id: int, season: str) -> pd.DataFrame:
    """
    Obtém os logs de jogos de todos os jogadores de um time em uma temporada.

    Args:
        team_id (int): O ID do time.
        season (str): A temporada no formato 'YYYY-YY'.

    Returns:
        pd.DataFrame: DataFrame com os logs de jogos de todos os jogadores.
    """
    print(f"Buscando elenco do time ID {team_id}...")
    try:
        roster_endpoint = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season
        )
        time.sleep(0.6)
        roster_dfs = _get_data_frames_with_retry(roster_endpoint, max_retries=5, backoff_factor=1)
        if not roster_dfs:
            print("Não foi possível obter o elenco.")
            return pd.DataFrame()
        roster_df = roster_dfs[0]
    except Exception as e:
        print(f"Erro ao obter o elenco: {e}")
        return pd.DataFrame()

    all_player_logs = []
    # extrai corretamente os IDs dos jogadores
    player_ids = roster_df['PLAYER_ID'].unique()
    print(f"Encontrados {len(player_ids)} jogadores. Buscando logs de jogos individuais...")

    for player_id in player_ids:
        try:
            player_endpoint = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            player_dfs = _get_data_frames_with_retry(player_endpoint, max_retries=4, backoff_factor=0.8)
            if not player_dfs:
                print(f"Sem logs para o jogador ID {player_id}.")
                continue
            # player_dfs geralmente retorna uma lista; pegamos o primeiro DataFrame
            player_log_df = player_dfs[0]
            all_player_logs.append(player_log_df)
            print(f"Logs obtidos para o jogador ID {player_id}.")
            time.sleep(0.6)
        except Exception as e:
            print(f"Erro ao buscar logs para o jogador ID {player_id}: {e}")
            continue
            
    if not all_player_logs:
        return pd.DataFrame()

    combined_logs = pd.concat(all_player_logs, ignore_index=True)
    print("Logs de todos os jogadores combinados com sucesso.")
    return combined_logs