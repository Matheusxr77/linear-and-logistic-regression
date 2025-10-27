import os
import pandas as pd

DATA_DIR = "data"

def ensure_data_dir():
    """Cria o diretório 'data' se não existir."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_team_data(df: pd.DataFrame, team_name: str, season: str):
    """Salva dados do time em CSV."""
    ensure_data_dir()
    filename = os.path.join(DATA_DIR, f"{team_name.replace(' ', '_')}_{season}_games.csv")
    df.to_csv(filename, index=False)
    print(f"Dados do time salvos em: {filename}")
    return filename

def save_all_games_data(df: pd.DataFrame, season: str):
    """Salva dados de todos os jogos em CSV."""
    ensure_data_dir()
    filename = os.path.join(DATA_DIR, f"all_games_{season}.csv")
    df.to_csv(filename, index=False)
    print(f"Todos os jogos salvos em: {filename}")
    return filename

def save_player_data(df: pd.DataFrame, team_name: str, season: str):
    """Salva dados dos jogadores em CSV."""
    ensure_data_dir()
    filename = os.path.join(DATA_DIR, f"{team_name.replace(' ', '_')}_{season}_players.csv")
    df.to_csv(filename, index=False)
    print(f"Dados dos jogadores salvos em: {filename}")
    return filename

def load_team_data(team_name: str, season: str) -> pd.DataFrame:
    """Carrega dados do time do CSV."""
    filename = os.path.join(DATA_DIR, f"{team_name.replace(' ', '_')}_{season}_games.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame()

def load_all_games_data(season: str) -> pd.DataFrame:
    """Carrega dados de todos os jogos do CSV."""
    filename = os.path.join(DATA_DIR, f"all_games_{season}.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame()

def load_player_data(team_name: str, season: str) -> pd.DataFrame:
    """Carrega dados dos jogadores do CSV."""
    filename = os.path.join(DATA_DIR, f"{team_name.replace(' ', '_')}_{season}_players.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame()

def data_exists(team_name: str, season: str) -> bool:
    """Verifica se os dados já estão salvos em CSV."""
    team_file = os.path.join(DATA_DIR, f"{team_name.replace(' ', '_')}_{season}_games.csv")
    all_games_file = os.path.join(DATA_DIR, f"all_games_{season}.csv")
    player_file = os.path.join(DATA_DIR, f"{team_name.replace(' ', '_')}_{season}_players.csv")
    return os.path.exists(team_file) and os.path.exists(all_games_file) and os.path.exists(player_file)
