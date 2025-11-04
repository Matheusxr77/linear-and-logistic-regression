import os
import pandas as pd

def get_data_path(filename: str) -> str:
    """
    Retorna o caminho completo para um arquivo na pasta data.
    Cria a pasta se não existir.
    
    Args:
        filename (str): Nome do arquivo
        
    Returns:
        str: Caminho completo do arquivo
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)

def save_team_data(df: pd.DataFrame, team_name: str, season: str):
    """Salva dados do time em CSV."""
    filename = f"{team_name.replace(' ', '_')}_{season}_games.csv"
    filepath = get_data_path(filename)
    df.to_csv(filepath, index=False)
    print(f"Dados do time salvos em: {filepath}")

def save_all_games_data(df: pd.DataFrame, season: str):
    """Salva todos os jogos da temporada em CSV."""
    filename = f"All_Games_{season}.csv"
    filepath = get_data_path(filename)
    df.to_csv(filepath, index=False)
    print(f"Todos os jogos salvos em: {filepath}")

def save_player_data(df: pd.DataFrame, team_name: str, season: str):
    """Salva dados dos jogadores em CSV."""
    filename = f"{team_name.replace(' ', '_')}_{season}_players.csv"
    filepath = get_data_path(filename)
    df.to_csv(filepath, index=False)
    print(f"Dados dos jogadores salvos em: {filepath}")

def load_team_data(team_name: str, season: str) -> pd.DataFrame:
    """Carrega dados do time de CSV."""
    filename = f"{team_name.replace(' ', '_')}_{season}_games.csv"
    filepath = get_data_path(filename)
    
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    return pd.read_csv(filepath)

def load_all_games_data(season: str) -> pd.DataFrame:
    """Carrega todos os jogos da temporada de CSV."""
    filename = f"All_Games_{season}.csv"
    filepath = get_data_path(filename)
    
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    return pd.read_csv(filepath)

def load_player_data(team_name: str, season: str) -> pd.DataFrame:
    """Carrega dados dos jogadores de CSV."""
    filename = f"{team_name.replace(' ', '_')}_{season}_players.csv"
    filepath = get_data_path(filename)
    
    if not os.path.exists(filepath):
        print(f"Aviso: Arquivo de jogadores não encontrado em {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    
    # Identifica coluna de nome do jogador
    name_column = None
    for col in ['PLAYER_NAME', 'Player_Name', 'PLAYER', 'Player']:
        if col in df.columns:
            name_column = col
            break
    
    if name_column:
        print(f"✓ Dados de jogadores carregados: {len(df)} registros, {df[name_column].nunique()} jogadores")
    else:
        print(f"✓ Dados de jogadores carregados: {len(df)} registros (sem coluna de nome identificada)")
    
    return df

def data_exists(team_name: str, season: str) -> bool:
    """Verifica se os dados existem."""
    team_file = get_data_path(f"{team_name.replace(' ', '_')}_{season}_games.csv")
    return os.path.exists(team_file)
