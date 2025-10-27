# src/data_preprocessing.py

import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e converte os tipos de dados do DataFrame de jogos.

    Args:
        df (pd.DataFrame): DataFrame bruto de logs de jogos.

    Returns:
        pd.DataFrame: DataFrame com tipos de dados corrigidos.
    """
    # Lista de colunas que devem ser numéricas
    numeric_cols = []
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Converte a data do jogo para o formato datetime
    df = pd.to_datetime(df)
    
    # Ordena os jogos por data para cálculos de médias móveis
    df = df.sort_values(by='GAME_DATE').reset_index(drop=True)
    
    return df

def feature_engineering(lakers_games_df: pd.DataFrame, all_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a engenharia de features, incluindo estatísticas do oponente e médias móveis.

    Args:
        lakers_games_df (pd.DataFrame): DataFrame com jogos apenas dos Lakers.
        all_games_df (pd.DataFrame): DataFrame com todos os jogos da temporada.

    Returns:
        pd.DataFrame: DataFrame enriquecido com novas features.
    """
    # 1. Adicionar estatísticas do oponente
    opponent_stats = []
    for index, row in lakers_games_df.iterrows():
        game_id = row
        team_id = row
        
        # Encontra a linha do oponente para o mesmo jogo
        opponent_row = all_games_df[(all_games_df['GAME_ID'] == game_id) & (all_games_df['TEAM_ID'] != team_id)]
        
        if not opponent_row.empty:
            # Pega a primeira linha (deve haver apenas uma) e renomeia as colunas
            opp_stats = opponent_row.iloc.add_prefix('OPP_')
            opponent_stats.append(opp_stats)
        else:
            # Caso não encontre oponente, preenche com NaNs
            opponent_stats.append(pd.Series(dtype='object').add_prefix('OPP_'))

    opponent_df = pd.DataFrame(opponent_stats).reset_index(drop=True)
    
    # Concatena as estatísticas do time com as do oponente
    df = pd.concat([df, opponent_df], axis=1)

    # 2. Criar features básicas
    df = df.apply(lambda x: 1 if x == 'W' else 0)
    df['HOME_GAME'] = df.apply(lambda x: 0 if '@' in x else 1)
    
    # 3. Criar médias móveis (excluindo o jogo atual)
    stats_to_average = []
    opp_stats_to_average = []

    for stat in stats_to_average:
        # shift(1) para que a média não inclua o jogo atual
        df[f'AVG_5G_{stat}'] = df[stat].shift(1).rolling(window=5, min_periods=1).mean()
        df[f'AVG_10G_{stat}'] = df[stat].shift(1).rolling(window=10, min_periods=1).mean()

    for stat in opp_stats_to_average:
        if stat in df.columns:
            df[f'AVG_5G_{stat}'] = df[stat].shift(1).rolling(window=5, min_periods=1).mean()
            df[f'AVG_10G_{stat}'] = df[stat].shift(1).rolling(window=10, min_periods=1).mean()

    # Remover linhas com valores NaN que podem ter sido gerados
    df = df.dropna(subset=[f'AVG_5G_{s}' for s in stats_to_average])
    
    return df.reset_index(drop=True)

def preprocess_player_data(player_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e prepara os dados de logs de jogos dos jogadores.

    Args:
        player_logs_df (pd.DataFrame): DataFrame bruto de logs de jogadores.

    Returns:
        pd.DataFrame: DataFrame limpo e pronto para análise.
    """
    # Lista de colunas que devem ser numéricas
    numeric_cols = []
    
    for col in numeric_cols:
        if col in player_logs_df.columns:
            player_logs_df[col] = pd.to_numeric(player_logs_df[col], errors='coerce')
            
    # Converte e ordena pela data do jogo se a coluna existir
    if 'GAME_DATE' in player_logs_df.columns:
        player_logs_df['GAME_DATE'] = pd.to_datetime(player_logs_df['GAME_DATE'], errors='coerce')
        player_logs_df = player_logs_df.sort_values(by='GAME_DATE').reset_index(drop=True)
    else:
        player_logs_df = player_logs_df.reset_index(drop=True)

    # Criar médias móveis para jogadores
    for stat in numeric_cols:
        if stat in player_logs_df.columns:
            player_logs_df[f'AVG_5G_{stat}'] = player_logs_df.groupby('PLAYER_ID')[stat].shift(1).rolling(window=5, min_periods=1).mean()

    # Só aplicar dropna se houver colunas na lista numeric_cols
    if numeric_cols:
        return player_logs_df.dropna(subset=numeric_cols)
    return player_logs_df