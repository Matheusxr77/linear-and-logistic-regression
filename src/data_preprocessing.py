import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e normaliza colunas de data no DataFrame.
    - Procura colunas comuns de data e converte para datetime.
    - Se encontrar colunas year/month/day monta a data.
    - Se não encontrar, tenta inferir a partir de colunas de texto.
    - Em caso de falha, lança ValueError com informação das colunas disponíveis.
    """
    if not isinstance(df, pd.DataFrame):
        # se não for DataFrame, delega ao pandas (ex.: Series ou string)
        return pd.to_datetime(df, errors='coerce')

    # cópia para evitar mutação inesperada
    df = df.copy()

    # 1) colunas comuns de data
    candidate_cols = ['GAME_DATE', 'GAME_DATE_EST', 'GAME_DATE_UTC', 'GAME_DATE_RAW', 'Date', 'date']
    for col in candidate_cols:
        if col in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df[col], errors='coerce')
            return df

    # 2) colunas separadas year/month/day (vários padrões)
    possible_sets = [
        ('year','month','day'),
        ('Year','Month','Day'),
        ('GAME_DATE_YEAR','GAME_DATE_MONTH','GAME_DATE_DAY'),
        ('YYYY','MM','DD')
    ]
    for ys in possible_sets:
        if all(c in df.columns for c in ys):
            df['GAME_DATE'] = pd.to_datetime(df[list(ys)].rename(columns={ys[0]:'year', ys[1]:'month', ys[2]:'day'}), errors='coerce')
            return df

    # 3) tentar inferir a partir de colunas de texto: pega a primeira que parseia uma fração razoável
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() >= max(1, int(len(df) * 0.1)):  # pelo menos 10% (ou 1) parseado
                df['PARSED_DATE'] = parsed
                return df

    # 4) nada encontrado — fornecer mensagem útil
    raise ValueError("Nenhuma coluna de data reconhecida no DataFrame. Colunas disponíveis: " + ", ".join(df.columns))

def feature_engineering(team_games_df: pd.DataFrame, all_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza engenharia de features entre os jogos do time e todos os jogos da temporada.
    Esta versão normaliza tipos de GAME_ID e TEAM_ID e itera por linhas para evitar
    comparações entre Series rotuladas de forma diferente.
    """
    # cópias para não mutar os originais
    team_games = team_games_df.copy()
    all_games = all_games_df.copy()

    # garantir colunas existentes
    if 'GAME_ID' not in team_games.columns or 'TEAM_ID' not in team_games.columns:
        raise ValueError("team_games_df deve conter colunas 'GAME_ID' e 'TEAM_ID'")
    if 'GAME_ID' not in all_games.columns or 'TEAM_ID' not in all_games.columns:
        raise ValueError("all_games_df deve conter colunas 'GAME_ID' e 'TEAM_ID'")

    # normalizar tipos: GAME_ID como string, TEAM_ID como inteiro
    team_games['GAME_ID'] = team_games['GAME_ID'].astype(str)
    all_games['GAME_ID'] = all_games['GAME_ID'].astype(str)

    # TEAM_ID pode vir como float/object, converter para inteiro seguro
    team_games['TEAM_ID'] = pd.to_numeric(team_games['TEAM_ID'], errors='coerce').fillna(-1).astype(int)
    all_games['TEAM_ID'] = pd.to_numeric(all_games['TEAM_ID'], errors='coerce').fillna(-1).astype(int)

    # estrutura para resultados (exemplo: adiciona colunas de opponent_PTS)
    results = []

    # iterar por jogos do time
    for _, row in team_games.iterrows():
        game_id = str(row['GAME_ID'])
        team_id = int(row['TEAM_ID'])

        # buscar linha(s) do mesmo jogo em all_games e pegar o adversário
        opponent_rows = all_games[(all_games['GAME_ID'] == game_id) & (all_games['TEAM_ID'] != team_id)]

        if opponent_rows.empty:
            # registro de debug leve e continuar
            print(f"Atenção: adversário não encontrado para GAME_ID={game_id}, TEAM_ID={team_id}")
            continue

        # se houver múltiplas linhas, pega a primeira (normalmente deve haver 1)
        opponent = opponent_rows.iloc[0]

        # exemplo de extração de features: diferença de pontos (ajuste conforme seu pipeline)
        # ... adaptar conforme necessidade real do projeto ...
        try:
            team_pts = float(row.get('PTS', 0))
        except Exception:
            team_pts = 0.0
        try:
            opp_pts = float(opponent.get('PTS', 0))
        except Exception:
            opp_pts = 0.0

        feature_row = row.to_dict()
        feature_row['OPP_PTS'] = opp_pts
        feature_row['PTS_DIFF'] = team_pts - opp_pts

        results.append(feature_row)

    if not results:
        return pd.DataFrame()

    processed_df = pd.DataFrame(results)
    return processed_df

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