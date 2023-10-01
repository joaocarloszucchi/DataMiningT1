import pandas as pd
from fuzzywuzzy import fuzz

def remove_rows_with_nan(dataframe):
    cleaned_df = dataframe.dropna(how='any')
    
    return cleaned_df

def remove_spaces_and_commas(name):
    return name.replace(' ', '').replace(',', '')

def format_pairs(dataframe):
    dataframe['Player1'] = ""
    dataframe['Player2'] = ""
    
    split_names = dataframe['Jogadore(a)s'].str.split(', ')
    dataframe['Player1'] = split_names.str[0].apply(remove_spaces_and_commas)
    dataframe['Player2'] = split_names.str[1].apply(remove_spaces_and_commas)
    
    return dataframe

def remove_jogadores_column(dataframe):
    dataframe = dataframe.drop(columns=['Jogadore(a)s'])

    return dataframe

def get_unique_players(dataframe):
    unique_players = pd.concat([dataframe['Player1'], dataframe['Player2']]).unique()

    return unique_players

def count_player_appearances(dataframe):
    all_names = dataframe[['Player1', 'Player2']].values.flatten()
    player_counts = pd.Series(all_names).value_counts()
    return player_counts

def merge_similar_names(dataframe, threshold=85):
    unique_players = pd.concat([dataframe['Player1'], dataframe['Player2']]).unique()
    
    merged_names = {}
    
    for name1 in unique_players:
        if name1 not in merged_names:
            merged_names[name1] = name1
        else:
            continue
        
        for name2 in unique_players:
            if name1 != name2 and fuzz.ratio(name1, name2) >= threshold:
                merged_names[name2] = name1
    
    dataframe['Player1'] = dataframe['Player1'].replace(merged_names)
    dataframe['Player2'] = dataframe['Player2'].replace(merged_names)
    
    return dataframe

df1 = pd.read_csv("_ASSOC_Truco_01.csv")
df2 = pd.read_csv("_ASSOC_Truco_02.csv")

df = pd.concat([df1, df2], ignore_index=True)

df = remove_rows_with_nan(df)
df = format_pairs(df)
df = remove_jogadores_column(df)
df = merge_similar_names(df)




