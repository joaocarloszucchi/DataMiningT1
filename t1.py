import pandas as pd
from fuzzywuzzy import fuzz
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def load_data():
    df1 = pd.read_csv("_ASSOC_Truco_01.csv")
    df2 = pd.read_csv("_ASSOC_Truco_02.csv")

    return pd.concat([df1, df2], ignore_index=True)

def data_preprocessing(df):
    df = remove_rows_with_nan(df)
    df = format_pairs(df)
    df = remove_jogadores_column(df)
    df = merge_similar_names(df)
    df = add_wins_columns(df)

    return df

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

def add_wins_columns(dataframe):
    dataframe['Wins'] = (dataframe['Amigos'] > dataframe['Oponentes'])
    dataframe['GreatWins'] = (dataframe['Amigos'] > dataframe['Oponentes']) & (dataframe['Oponentes'] < 12)
    return dataframe

def create_association_rules(dataframe):
    frequent = apriori(dataframe, min_support=0.05, use_colnames=True)
    association_rules_df = association_rules(frequent, metric='lift', min_threshold=1.0)
    association_rules_df = association_rules_df.drop(columns=['antecedent support', 'consequent support', 'lift', 'leverage', 'conviction', 'zhangs_metric'])
    association_rules_df['support'] = association_rules_df['support'].round(3)
    association_rules_df['confidence'] = association_rules_df['confidence'].round(3)

    return association_rules_df


df = load_data()
df = data_preprocessing(df)

df_general = pd.get_dummies(df[['Player1', 'Player2', 'Wins', 'GreatWins']], prefix='', prefix_sep='')

assocGeneral = create_association_rules(df_general)

assoc1 = assocGeneral[
    assocGeneral['antecedents'].apply(lambda x: 'GreatWins' not in x and 'Wins' not in x) &
    assocGeneral['consequents'].apply(lambda x: len(x) == 1 and ('Wins' in x or 'GreatWins' in x))
]

assoc2 = assocGeneral[
    assocGeneral['antecedents'].apply(lambda x: 'GreatWins' not in x and 'Wins' not in x) &
    assocGeneral['consequents'].apply(lambda x: len(x) < 3 and ('Wins' in x and 'GreatWins' in x))
]

print("\nAssociation Rules:")

assoc = pd.concat([assoc1, assoc2], ignore_index=True)

print(assoc)

one_player_rules = assoc[assoc['antecedents'].apply(lambda x: ',' not in str(x))]
two_player_rules = assoc[assoc['antecedents'].apply(lambda x: ',' in str(x))]

print("One Player Rules:")
print(one_player_rules)

print("\nTwo Player Rules:")
print(two_player_rules)