import pandas as pd
from irnlm.utils import write

df = pd.read_excel('data/lexique_database_french.xlsx')
def filter_words(row):
    if ('ART' in row) or ('PRO' in row) or ('PRE' in row) or ('CON' in row):
        return True
    else:
        return False

words = df['cgramortho'].apply(filter_words)


path = 'data/french_function_words.txt'
for word in list(set(df['Word'][words].values)):
    write(path, word)

# Manually removed from the list
# puissamment, non



## For pronouns
def filter_words(row):
    if 'PRO' in row:
        return True
    else:
        return False

words = df['cgramortho'].apply(filter_words)


path = 'data/french_pronouns.txt'
for word in list(set(df['Word'][words].values)):
    write(path, word)

# Manually removed from the list
# to, tézigue, icelles, tézig, mézig, sézigue, icelle, mézigue, sécolle, con