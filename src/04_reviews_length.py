from google.colab import drive
import os
import pandas as pd
import numpy as np
import spacy
import re
from konlpy.tag import Okt
# drive.mount('/content/drive')
# drive_path = '/content/drive/MyDrive/Scuola/Uni/Tesi'
# os.chdir(drive_path)
# print(f"Current working directory set to: {os.getcwd()}")
pip install konlpy
python -m spacy download en_core_web_lg

# 1. Load the models
nlp_en = spacy.load("en_core_web_lg")
okt = Okt()

# 2. Load the datasets
df_data = pd.read_excel("data/01_interim/data.xlsx") 
df_scored = pd.read_excel("data/03_scored/data_scored.xlsx")

# 3. Define the morpheme counting function
def count_morphemes(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    
    if re.search(r'[가-힣]', text):
        # Count Korean morphemes
        return len([m for m, pos in okt.pos(text) if pos != 'Punctuation'])
    else:
        # Count English morphemes/tokens
        return len([token for token in nlp_en(text) if not token.is_punct])

# 4. Apply the function to the text column
print("Calculating morphemes...")
df_data['morpheme_count'] = df_data['review_text'].apply(count_morphemes)

# 5. Merge the new column into the scored dataset
df_final = pd.merge(
    df_scored, 
    df_data[['ID', 'morpheme_count']], 
    on='ID', 
    how='left'
)

# 6. Add aspect_count as a variable
aspect_columns = ['sensoriality', 'performance', 'finish', 'safety', 'extrinsic']
mentions_columns = ['mentions_ingredient', 'mentions_routine', 'mentions_makeup']

aspect_sum = df_final[aspect_columns].notna().sum(axis=1)

mentions_sum = (df_final[mentions_columns] == 1).sum(axis=1)

df_final['aspect_count'] = aspect_sum + mentions_sum

# 7. Add source and brand columns
df_final = pd.merge(
    df_final, 
    df_data[['ID', 'source', 'brand']], 
    on='ID', 
    how='left'
)

# 8. Save the updated dataset
output_path = "data/03_scored/data_scored_with_length.xlsx"
df_final.to_excel(output_path, index=False)

print(f"Done!")
