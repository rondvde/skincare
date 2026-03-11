import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#### Setting working directory ####

# from google.colab import drive
# drive.mount('/content/drive')
# 
# drive_path = '/content/drive/MyDrive/Scuola/Uni/Tesi'
# os.chdir(drive_path)

#### Storing in a vector the names of the brand for better exploration ####

brand_names = list(['AESTURA', 'BEAUTYOFJOSEON', 'BELIF', 'COSRX', 'DRG', 
'ILLIYOON', 'ISNTREE', 'KLAIRS', 'LANEIGE',  'MIXSOON', 'PYUNKANGYUL', 'ROUNDLAB', 
'SKIN1004', 'SOMEBYME', 'SOONJUNG', 'SULWHASOO', 'TORRIDEN'])

#### Translate the Amazon reviews in english using Deepl API ####

import deepl

AUTH_KEY = API
translator = deepl.Translator(AUTH_KEY)

base_path = "data/00_raw"

# 2. function to translate
def translate_to_english_deepl(text):
    """translate the review in english, skip null and english reviews"""
    if pd.isna(text) or str(text).strip() == "":
        return text

    text_str = str(text)
    try:
        # target_lang="EN-US"
        result = translator.translate_text(text_str, target_lang="EN-US")
        return result.text
    except Exception as e:
        print(f" Error: {e}")
        return text_str

# 3. storing the translations

for brand in brand_names:
    file_path = os.path.join(base_path, brand, f"{brand}_Amazon.xlsx")

    if os.path.exists(file_path):
        print(f"File found in {brand}: {file_path}")

        try:
            df = pd.read_excel(file_path, engine='openpyxl')

            if not df.empty:
                col_name = df.columns[0]

                # Translating all the entries in the column
                df[col_name] = df[col_name].apply(translate_to_english_deepl)

                df.to_excel(file_path, index=False, engine='openpyxl')
            else:
                print(f"Empty {brand} file.\n")

        except Exception as e:
            print(f"Error on {brand}: {e}\n")
    else:
        print(f"File not found in {file_path}\n")

#### Unifying the reveiws ####

sources = ["Amazon", "Coupang"]
base_path = "data/00_raw"

# creating a function to import .xlsx data
def load_excel_brand(file_path, brand, source):
    try:
        df_tmp = pd.read_excel(file_path, engine='openpyxl')

        if not df_tmp.empty:
            #unifying the name of the column to 'review_text'
            df_tmp = df_tmp.rename(columns={df_tmp.columns[0]: 'review_text'})

            # making sure the imported data are only the reviews
            df_tmp = df_tmp[['review_text']]

            # adding metadata
            df_tmp['brand'] = brand
            df_tmp['source'] = source

            return df_tmp
    except Exception as e:
        print(f"Error with {file_path}: {e}")
    return None

all_dfs = []
for brand in brand_names:
    for source in ["Amazon", "Coupang"]:
        path = f"data/00_raw/{brand}/{brand}_{source}.xlsx"

        if os.path.exists(path):
            new_df = load_excel_brand(path, brand, source)
            if new_df is not None:
                all_dfs.append(new_df)

df = pd.concat(all_dfs, ignore_index=True)

# checking missing values
print("\nMissing values:")
print(df.isnull().sum())
df = df.dropna(subset=['review_text']).copy()

# dimensions
print(f"Total of reviews: {len(df)}")
print("\nDistribution for brand and source:")
print(df.groupby(['brand', 'source']).size().unstack())

#### Outputting dataset in interim folder ####

output_path = "data/01_interim/data.xlsx"

os.makedirs("data/01_interim", exist_ok=True)

df.to_excel(output_path, index=False, engine='openpyxl')

print(f"File saved in: {output_path}")

input_path = "data/01_interim/data.xlsx"
df = pd.read_excel(input_path, engine='openpyxl')

#### adding a unique ID to each review for better tracking during annotation ####
if 'ID' not in df.columns:
  df.insert(0, 'ID', range(1, len(df) + 1))
  df.to_excel(input_path, index=False, engine='openpyxl')

#### Creating the "Gold Standard" dataset ####
def sample_uniformly(df_source, target_n=340):
    brands = df_source['brand'].unique()
    n_brands = len(brands)

    base_n = target_n // n_brands
    remainder = target_n % n_brands

    sampled_dfs = []
    for i, brand in enumerate(brands):
        n_to_sample = base_n + 1 if i < remainder else base_n

        df_brand = df_source[df_source['brand'] == brand]

        sampled = df_brand.sample(n=n_to_sample, random_state=42)
        sampled_dfs.append(sampled)

    return pd.concat(sampled_dfs)

# Sampling 340 reviews from each source, ensuring a uniform distribution across brands
df_amazon = df[df['source'] == 'Amazon']
df_coupang = df[df['source'] == 'Coupang']

sample_amazon = sample_uniformly(df_amazon, target_n=340)
sample_coupang = sample_uniformly(df_coupang, target_n=340)

df_gold_standard = pd.concat([sample_amazon, sample_coupang]).reset_index(drop=True)

# Adding empty columns for annotation
aspects = ['sensoriality', 'performance', 'finish', 'safety', 'extrinsic',
           'mentions_ingredient', 'mentions_routine', 'mentions_makeup', 'unspecified_sentiment']
for aspect in aspects:
    df_gold_standard[aspect] = ""

# Reordering columns for better readability
cols = ['ID', 'brand', 'source', 'review_text'] + aspects
df_gold_standard = df_gold_standard[cols]

# Saving the gold standard dataset
output_path = "data/02_annotated/data_annotated.xlsx"
df_gold_standard.to_excel(output_path, index=False, engine='openpyxl')

print(f"Totale recensioni estratte: {len(df_gold_standard)}")
print("\nDistribuzione per mercato:")
print(df_gold_standard['source'].value_counts())

