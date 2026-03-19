from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
import pandas as pd
import numpy as np

data_annotated_file = "../data/02_annotated/data_annotated.xlsx"
test = pd.read_excel(data_annotated_file, engine='openpyxl')
df = pd.read_excel("../data/03_scored/data_scored.xlsx", engine='openpyxl')
interim = pd.read_excel("../data/01_interim/data.xlsx", engine='openpyxl')

data = pd.merge(df, test, on='ID', how='inner', suffixes=('_ai', '_gt'))

data = data.replace(to_replace=r'(?i)^nan$', value=np.nan, regex=True)

data = pd.merge(data, interim[['ID', 'source']], on='ID', how='left')

# Identify columns to convert
aspects = ['sensoriality', 'performance', 'finish', 'safety', 'extrinsic']
mentions = ['mentions_ingredient', 'mentions_routine', 'mentions_makeup', 'mentions_korea', 'unspecified_sentiment']

columns_to_fix = []
for col in aspects + mentions:
    if f'{col}_ai' in data.columns:
        columns_to_fix.append(f'{col}_ai')
    if f'{col}_gt' in data.columns:
        columns_to_fix.append(f'{col}_gt')

# Convert identified columns to numeric
for col in columns_to_fix:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 4. Verify the conversion
print("Data types after conversion:")
print(data[columns_to_fix].dtypes.head())
print("\nNull counts per column:")
print(data[columns_to_fix].isnull().sum())

data.head()

# Define the lists of categories to evaluate
aspects = ['sensoriality', 'performance', 'finish', 'safety', 'extrinsic']
mentions = ['mentions_ingredient', 'mentions_routine', 'mentions_makeup', 'mentions_korea', 'unspecified_sentiment']
all_categories = aspects + mentions

results = {}

print(data.columns)

print("--- Classification Performance Metrics ---\n")

for category in all_categories:
    ai_col = f'{category}_ai'
    gt_col = f'{category}_gt'

    if ai_col in data.columns and gt_col in data.columns:

        current_df = data

        if category == 'mentions_korea':
            current_df = data[data['source'] == 'Amazon']
            print(f"Calculated only on Amazon {category.upper()} (Analyzed records: {len(current_df)})")

        if len(current_df) == 0:
            print(f"Skipping {category}: No data found.\n")
            print("-" * 30 + "\n")
            continue

        y_true = current_df[gt_col].fillna(-99)
        y_pred = current_df[ai_col].fillna(-99)

        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0)

        results[category] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1,
            'report': report
        }

        print(f"Category: {category.upper()}")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Macro F1-Score:    {macro_f1:.4f}")
        print("\nClassification Report:")
        print(report)
        print("-" * 30 + "\n")
    else:
        print(f"Skipping {category}: Columns not found in data.\n")
