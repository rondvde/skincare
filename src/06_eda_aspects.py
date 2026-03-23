import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Mount Google Drive to access files
# from google.colab import drive
# drive.mount('/content/drive')
# drive_path = '/content/drive/MyDrive/Scuola/Uni/Tesi'
# 
# if not os.path.exists(drive_path):
#     print(f"Warning: The specified path does not exist: {drive_path}")
#     print("Please ensure your Google Drive is mounted and the path is correct.")
# else:
#     os.chdir(drive_path)
#     print(f"Current working directory set to: {os.getcwd()}")

base_path = "data/03_scored/data_scored_with_length.xlsx"
sample_path = "data/03_scored/sample_data_scored_with_length.xlsx"

if not os.path.exists(base_path) and os.path.exists():
    print(f"Warning: Using sample data. The original data path does not exist: {base_path}")
    data = pd.read_excel(sample_path)
else:
    if os.path.exists(base_path):
        data = pd.read_excel(base_path)
    else:
        print(f"Error: The original data path does not exist: {base_path}")
        data = None

#### Comparing Aspect Count Between Coupang and Amazon Customers ####

sns.set(style="whitegrid")

# Create a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(
    x="source",
    y="aspect_count",
    data=data,
    palette="Set2"
)

plt.title("Comparison of Aspect Count Between Coupang and Amazon Customers", fontsize=14)
plt.xlabel("Source", fontsize=12)
plt.ylabel("Aspect Count", fontsize=12)

plot_path = os.path.join('results', "aspect_count_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.close()

#### Comparing Morpheme Count Between Coupang and Amazon Customers ####

sns.set(style="whitegrid")

# Create a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(
    x="source",
    y="morpheme_count",
    data=data,
    palette="Set2",

)
plt.ylim(0, 600)

plt.title("Comparison of Morpheme Count Between Coupang and Amazon Customers", fontsize=14)
plt.xlabel("Source", fontsize=12)
plt.ylabel("Morpheme Count", fontsize=12)

plot_path = os.path.join('results', "morpheme_count_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.close()

#### NA analysis ####

variables = ['sensoriality', 'performance', 'finish', 'safety', 'extrinsic']

na_percentages = data.groupby('source')[variables].apply(lambda x: x.isna().mean() * 100).reset_index()

print(na_percentages)

observed_percentages = na_percentages.copy()
observed_percentages[variables] = 100 - na_percentages[variables]

print(observed_percentages)

# var list
aspects = ['sensoriality', 'performance', 'finish', 'safety', 'extrinsic']
mentions = ['mentions_ingredient', 'mentions_routine', 'mentions_makeup']
all_vars = aspects + mentions

# calculating percentages of observed values for each variable by source
rows = []
for source in data['source'].unique():
    source_df = data[data['source'] == source]
    total = len(source_df)
    
    row = {'source': source}
    for var in all_vars:
        if var in aspects:
            count = source_df[var].notna().sum()
        else:
            count = (source_df[var] == 1).sum()
            
        row[var] = (count / total) * 100
    rows.append(row)

observed_percentages = pd.DataFrame(rows)

# Melt the DataFrame for easier plotting with seaborn
observed_percentages_melted = observed_percentages.melt(
    id_vars='source', 
    var_name='variable', 
    value_name='observed_percentage'
)
# Remove 'mentions_' prefix for better readability in the plot
observed_percentages_melted['variable'] = observed_percentages_melted['variable'].str.replace('mentions_', '')

plt.figure(figsize=(12, 7))
sns.barplot(
    x='variable',
    y='observed_percentage',
    hue='source',
    data=observed_percentages_melted,
    palette='Set2'
)

plt.title('Percentage of Observed Values per Aspect/Mention by Source', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Observed Percentage (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.legend(title='Source')
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plot_path = os.path.join('results', 'observed_percentages_by_source.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

#### Sentiment Proportion Analysis ####
# List of aspects to analyze
variables = ['sensoriality', 'performance', 'finish', 'safety', 'extrinsic']

sentiment_data_list = []

for aspect in variables:
    # Filter data where the current aspect is not null
    filtered_df = data[['source', aspect]].dropna(subset=[aspect])
    
    # Group by source and aspect value, then count occurrences
    counts = filtered_df.groupby(['source', aspect]).size().reset_index(name='count')
    
    # Calculate percentages within each source
    source_totals = counts.groupby('source')['count'].transform('sum')
    counts['percentage'] = (counts['count'] / source_totals) * 100
    
    # Rename columns to a standard format for merging
    counts = counts.rename(columns={aspect: 'sentiment'})
    counts['aspect'] = aspect
    
    sentiment_data_list.append(counts)

sentiment_proportions = pd.concat(sentiment_data_list, ignore_index=True)

colors = {-1: '#e74c3c', 0: '#95a5a6', 1: '#2ecc71'}

# Create subplots for each aspect
fig, axes = plt.subplots(1, len(variables), figsize=(20, 6), sharey=True)

for i, aspect in enumerate(variables):
    ax = axes[i]
    aspect_df = sentiment_proportions[sentiment_proportions['aspect'] == aspect]
    
    # Pivot the DataFrame to have sources as index and sentiments as columns, filling missing values with 0
    pivot_df = aspect_df.pivot(index='source', columns='sentiment', values='percentage').fillna(0)
    
    for val in [-1, 0, 1]:
        if val not in pivot_df.columns:
            pivot_df[val] = 0
    pivot_df = pivot_df[[-1, 0, 1]]
    
    # Plotting
    pivot_df.plot(kind='bar', stacked=True, ax=ax, color=[colors[c] for c in pivot_df.columns], legend=(i == len(variables)-1))
    
    ax.set_title(aspect.capitalize(), fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylim(0, 100)
    if i == 0:
        ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=0)

plt.suptitle('Sentiment Proportions by Aspect and Source (%)', fontsize=16, y=1.05)
plt.tight_layout()

plot_path = os.path.join('results', 'sentiment_percentages_by_aspect_and_source.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

plt.close()

#### Aspect Count Analysis ####
# Calculate the percentage of reviews with aspect_count > 0 per source
percentages = (data['aspect_count'] > 0).groupby(data['source']).mean() * 100

result_df = percentages.reset_index(name='percentage_at_least_1_aspect')
print(result_df)

#### Unspecified Sentiment Analysis for Reviews with 0 Aspects ####

# Set the default cases (aspect_count > 0) to 99 to filter them out
data.loc[data['aspect_count'] > 0, 'unspecified_sentiment'] = 99
plot_data = data[data['unspecified_sentiment'] != 99].copy()

# Fill NaNs with a string so they are plotted as a distinct category
plot_data['unspecified_sentiment'] = plot_data['unspecified_sentiment'].fillna('NA')

# Calculate percentages per source
percentages = (
    plot_data.groupby('source')['unspecified_sentiment']
    .value_counts(normalize=True)
    .rename('percentage')
    .mul(100)
    .reset_index()
)

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=percentages,
    x='unspecified_sentiment',
    y='percentage',
    hue='source',
    palette='Set2',
    order=[-1.0, 0.0, 1.0, 'NA'] # Forces this specific order on the X-axis
)

plt.title('Unspecified Sentiment Distribution by Source (0 Aspects)', fontsize=16)
plt.xlabel('Sentiment (-1: Negative, 0: Neutral, 1: Positive, NA: No Sentiment)', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.tight_layout()

# Save the plot
os.makedirs('results', exist_ok=True)
plot_path = os.path.join('results', 'unspecified_sentiment_percentage.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

#### Sample Reviews with Unspecified Sentiment ####

n_samples = 5 

# Filter for 0 aspects and the specific sources
filtered_data = data[
    (data['source'].isin(['Amazon', 'Coupang'])) & 
    (data['unspecified_sentiment'].isna())
]

sampled_ids = (
    filtered_data.groupby('source')['ID'].sample(n=n_samples, random_state=24)
)

print(sampled_ids)

reviews = pd.read_excel("data/01_interim/data.xlsx")

sampled_reviews = reviews[reviews['ID'].isin(sampled_ids)]

with open('results/no_sentiment_sampled_reviews.txt', 'w', encoding='utf-8') as f:
    for index, row in sampled_reviews.iterrows():
        f.write(f"ID: {row['ID']} | Source: {row['source']} | Brand: {row['brand']}\n")
        f.write("-" * 50 + "\n")
        
        f.write(f"{row['review_text']}\n\n")

        f.write("=" * 80 + "\n\n")
