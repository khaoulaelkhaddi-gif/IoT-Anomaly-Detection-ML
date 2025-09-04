import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns


####################### Data Pre-Processing ################################################

# Load the dataset
data = pd.read_csv('full_private_dataset.csv', low_memory=False)

# Define target variable
target = 'label'

# Separate features and target
X = data.drop(columns=[target])
y = data[target]

# Split data into train and test sets 80% / 20% ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y
)

# Dropping irrelevant columns
columns_to_drop = ['ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'dns_query',
                   'ssl_subject', 'ssl_issuer', 'http_uri', 'http_referrer',
                   'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
                   'weird_name', 'weird_addl', 'weird_notice', 'type']
X_train = X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns])
X_test  = X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns])

# Handle missing values
print("\nHandling missing values...")
for dataset in [X_train, X_test]:
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            mode_vals = dataset[col].mode(dropna=True)
            fillv = mode_vals.iloc[0] if not mode_vals.empty else ""
            dataset[col] = dataset[col].fillna(fillv)
        else:
            dataset[col] = dataset[col].fillna(dataset[col].median())

# Encode categorical variables
print("\nEncoding categorical variables...")
categorical_cols = X_train.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
if len(categorical_cols) > 0:
    X_train.loc[:, categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test.loc[:,  categorical_cols]  = encoder.transform(X_test[categorical_cols])

# Handle outliers using IQR method
print("\nHandling outliers...")
num_cols = X_train.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    q1 = X_train[num_cols].quantile(0.25)
    q3 = X_train[num_cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    X_train.loc[:, num_cols] = X_train[num_cols].clip(lower=lower, upper=upper, axis=1)
    X_test.loc[:,  num_cols]  = X_test[num_cols].clip(lower=lower,  upper=upper,  axis=1)


# Remove near constant features
print("\nRemoving constant features...")
constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
print("Constant features to drop:", constant_cols)
X_train = X_train.drop(columns=constant_cols)
X_test = X_test.drop(columns=constant_cols)


# Remove duplicate rows
print("\nRemoving duplicate rows...")
before_train = len(pd.concat([X_train, y_train], axis=1))
before_test  = len(pd.concat([X_test,  y_test],  axis=1))
train_data = pd.concat([X_train, y_train], axis=1).drop_duplicates().reset_index(drop=True)
test_data  = pd.concat([X_test,  y_test],  axis=1).drop_duplicates().reset_index(drop=True)
print(f"Removed {before_train - len(train_data)} duplicates from train set")
print(f"Removed {before_test  - len(test_data)} duplicates from test set")
X_train, y_train = train_data.drop(columns=[target]), train_data[target]
X_test,  y_test  = test_data.drop(columns=[target]),  test_data[target]




########################### Feature Selection ###################################

# Normalise to non-negative for Chi-square

print("\nNormalising features for Chi-Squared...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns,  index=X_test.index)


# Chi-square feature scores

print("\nComputing Chi-square feature scores...")
selector = SelectKBest(score_func=chi2, k='all')
selector.fit(X_train_scaled, y_train)

# Convert potential NaNs to 0
scores = np.nan_to_num(selector.scores_, nan=0.0)

feature_importance_df = (
    pd.DataFrame({'Feature': X_train_scaled.columns, 'Importance': scores})
      .sort_values('Importance', ascending=False)
      .reset_index(drop=True)
)
feature_importance_df['Rank'] = feature_importance_df.index + 1
total = feature_importance_df['Importance'].sum()
if total > 0:
    feature_importance_df['Importance_%'] = feature_importance_df['Importance'] / total * 100.0
else:
    feature_importance_df['Importance_%'] = 0.0

print(f"\nTotal features ranked: {len(feature_importance_df)}")
print("\nTop 10 Features based on Chi-square score:")
print(feature_importance_df[['Rank', 'Feature', 'Importance', 'Importance_%']].head(10))


# Plot: Ranked bar chart
topN = min(30, len(feature_importance_df))
ranked_top = feature_importance_df.head(topN).copy()
ranked_top['Label'] = ranked_top.apply(lambda r: f"{int(r['Rank'])}. {r['Feature']}", axis=1)

plt.figure(figsize=(13, 10))
ax = sns.barplot(data=ranked_top, x='Importance_%', y='Label', orient='h')
for i, imp_pct in enumerate(ranked_top['Importance_%']):
    ax.text(float(imp_pct) + 0.1, i, f"{imp_pct:.2f}%", va='center')
plt.title('Chi-Square Feature Importance Ranking - Simulated Dataset')
plt.xlabel('Importance Weight (%)')
plt.ylabel('Ranked Features')
plt.tight_layout()
plt.savefig('chart_chi2_feature_importance_ranking_Private.png')
plt.close()


# Select Top-10 features
k = min(10, len(feature_importance_df))
selected_features = feature_importance_df['Feature'].head(k).tolist()
print(f"\nSelected top-{k} features:\n{selected_features}")

# Save cleaned dataset with only the Top-10 features
X_train_selected = X_train_scaled[selected_features]
X_test_selected  = X_test_scaled[selected_features]
train_cleaned = pd.concat([X_train_selected, y_train.reset_index(drop=True)], axis=1)
test_cleaned  = pd.concat([X_test_selected,  y_test.reset_index(drop=True)],  axis=1)

train_cleaned.to_csv('train_cleaned_chi_private.csv', index=False)
test_cleaned.to_csv('test_cleaned_chi_private.csv', index=False)


