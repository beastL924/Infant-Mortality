import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, roc_auc_score

# Load dataset
data = pd.read_excel("data.xlsx")

# Remove unnecessary columns
drop_columns = [
    "se", "population", "flag", "setting_average",
    "update", "dataset_id", "source", "indicator_abbr"
]
clean_data = data.drop(columns=drop_columns)

# Filter West African countries
West_African_Countries = [
    "LBR", "NGA", "GHA", "SLE", "GIN", "SEN",
    "MLI", "NER", "BFA", "CIV", "BEN", "TGO",
    "GMB", "CPV"
]
selected_countries = clean_data[clean_data["iso3"].isin(West_African_Countries)]
selected_countries = selected_countries.dropna(subset=["estimate"])
selected_countries = selected_countries.sort_values(["iso3", "date"])

# Create lag features & rolling averages
selected_countries['lag_1'] = selected_countries.groupby('iso3')['estimate'].shift(1)
selected_countries['lag_5'] = selected_countries.groupby('iso3')['estimate'].shift(5)
selected_countries['roll_mean_5'] = selected_countries.groupby('iso3')['estimate'].rolling(5).mean().reset_index(0, drop=True)
selected_countries['roll_mean_10'] = selected_countries.groupby('iso3')['estimate'].rolling(10).mean().reset_index(0, drop=True)
selected_countries = selected_countries.dropna()

# Encode categorical columns
categorical_columns = ["setting", "indicator_name", "wbincome2025", "subgroup"]
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    selected_countries[col] = le.fit_transform(selected_countries[col])
    encoders[col] = le

# Regression features
new_feature_columns = [
    "date", "setting", "subgroup",
    "indicator_name", "wbincome2025","lag_1", "lag_5",
    "roll_mean_5", "roll_mean_10"
]

X = selected_countries[new_feature_columns]
y = selected_countries["estimate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
Rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
Rf.fit(X_train, y_train)
y_pred = Rf.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# Feature importance bar chart
importances = pd.Series(Rf.feature_importances_, index=new_feature_columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
importances.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Feature Importance & Mortality Prediction")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# Risk level classification
median_mortality = selected_countries["estimate"].median()
selected_countries["risk_level"] = (selected_countries["estimate"] > median_mortality).astype(int)

X_cls = selected_countries[new_feature_columns]
y_cls = selected_countries["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

print(classification_report(y_test, y_pred_class))
print("ROC-AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))


# Forecast next 3 years 
future_years = 3
future_risk_forecasts = []

for country in selected_countries['iso3'].unique():
    df_country = selected_countries[selected_countries['iso3'] == country].copy()
    last_known = df_country.iloc[-5:].copy()  # for lag_5
    
    for i in range(future_years):
        lag_1 = last_known['estimate'].iloc[-1]
        lag_5 = last_known['estimate'].iloc[-5]
        roll_mean_5 = last_known['estimate'].iloc[-5:].mean()
        roll_mean_10 = last_known['estimate'].iloc[-10:].mean() if len(last_known) >= 10 else roll_mean_5
        
        setting = last_known['setting'].iloc[-1]
        subgroup = last_known['subgroup'].iloc[-1]
        indicator_name = last_known['indicator_name'].iloc[-1]
        wbincome2025 = last_known['wbincome2025'].iloc[-1]
        date = last_known['date'].iloc[-1] + 1
        
        X_future = pd.DataFrame([{
            'date': date,
            'setting': setting,
            'subgroup': subgroup,
            'indicator_name': indicator_name,
            'wbincome2025': wbincome2025,
            'lag_1': lag_1,
            'lag_5': lag_5,
            'roll_mean_5': roll_mean_5,
            'roll_mean_10': roll_mean_10
        }])
        
        risk_pred = clf.predict(X_future)[0]
        future_risk_forecasts.append({
            'iso3': country,
            'year': date,
            'predicted_risk': 'High' if risk_pred == 1 else 'Low'
        })
        
        new_row = pd.DataFrame({
            'estimate': [lag_1],
            'date': [date],
            'setting': [setting],
            'subgroup': [subgroup],
            'indicator_name': [indicator_name],
            'wbincome2025': [wbincome2025]
        })
        last_known = pd.concat([last_known, new_row], ignore_index=True)


# Convert forecast list to DataFrame
future_risk_df = pd.DataFrame(future_risk_forecasts)

# Pivot for heatmap
pivot_df = future_risk_df.pivot(index="iso3", columns="year", values="predicted_risk")

# Map risks to numeric values for coloring
risk_map = {"Low": 0, "High": 1}
heatmap_data = pivot_df.replace(risk_map)

#Plot
plt.figure(figsize=(12,8))
sns.heatmap(
    heatmap_data,
    cmap=sns.color_palette(["green","red"]),
    annot=pivot_df, 
    fmt="",
    cbar=False,
    linewidths=0.5,
    linecolor="black"
)

plt.title("Predicted Risk Levels per Country (2024-2026)", fontsize=16, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Country")
plt.tight_layout()
plt.show()