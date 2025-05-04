import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Ler os ficheiros ===
crimes = pd.read_csv(r"C:\Mestrado\1.2_ADE\ade_2425\originais\crimes_com_nivel.csv", sep=",")
estado_civil = pd.read_csv(r"C:\Mestrado\1.2_ADE\ade_2425\originais\estado_civil.csv", sep=";")

# === 2. Pré-processamento básico ===

# Crimes: total de crimes por Dicofre
crimes['dicofre'] = crimes['dicofre'].astype(str).str.split('.').str[0].str.zfill(6)
crimes_agg = crimes.groupby('dicofre')['ocorrencias'].sum().reset_index()
crimes_agg.rename(columns={'dicofre': 'Dicofre', 'ocorrencias': 'Total_Crimes'}, inplace=True)

# Estado civil: limpar e padronizar Dicofre
estado_civil = estado_civil.drop(columns=[col for col in estado_civil.columns if 'Unnamed' in col or col == 'Total'], errors='ignore')
estado_civil['Dicofre'] = estado_civil['Dicofre'].astype(str).str.split('.').str[0].str.zfill(6)

# === 3. Merge entre crimes e estado_civil ===
df_merged = pd.merge(crimes_agg, estado_civil, on='Dicofre', how='inner')
print(f"✅ Dicofres comuns depois da limpeza: {df_merged.shape[0]}")
print(df_merged.head())

# === 4. Normalização dos dados ===

# Excluir colunas não numéricas ou identificadoras
cols_to_exclude = ['Dicofre']
features = [col for col in df_merged.columns if col not in cols_to_exclude]

# Remover entradas com valores NaN (se existirem)
df_clean = df_merged.dropna(subset=features)

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])
print(f"✅ Normalização feita com sucesso. Shape dos dados normalizados: {X_scaled.shape}")

# === 5. Clustering ===
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# Ver médias por cluster
cluster_summary = df_clean.groupby('Cluster')[features].mean(numeric_only=True)
print("\n📊 Médias por cluster:")
print(cluster_summary)

# === 6. Visualização ===
sns.pairplot(df_clean[features + ['Cluster']], hue='Cluster')
plt.suptitle("Clusters - Crimes vs Estado Civil", y=1.02)
plt.tight_layout()
plt.show()
