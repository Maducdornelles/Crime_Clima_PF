# PROJETO: CRIMINALIDADE E METEOROLOGIA - PASSO FUNDO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Garante que a pasta de destino exista
os.makedirs('../processed', exist_ok=True)

# =============================================================================
# ETAPA 2 - CARREGANDO E CONCATENANDO OS DADOS DE CRIMES
# =============================================================================
print("ETAPA 2 - Carregando dados de crimes...")

df_2021 = pd.read_csv('../raw/dados_crime/2021.csv', sep=';', encoding='latin1', low_memory=False)
df_2022 = pd.read_csv('../raw/dados_crime/2022.csv', sep=';', encoding='latin1', low_memory=False)
df_2023 = pd.read_csv('../raw/dados_crime/2023.csv', sep=';', encoding='latin1', low_memory=False)
df_2024 = pd.read_csv('../raw/dados_crime/2024.csv', sep=';', encoding='latin1', low_memory=False)
df_2025 = pd.read_csv('../raw/dados_crime/2025.csv', sep=';', encoding='latin1', low_memory=False)
df_2026 = pd.read_csv('../raw/dados_crime/2026.csv', sep=';', encoding='latin1', low_memory=False)

df_crimes = pd.concat([df_2021, df_2022, df_2023, df_2024, df_2025, df_2026], ignore_index=True)
df_crimes.columns = df_crimes.columns.str.strip().str.lower().str.replace(' ', '_')
df_crimes = df_crimes[df_crimes['Municipio'] == 'PASSO FUNDO']

print(f"-> Total de ocorrências em Passo Fundo: {len(df_crimes)}")


# =============================================================================
# ETAPA 3 - LIMPEZA E PADRONIZAÇÃO
# =============================================================================
print("\nETAPA 3 - Realizando limpeza de dados...")

df_crimes = df_crimes.drop_duplicates()
df_crimes['DATA_PADRAO'] = pd.to_datetime(df_crimes['data_fato'], format='%d/%m/%Y', errors='coerce')

colunas_texto = df_crimes.select_dtypes(include='object').columns
df_crimes[colunas_texto] = df_crimes[colunas_texto].fillna('NÃO INFORMADO')
df_crimes['TIPO_CRIME_PADRAO'] = df_crimes['tipo_crime'].str.lower().str.strip()

df_crimes_diario = df_crimes.groupby('DATA_PADRAO').size().reset_index(name='TOTAL_CRIMES')
print(f"-> Dias únicos com ocorrências: {len(df_crimes_diario)}")


# =============================================================================
# ETAPA 4 - INTEGRAÇÃO COM DADOS METEOROLÓGICOS
# =============================================================================
print("\nETAPA 4 - Integrando com dados de clima...")

df_clima_2021 = pd.read_csv('../raw/dado_clima/INMET_2021.CSV', sep=';', encoding='latin1', skiprows=8, decimal=',')
df_clima_2022 = pd.read_csv('../raw/dado_clima/INMET_2022.CSV', sep=';', encoding='latin1', skiprows=8, decimal=',')
df_clima_2023 = pd.read_csv('../raw/dado_clima/INMET_2023.CSV', sep=';', encoding='latin1', skiprows=8, decimal=',')
df_clima_2024 = pd.read_csv('../raw/dado_clima/INMET_2024.CSV', sep=';', encoding='latin1', skiprows=8, decimal=',')
df_clima_2025 = pd.read_csv('../raw/dado_clima/INMET_2025.CSV', sep=';', encoding='latin1', skiprows=8, decimal=',')
df_clima_2026 = pd.read_csv('../raw/dado_clima/INMET_2026.CSV', sep=';', encoding='latin1', skiprows=8, decimal=',')

df_clima = pd.concat([df_clima_2021, df_clima_2022, df_clima_2023, df_clima_2024, df_clima_2025, df_clima_2026], ignore_index=True)
df_clima.columns = df_clima.columns.str.strip().str.upper()

df_clima['DATA_PADRAO'] = pd.to_datetime(df_clima['DATA'], format='%Y/%m/%d', errors='coerce')
df_clima['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'] = pd.to_numeric(df_clima['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'], errors='coerce')
df_clima['PRECIPITAÇÃO TOTAL, HORÁRIO (MM)'] = pd.to_numeric(df_clima['PRECIPITAÇÃO TOTAL, HORÁRIO (MM)'], errors='coerce')

df_clima_diario = df_clima.groupby('DATA_PADRAO').agg(
    TEMP_MEDIA=('TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)', 'mean'),
    TEMP_MAX=('TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)', 'max'),
    CHUVA_TOTAL=('PRECIPITAÇÃO TOTAL, HORÁRIO (MM)', 'sum')
).reset_index()

df_final = pd.merge(df_crimes_diario, df_clima_diario, on='DATA_PADRAO', how='inner')
df_final.fillna(df_final.mean(numeric_only=True), inplace=True)

print(f"-> Dataset final: {len(df_final)} linhas")


# =============================================================================
# ETAPA 5 - TRANSFORMAÇÕES
# =============================================================================
print("\nETAPA 5 - Aplicando transformações e salvando CSV...")

df_final['DIA_SEMANA'] = df_final['DATA_PADRAO'].dt.dayofweek   
df_final['MES'] = df_final['DATA_PADRAO'].dt.month
df_final['FIM_DE_SEMANA'] = (df_final['DIA_SEMANA'] >= 5).astype(int)  
df_final['CHUVA_BINARIA'] = (df_final['CHUVA_TOTAL'] > 0).astype(int)  

df_final['CLIMA_CATEGORIA'] = pd.cut(
    df_final['TEMP_MEDIA'],
    bins=[0, 15, 25, 45],
    labels=['Frio', 'Ameno', 'Quente']
)
df_final['CLIMA_NUM'] = df_final['CLIMA_CATEGORIA'].map({'Frio': 0, 'Ameno': 1, 'Quente': 2})

Q1 = df_final['CHUVA_TOTAL'].quantile(0.25)
Q3 = df_final['CHUVA_TOTAL'].quantile(0.75)
IQR = Q3 - Q1
limite = Q3 + 1.5 * IQR
df_final['DIA_TEMPESTADE'] = (df_final['CHUVA_TOTAL'] > limite).astype(int)
print(f"-> Dias com tempestade extrema (outliers de chuva): {df_final['DIA_TEMPESTADE'].sum()}")

scaler = StandardScaler()
df_final[['CRIMES_SCALED', 'TEMP_SCALED', 'CHUVA_SCALED']] = scaler.fit_transform(
    df_final[['TOTAL_CRIMES', 'TEMP_MEDIA', 'CHUVA_TOTAL']]
)

df_final.to_csv('../processed/dataset_final_tratado.csv', index=False)
print("-> Dataset final salvo com sucesso em '../processed/dataset_final_tratado.csv'!")


# =============================================================================
# ETAPA 6 - GRÁFICOS E CORRELAÇÕES (SALVOS LOCALMENTE)
# =============================================================================
print("\nETAPA 6 - Gerando e salvando gráficos na pasta 'processed'...")

df_final['ANO_MES'] = df_final['DATA_PADRAO'].dt.to_period('M').astype(str)

# --- Gráfico 1 ---
df_mensal = df_final.groupby('ANO_MES')['TOTAL_CRIMES'].sum().reset_index()
plt.figure(figsize=(13, 5))
sns.barplot(data=df_mensal, x='ANO_MES', y='TOTAL_CRIMES', color='#2c3e50')
plt.title('Total de Crimes por Mês em Passo Fundo (2021-2026)')
plt.xlabel('Mês')
plt.ylabel('Total de Crimes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../processed/01_crimes_por_mes.png')
plt.close()

# --- Gráfico 2 ---
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df_final.dropna(subset=['CLIMA_CATEGORIA']),
                 x='CLIMA_CATEGORIA', y='TOTAL_CRIMES',
                 estimator='mean', errorbar=None, palette='OrRd')
plt.title('Média de Crimes por Temperatura')
plt.xlabel('Clima')
plt.ylabel('Média de crimes/dia')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig('../processed/02_crimes_por_temperatura.png')
plt.close()

# --- Gráfico 3 ---
df_final['TIPO_DIA'] = df_final['DIA_TEMPESTADE'].map({0: 'Dia Normal', 1: 'Tempestade'})
plt.figure(figsize=(7, 5))
ax = sns.barplot(data=df_final, x='TIPO_DIA', y='TOTAL_CRIMES',
                 estimator='mean', errorbar=None, palette='Blues_r')
plt.title('Crimes em Dias de Tempestade vs Dias Normais')
plt.xlabel('Condição')
plt.ylabel('Média de crimes/dia')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig('../processed/03_crimes_tempestade.png')
plt.close()

# --- Correlação de Spearman ---
colunas_corr = ['TOTAL_CRIMES', 'TEMP_MEDIA', 'TEMP_MAX', 'CHUVA_TOTAL',
                 'DIA_SEMANA', 'MES', 'FIM_DE_SEMANA', 'CHUVA_BINARIA', 'CLIMA_NUM']
corr = df_final[colunas_corr].corr(method='spearman')

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlação de Spearman – Crimes x Clima x Calendário')
plt.tight_layout()
plt.savefig('../processed/04_heatmap_correlacao.png')
plt.close()

print("\nCorrelações com TOTAL_CRIMES calculadas.")
# Salva as correlações em um txt para fácil visualização
with open('../processed/correlacoes_crimes.txt', 'w') as f:
    f.write("Correlações de Spearman com TOTAL_CRIMES:\n\n")
    f.write(corr['TOTAL_CRIMES'].sort_values(ascending=False).to_string())

print("\n✅ Processamento concluído! Verifique a pasta 'processed' para ver o CSV, os gráficos em PNG e o relatório em TXT.")