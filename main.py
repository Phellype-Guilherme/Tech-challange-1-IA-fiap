# main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== ETAPA 1: Carregar e Explorar os Dados ==========
df = pd.read_csv("data/dados_seguros.csv")

print("👀 Primeiras linhas do dataset:\n", df.head())
print("\n🧼 Verificação de valores ausentes:\n", df.isnull().sum())

# ========== ETAPA 2: Pré-processamento ==========
# Transformar variáveis categóricas em variáveis numéricas (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=["gênero", "fumante", "região"], drop_first=True)

# Garantir que todos os dados estejam no formato numérico
df_encoded = df_encoded.astype(float)

# Separar variáveis independentes e dependente
X = df_encoded.drop("encargos", axis=1)
y = df_encoded["encargos"]

# Dividir os dados em conjunto de treino e teste (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== ETAPA 3: Modelagem ==========
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# ========== ETAPA 4: Avaliação do Modelo ==========
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # corrigido para funcionar em qualquer versão
r2 = r2_score(y_test, y_pred)

print(f"\n📊 MAE: {mae:.2f}")
print(f"📊 RMSE: {rmse:.2f}")
print(f"📊 R²: {r2:.2f}")

# ========== ETAPA 5: Validação Estatística ==========
# Adiciona constante para o modelo OLS
X_train_const = sm.add_constant(X_train).astype(float)
y_train_float = y_train.astype(float)

ols_model = sm.OLS(y_train_float, X_train_const).fit()
resumo_ols = ols_model.summary()

# ========== ETAPA 6: Visualização ==========
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.title("Valores Reais vs Previsões")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/grafico_previsao_vs_real.png")
plt.close()

# ========== ETAPA 7: Geração de Relatório ==========
with open("output/relatorio_modelo.txt", "w", encoding="utf-8") as f:
    f.write("Relatório - Modelo de Regressão Linear\n")
    f.write("=" * 40 + "\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R²: {r2:.2f}\n\n")
    f.write("Resumo Estatístico (Statsmodels):\n")
    f.write(str(resumo_ols))

print("\n✅ Execução finalizada. Resultados salvos em:\n - images/grafico_previsao_vs_real.png\n - output/relatorio_modelo.txt")
