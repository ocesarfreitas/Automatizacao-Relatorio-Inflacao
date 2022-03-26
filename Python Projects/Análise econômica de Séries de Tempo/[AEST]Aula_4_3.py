"""
Análise de Séries Temporais - Aula 4 (parte 3)
"""
################################## Filtro HP ##################################

## Pacotes do Python
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

## Funções de econometria
from econometric_functions import ols_reg

## Patsy formulas
from statsmodels.tsa.filters.hp_filter import hpfilter

##Tratamento
# Importaçãp
df_pib = pd.read_csv("PIB aplicação HP.csv")

# Coluna de data no formato data
df_pib['Date'] = pd.to_datetime(df_pib['Date'])

# Extraindo mês e ano de data
df_pib['Month'] = [i.month for i in df_pib['Date']]
df_pib['Year'] = [i.year for i in df_pib['Date']]

# Filtrando até onde os dados fazem sentido 
df_pib_filter = df_pib[df_pib['Date'] <= '2021-07-01']

## Filtro HP 
# Aplicando o filtro no modelo
df_pib_filter['cycle'],df_pib_filter['trend'] = hpfilter(df_pib_filter['PIB'], lamb=1600)

## Plot dos gráficos
# Filtro HP x Valores Observados
fig, ax = plt.subplots(figsize = (12,6)) 

# Definindo características do gráfico
ax.plot(df_pib_filter['Date'], df_pib_filter['PIB'])
ax.plot(df_pib_filter['Date'], df_pib_filter['trend'])
ax.set(title='Filtro HP (Lambda = 1600)', xlabel='Ano', ylabel='PIB')

# Ciclos 
fig, ax = plt.subplots(figsize = (12,6)) 

# Definindo características do gráfico
ax.plot(df_pib_filter['Date'], df_pib_filter['cycle'], color = 'green')
ax.set(title='Ciclos', xlabel='Ano', ylabel='PIB')

## Modelo de regressão 
# Criando um vetor de sequencia (1,2,...,n) para estimar a tendência no modelo
df_pib_filter['Series'] = np.arange(1,len(df_pib_filter)+1)

# Estimando o modelo
fit_pib = ols_reg('PIB ~ Series', df_pib_filter)

# Unindo os valores estimados e resíduos com a base
df_pib_filter['fitted_values_0'] = fit_pib.fittedvalues
df_pib_filter['residuals_0'] = fit_pib.resid

# Gráfico do modelode regressão
fig, ax = plt.subplots(figsize = (12,6)) 

# Definindo características do gráfico
ax.plot(df_pib_filter['Date'], df_pib_filter['PIB'])
ax.plot(df_pib_filter['Date'], df_pib_filter['fitted_values_0'])
ax.set(title='Valores estimados x Valores Observados', xlabel='Ano', ylabel='PIB')

# Resíduos
fig, ax = plt.subplots(figsize = (12,6)) 

# Definindo características do gráfico
ax.plot(df_pib_filter['Date'], df_pib_filter['residuals_0'], color = 'green')
ax.set(title='Resíduos', xlabel='Ano', ylabel='PIB')

"Sem saco para fazer a mesma coisa com as variáveis de PIB"

df_pib_ac = df_pib_filter[['Date', 'PIB']].set_index(['Date'])

fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)

plot_acf(df_pib_ac, lags=36, ax=axes[0])
plot_pacf(df_pib_ac, lags=36, ax=axes[1])

array_pib_acf = np.asarray(df_pib_ac) 

resultado_acf = acf(array_pib_acf, nlags = 36, qstat = True)
resultado_pacf = pacf(array_pib_acf, nlags = 36, method = "ywm")

# Aplicando primeira diferença
df_pib_ac_diff = df_pib_ac.diff()
df_pib_ac_diff = df_pib_ac_diff.dropna()

fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)

plot_acf(df_pib_ac_diff, lags=36, ax=axes[0])
plot_pacf(df_pib_ac_diff, lags=36, ax=axes[1])

array_pib_ac_diff = np.asarray(df_pib_ac_diff) 

resultado_acf_diff = acf(array_pib_ac_diff, nlags = 36, qstat = True)
resultado_pacf_diff = pacf(array_pib_ac_diff, nlags = 36, method = "ywm")




