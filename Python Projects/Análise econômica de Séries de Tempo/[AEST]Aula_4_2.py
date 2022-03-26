"""
Análise de Séries Temporais - Aula 4 (Parte 2)
"""

######################## Fundamento de séries de tempo ########################
############################# Regressões Espúrias #############################

## Pacotes do Python
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Patsy formulas
from statsmodels.formula.api import ols

## Funções de econometria
from econometric_functions import ols_reg

### Exercício em sala
## Importando 
df_reg_esp = pd.read_csv("regressao_espuria.csv")

## Tratando a base -> Séries
# Coluna de data no formato data
df_reg_esp['Date'] = pd.to_datetime(df_reg_esp['Date'])

# Média móvel (Trimestres) -> Acabou sendo inutil mas deixei para lembrar
# df_reg_esp['MM_UK'] = df_reg_esp['LYUK'].rolling(4).mean()
# df_reg_esp['MM_US'] = df_reg_esp['LCUS'].rolling(4).mean()

# Extraindo mês e ano de data
df_reg_esp['Month'] = [i.month for i in df_reg_esp['Date']]
df_reg_esp['Year'] = [i.year for i in df_reg_esp['Date']]

# create a sequence of numbers
df_reg_esp['Series'] = np.arange(1,len(df_reg_esp)+1)

# Filtrando 
df_reg_esp_filter = df_reg_esp[df_reg_esp['Date'] <= '1998-01-01']

## Estimando o modelo -> Log do PIB do UK vs Tendência da série
fit_reg_espu = ols_reg('LYUK ~ Series', df_reg_esp_filter)

## Unindo os valores estimados e o vetos dos resíduos com a base para fazer os 
## gráficos bonitinhos
# Vetor dos valores estimados
df_reg_esp_filter['fitted_values'] = fit_reg_espu.fittedvalues
# Vetor dos resíduos
df_reg_esp_filter['residuals'] = fit_reg_espu.resid

## Gráfico
# Fitted values x Actual
fig, ax_re = plt.subplots(figsize = (12,6)) 

ax_re.plot(df_reg_esp_filter['Date'], df_reg_esp_filter['fitted_values'])
ax_re.plot(df_reg_esp_filter['Date'], df_reg_esp_filter['LYUK'])
ax_re.set(title='Regressão x Valores Observados', ylabel='Indice', xlabel='Ano')

# Residuals
fig, ax_res = plt.subplots(figsize = (12,6)) 

ax_res.plot(df_reg_esp_filter['Date'], df_reg_esp_filter['residuals'])
ax_res.set(title='Resíduos do modelo', xlabel='Ano', ylabel='Amplitude')

## Estimando modelo -> Log do consumo do US pelo log do PIB do UK
fit_reg_esp1 = ols_reg('LCUS ~ LYUK', df_reg_esp_filter)

## Unindo os valores estimados e o vetos dos resíduos com a base para fazer os 
## gráficos bonitinhos
# Vetor dos valores estimados
df_reg_esp_filter['fitted_values_1'] = fit_reg_esp1.fittedvalues
# Vetor dos resíduos
df_reg_esp_filter['residuals_1'] = fit_reg_esp1.resid

## Gráfico
# Fitted values x Actual
fig, ax_re = plt.subplots(figsize = (12,6)) 

ax_re.plot(df_reg_esp_filter['Date'], df_reg_esp_filter['fitted_values_1'])
ax_re.plot(df_reg_esp_filter['Date'], df_reg_esp_filter['LCUS'])
ax_re.set(title='Regressão x Valores Observados', xlabel='Ano', ylabel='Indices')

# Residuals
fig, ax_res = plt.subplots(figsize = (12,6)) 

ax_res.plot(df_reg_esp_filter['Date'], df_reg_esp_filter['residuals_1'])
ax_res.set(title='Resíduos do modelo', xlabel='Ano', ylabel='Amplitude')

## Estimando o ols com diferenciando
# Primeira diferença mano aleatório do StackOVerflow
def lag_reg(data, formula):
    data_diff = data.diff()
    
    mod = ols(formula=formula, data=data_diff).fit()
    print(mod.summary())
    return mod
    
fit_lag_reg = lag_reg(df_reg_esp_filter, 'LCUS ~ LYUK')

# Modelo só com a trend
fit_reg_esp_2 = ols_reg('LCUS ~ LYUK + Series', df_reg_esp_filter)

## Atividade woooldridge
# Importando base
df_earns = pd.read_csv("earns.csv")

reg_earns = ols_reg('LHRWAGE ~ LOUTPHR + T', df_earns)