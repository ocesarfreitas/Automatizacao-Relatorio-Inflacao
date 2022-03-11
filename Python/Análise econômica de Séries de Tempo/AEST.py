"""
Análise de Séries Temporais 
"""

################################### Imports ###################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Patsy formulas
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.formula.api import ols

## Funções de econometria
from econometric_functions import ols_reg, first_difference

""" Aula 2 - Modelo de Solow """

mrw_df = pd.read_csv("mrw.csv")


## Modelo estimado (sem capital humano)
fit_model_0 = ols_reg("LYL85~LNS+LNNGD", mrw_df)

# Intervalos de confiança
conf_int_a = [0.1,0.05,0.01]
for alpha in conf_int_a:
    print(fit_model_0.conf_int(alpha))

# Teste de Wald
hypothesis_0 = '(LNS=0.53)'
print('H0:', hypothesis_0)
print(fit_model_0.wald_test(hypothesis_0))

## Modelo de Solow Estimado (com capital humano) 
fit_model_1 = ols_reg("LYL85~LNS+LNNGD+LNSCH", mrw_df)

# Intervalos de confiança
for alpha in conf_int_a:
    print(fit_model_1.conf_int(alpha))


""" Aula 3 - Equações em diferenças """
### Gerando um vetor que representa os choques aletórios 

## Série X1

# Números normais aleatórios 
n_norm_a = np.random.normal(size = (200,1))

# Array vazio para fazer o bagulho do looping
x = np.full([200,1], float(0))

# Looping para fazer com rolês com a variável defasada
for i in range(len(x)):
    if i == 0:
        x[i,0] = 0
    else:
        x[i,0] = x[i-1,0]*(0.9) + n_norm_a[i-1,0]

# Transformando o resultado do loop em df
random_x1 = pd.DataFrame(x)

# Criando uma variável indice e incorporando ao df
indice = list(range(0,200))
random_x1['indice'] = indice

# Renomenado colunas
random_x1.columns = ['X1','indice']

# Plotando o resultado
fig, ax1 = plt.subplots(figsize = (12,6)) 

ax1.plot(random_x1['indice'], random_x1['X1'], color='red')
ax1.set(title='Choque aleatório 1', ylabel='X1', xlabel='Índice')

## Série X2

# Array vazio para realizar o looping
x = np.full([200,1], float(0))

# Looping para fazer com rolês com a variável defasada
for i in range(len(x)):
    if i == 0:
        x[i,0] = 0
    else:
        x[i,0] = x[i-1,0]*(0.7) + n_norm_a[i-1,0]

# Transformando o resultado do loop em df
random_x2 = pd.DataFrame(x)

# Criando uma variável indice e incorporando ao df
indice = list(range(0,200))
random_x2['indice'] = indice

# Renomenado colunas
random_x2.columns = ['X2','indice']

# Plotando o resultado
fig, ax2 = plt.subplots(figsize = (12,6)) 

ax2.plot(random_x2['indice'], random_x2['X2'], color='red')
ax2.set(title='Choque aleatório 2', ylabel='X2', xlabel='Índice')

# Merge dos dados 
random_x = random_x1.merge(random_x2, how = 'left', on = 'indice')

## Série X3

# Array vazio para realizar o looping
x = np.full([200,1], float(0))

# Looping para fazer com rolês com a variável defasada
for i in range(len(x)):
    if i == 0:
        x[i,0] = 0
    else:
        x[i,0] = x[i-1,0]*(1.01) + n_norm_a[i-1,0]

# Transformando o resultado do loop em df
random_x3 = pd.DataFrame(x)

# Criando uma variável indice e incorporando ao df
indice = list(range(0,200))
random_x3['indice'] = indice

# Renomenado colunas
random_x3.columns = ['X3','indice']

# Plotando o resultado
fig, ax3 = plt.subplots(figsize = (12,6)) 

ax3.plot(random_x3['indice'], random_x3['X3'], color='red')
ax3.set(title='Choque aleatório 3', ylabel='X3', xlabel='Índice')

# Merge dos dados 
random_x = random_x.merge(random_x3, how = 'left', on = 'indice')

""" Aula 4 - Fundamento de séries de tempo (parte 2 -> Regressões Espúrias) """
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
## Estimando o modelo de primeira diferença 
"Dúvidas da vez, porque a função de primeira diferença não funfou?? e outra feita a mão sim??"
"próxima coisa a procurar"

# Modelo de primeira diferença Vini (não funfou igual o do prof)
fit_fd_reg_esp = first_difference(data= df_reg_esp_filter,
                                  formula= 'LCUS ~ LYUK + Series',
                                  index= (['Year', 'Month']))

# Modelo de primeira diferença mano aleatório do StackOVerflow (funfou igual o do prof)
"Obs.: Fazer um bagulho de 'def' bonitinho depois"
df_reg_esp_diff = df_reg_esp_filter.diff()

result = ols('LCUS ~ LYUK', df_reg_esp_diff).fit()
print(result.summary())

# Modelo só com a trend
fit_reg_esp_2 = ols_reg('LCUS ~ LYUK + Series', df_reg_esp_filter)

## Atividade woooldridge
# Importando base
df_earns = pd.read_csv("earns.csv")

reg_earns = ols_reg('LHRWAGE ~ LOUTPHR + T', df_earns)

""" Aula 4 - Fundamento de séries de tempo (parte 3 -> Filtro HP) """
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

""" Aula 5 - Metodologia Box Jenkins"""











