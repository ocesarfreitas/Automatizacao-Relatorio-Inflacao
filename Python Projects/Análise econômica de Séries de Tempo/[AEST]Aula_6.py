"""
Análise de Séries Temporais - Aula 5
"""
########################### Metodologia Box Jenkins ###########################

## Pacotes do Python
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 
from econometric_functions import ols_reg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.recursive_ls import RecursiveLS

#from chowtest import ChowTest

## Funções de econometria
# from econometric_functions import ols_reg

## Patsy formulas
# from statsmodels.tsa.filters.hp_filter import hpfilter

##
# Arquivo com os vetores aleatórios
df_random = pd.read_csv("AR1.csv")

# Selecionando variável de interesse
df_random_x1 = df_random[['X1']]

# 
fig, axes = plt.subplots(1,2, figsize=(16,3), dpi= 100)

plot_acf(df_random_x1, lags=15, ax=axes[0])
plot_pacf(df_random_x1, lags=15, ax=axes[1])

# 
array_random_ac = np.asarray(df_random_x1) 

resultado_acf = acf(array_random_ac, nlags = 15, qstat = True)
resultado_pacf = pacf(array_random_ac, nlags = 15, method = "ywm")

arma_mod = ARIMA(array_random_ac, order=(2, 0, 2), trend="n")
arma_res = arma_mod.fit()

arma_res.summary()

## Quartely
df_quart = pd.read_csv("quarterly.csv")
df_quart_s = df_quart[['S']]

# Filtrando 
df_quart_f = df_quart[df_quart['Date'] <= '2012-10-01']


#
df_quart_f_s = df_quart_f[['S']]

# 
fig, axes = plt.subplots(1,2, figsize=(16,3), dpi= 100)

plot_acf(df_quart_f_s, lags=12, ax=axes[0])
plot_pacf(df_quart_f_s, lags=12, ax=axes[1])

# 
array_quart_s = np.array(df_quart_f_s) 

resultado_acf = acf(array_quart_s, nlags = 12, qstat = True)
resultado_pacf = pacf(array_quart_s, nlags = 12, method = "ywm")

arma_mod = ARIMA(array_quart_s, order = (list(range(1,10)),0,0), trend = 'n')
model = arma_mod.fit()
model.summary()

# Retirando variáveis do modelo
#lista_ar = list(range(1,5)) + [8,9]
#lista_ma = list(range(1,5)) + [8,9]

lista_ar = [1]
lista_ma = [1,2,7]

arma_mod = ARIMA(array_quart_s, order = (lista_ar,0,lista_ma), trend = 'n')
model = arma_mod.fit()
model.summary()

# Modelo acima com o intercepto
arma_mod = ARIMA(array_quart_s, order = (lista_ar,0,lista_ma), trend = 'c')
mod_arima = arma_mod.fit()
mod_arima.summary()

# Plots
mod_arima.plot_diagnostics(figsize=(10, 10))
plt.show()

# ACF e PACF dos resíduos 
res_acf = acf(mod_arima.resid, nlags = 12, qstat = True)
res_pacf = pacf(mod_arima.resid, nlags = 12, method = "ywm")


### Break Tests
## Metodo alternativo 
df_quart_f.Date = pd.to_datetime(df_quart_f.Date)
variables_to_keep = ['Date', 'S', 'D_S']
my_data = df_quart_f[variables_to_keep]

my_data.index = pd.DatetimeIndex(my_data.Date)
my_data['S_lag'] = my_data[['S']].shift(1)
my_data = my_data.dropna()


#my_data['quart_1980_01'] = np.where(my_data['Date'] == '1980-01-01 00:00:00', 1, 0)

## Teste de Chow
#start = '1979-10-01 00:00:00'
#stop = '1980-04-01 00:00:00'
#
#ChowTest(X = my_data[['quart_1980_01']],
#         y = my_data[['S']],
#         last_index_in_model_1 = stop,
#         first_index_in_model_2 = start)
# Fazer por dummy cansei de tentar

## Teste de cusum

# Fiz coisa inútil, mas vale apena registrar o aprendizado kk
#predicted_y = np.array(mod_arima.fittedvalues)
#actual_y = np.array(my_data[['S']]) 

#error = list()
#for item1, item2 in  zip(actual_y, predicted_y):
#    item = item1 - item2
#    error.append(item)
#    error = np.array(error)

# Mais coisas provavelmente erradas kk
#error = np.array(mod_arima.forecasts_error)
#error = np.transpose(error)

mod_break = ols_reg('S ~ D_S + S_lag', my_data)

endog = my_data[['S']]
exog = sm.add_constant(my_data[['D_S', 'S_lag']])

mod_break = RecursiveLS(endog, exog)
mod_break = mod_break.fit()
print(mod_break.summary())

mod_break.plot_cusum()







