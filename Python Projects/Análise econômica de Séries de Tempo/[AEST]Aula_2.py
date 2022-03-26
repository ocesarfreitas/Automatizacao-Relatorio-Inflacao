"""
Análise de Séries Temporais - Aula 2
"""

############################### Modelo de Solow ###############################

## Pacotes do Pythom
# 
import pandas as pd

# Funções de econometria
from econometric_functions import ols_reg

## Importando dados
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