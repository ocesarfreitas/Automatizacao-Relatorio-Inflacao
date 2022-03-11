# Pacotes
library(sidrar)
library(tidyverse)
library(forecast)
library(janitor)
library(purrr)
library(zoo)
library(ggthemes)

# Importando base do PIM sazonalizada da Indústria de Transformação por UF
PIM <-  get_sidra(api='/t/3653/p/all/v/3135/c544/129316/n3/all')

# Limpado a base
PIM_saz <- PIM %>% 
  select(Valor, `Mês (Código)`, `Unidade da Federação`) %>%
  rename(mes = `Mês (Código)`, valor = Valor, UF = `Unidade da Federação`) %>%
  filter(mes >= 201201)

# Tratando a base e transformando os vetores das séries em "ts"
PIM_Estado_saz <- PIM_saz %>%
  tidyr::pivot_wider(names_from = "UF", values_from = c("valor")) %>%
  janitor::clean_names() %>%
  dplyr::mutate(across(where(is.numeric), ~ ts(.,
                                        start = c(2012,1),
                                        end = c(2021,9),
                                        frequency = 12)))
# Separando o vetor de data
datas <- PIM_Estado_saz$mes

# Função para dessazonalizar a base
dessazonalizar <- function(y, Estado) {
  ts.sla <- stats::stl(y, "periodic")
  ts.sl <- forecast::seasadj(ts.sla)
  
  tibble::as.tibble(ts.sl) %>%
    dplyr::rename(Indice = x) %>%
    dplyr::mutate(UF = ifelse(is.ts(Indice) == T, Estado, Estado),
                  Mes = datas)
}

# Aplicando a função para dessazonalizar nas UF
## Ps. Procurando uma forma de automizar esse processo
PIM_AM <- dessazonalizar(PIM_Estado_saz$amazonas, "Amazonas")                   
PIM_PA <- dessazonalizar(PIM_Estado_saz$para, "Pará")
PIM_CE <- dessazonalizar(PIM_Estado_saz$ceara, "Ceará")
PIM_PE <- dessazonalizar(PIM_Estado_saz$pernambuco, "Pernambuco")
PIM_BA <- dessazonalizar(PIM_Estado_saz$bahia, "Bahia")
PIM_MG <- dessazonalizar(PIM_Estado_saz$minas_gerais, "Minas Gerais")
PIM_ES <- dessazonalizar(PIM_Estado_saz$espirito_santo, "Espírito Santo")
PIM_RJ <- dessazonalizar(PIM_Estado_saz$rio_de_janeiro, "Rio de Janeiro")
PIM_SP <- dessazonalizar(PIM_Estado_saz$sao_paulo, "São Paulo")
PIM_PR <- dessazonalizar(PIM_Estado_saz$parana, "Paraná")
PIM_SC <- dessazonalizar(PIM_Estado_saz$santa_catarina, "Santa Catarina")
PIM_RS <- dessazonalizar(PIM_Estado_saz$rio_grande_do_sul, "Rio Grande do Sul")
PIM_MT <- dessazonalizar(PIM_Estado_saz$mato_grosso, "Mato Grosso")
PIM_GO <- dessazonalizar(PIM_Estado_saz$goias, "Goiás")

# Listando os vetores por UF dessazonalizados 
list_dessaz <- list(PIM_AM, PIM_PA, PIM_CE,
                    PIM_PE, PIM_BA, PIM_MG,
                    PIM_ES, PIM_RJ, PIM_SP,
                    PIM_PR, PIM_SC, PIM_RS,
                    PIM_MT, PIM_GO)

# Unindo todas as bases
PIM_Estado_dessaz_0 <- purrr::reduce(list_dessaz, full_join)

# Arrumando datas
PIM_Estado_dessaz <- PIM_Estado_dessaz_0 %>%
  mutate(data = as.yearmon(Mes, "%Y%m")) %>%
  select(-Mes)

# Agrupando por região
PIM_Est_Reg_dessaz <- PIM_Estado_dessaz %>%
  mutate(Regiao = case_when(
    UF %in% c("Amazonas","Pará") ~ "Norte",
    UF %in% c("Ceará", "Bahia","Pernambuco") ~ "Nordeste",
    UF %in% c("Minas Gerais", "Espírito Santo", "Rio de Janeiro", "São Paulo") ~ "Sudeste",
    UF %in% c("Paraná", "Rio Grande do Sul", "Santa Catarina") ~ "Sul",
    UF %in% c("Goiás", "Mato Grosso") ~ "Centro-Oeste"
  ))

# Salvando a base dessazonalizada
write.csv(PIM_Estado_dessaz, file = "PIM_ind_trans_dessaz.csv")

# Retirando os vetores que não serão mais precisos
rm(list = c("PIM_AM", "PIM_PA", "PIM_CE", "PIM_PE", "PIM_BA", "PIM_MG", "PIM_ES",
            "PIM_RJ", "PIM_SP", "PIM_PR", "PIM_SC", "PIM_RS", "PIM_MT", "PIM_GO",
            "list_dessaz", "PIM_Estado_dessaz_0", "PIM_Estado_saz"))
