import pandas as pd

teams = pd.read_csv(r'https://raw.githubusercontent.com/dataquestio/project-walkthroughs/master/beginner_ml/teams.csv')

#O primeiro passo é remover as colunas que não fazem sentido para nossa análise
teams = teams[['team','country','year','athletes', 'age','prev_medals','medals']]

# print(teams.head())
#Vamos saber a correlação de uma coluna com as outras? Medals é a coluna que estamos tentando prever,
#veremos qual a relação dela com as outras colunas
# print(teams.corr()['medals'])


#AQUI PODEMOS VISUALIZAR A RELAÇÃO ENTRE AS COLUNAS
import seaborn as sns
import matplotlib.pyplot as plt

sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None)
# plt.show()

#VAMOS VERIFICAR SE EXISTEM DADOS NULOS NA NOSSA BASE DE DADOS
teams[teams.isnull().any(axis=1)]
#IREMOS DROPAR AS LINHAS QUE TIVEREM DADOS FALTANTES

teams = teams.dropna()

train = teams[teams['year'] < 2012].copy()
test = teams[teams['year'] >= 2012].copy()


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

predictors = ['athletes','prev_medals']
target = 'medals'

reg.fit(train[predictors], train['medals'])


predictions = reg.predict(test[predictors])
#PERCEBA QUE HÁ UM ERRO NESSAS PREVISOES, TEMOS NÚMEROS REAIS, 
#SENDO QUE É IMPOSSÍVEL GANHAR 1.9 MEDALHAS POR EXEMPLO. ALÉM DISSO TEMOS NÚMEROS NEGATIVOS.
test['predictions'] = predictions
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test['medals'], test['predictions'])

print(error)


#VAMOS SUBSTITUIR AS COLUNAS COM NUMEROS NEGATIVOS POR 0
test.loc[test['predictions'] < 0, 'predictions'] = 0

test['predictions'] = test['predictions'].round()
#VAMOS VERIFICAR A MEDIA DE ERROS APÓS AS ALTERAÇÕES ACIMA.

from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test['medals'], test['predictions'])

print(error)
#JÁ MELHORAMOS, DIMINUIMOS UM POUCO A MARGEM, MAS AINDA TEMOS TRABALHO A FAZER...

#VAMOS VISUALIZAR ALGUMAS INFORMAÇÕES A RESPEITO DO NOSSO DATASET...
teams.describe()['medals']

#O INTERESSANTE É QUE SUA MÉDIA DE ERRO ESTEJA ABAIXO DO DESVIO PADRÃO, SE ESTIVER ACIMA. ESTÁ MUITO ERRADO.
#VAMOS ANALISAR A PREVISAO QUE FIZEMOS PARA ALGUNS PAÍSES

print(test[test['team'] == 'IND'])

print(test[test['team'] == 'USA'])

#VAMOS AGORA VISUALIZAR O ERRO ABSOLUTO.
errors = (test["medals"] - test['predictions']).abs()
print(errors)

#VAMOS AGRUPAR O ERRO ABSOLUTO POR TIME E CALCULAR SUA MÉDIA.
error_by_team = errors.groupby(test['team']).mean()
print(error_by_team)


#O QUE FAZER PARA MELHORAR?
#TENTAR UM NOVO MODELO
#ADICIONAR MAIS DADOS