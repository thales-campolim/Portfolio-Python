#import libraries 

import datetime
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy.stats import linregress


#Loading database

df_corrida = pd.read_csv('Performancecorrida.csv',parse_dates=['Data'],dayfirst=True, delimiter=';')

#Setting 'Date'as index

df_corrida.set_index('Data', inplace=True)

#Convert avarage pace to numerical float

df_corrida['Ritmo médio'] = pd.to_numeric(df_corrida['Ritmo médio'], errors='coerce').astype(float)
df_corrida['Distancia(km)'] = pd.to_numeric(df_corrida['Distancia(km)'], errors='coerce').astype(float)
df_corrida['1km'] = pd.to_numeric(df_corrida['1km'], errors='coerce').astype(float)


#Calculate speed through duration(s) and distance(km)

df_corrida['tempo_h'] = (df_corrida['Duração(segundos)'] / 3600.0)
df_corrida['Km/h médio'] = (df_corrida['Distancia(km)'] / df_corrida['tempo_h']).round(2)



#Check if all the data is in the correct type
df_corrida.info()

#Show the dataset

df_corrida

#Check for null values and its position

valores_nulos = df_corrida.loc[df_corrida['Distancia(km)'].isnull()]
valores_nulos


#calcula a média de ritmo por mês, e o número de corridas no mêsCalculate the monthly average speed, number of runs, and distance.

media_mensal = df_corrida.groupby(pd.Grouper(freq='M')).agg({'Km/h médio': ['mean', 'count'],'Distancia(km)': ['sum'],'1km': ['mean']}).round(2)

#Rename collums

media_mensal.columns = ['Km/h médio', 'Número de Corridas','Distancia mensal','1km']


#Set graph style

plt.style.use(['ggplot'])

#Show average speedy, number of runs, and distance distributuion

plt.figure(figsize=(10,3))

plt.subplot(1,3,1) 
plt.title('Distribuição velocidade média') 
sns.histplot(media_mensal, x='Km/h médio', bins=4) 
plt.xticks(np.arange(9, max(media_mensal['Número de Corridas']), 0.5))

plt.subplot(1,3,2) 
plt.title('Distribuição da qtd. corridas') 
sns.histplot(media_mensal, x='Número de Corridas', bins=4) 
plt.xticks(np.arange(2, max(media_mensal['Número de Corridas']), 1))

plt.subplot(1,3,3) 
plt.title('Distribuição da distância mensal') 
sns.histplot(media_mensal, x='Distancia mensal', bins=4) 
plt.xticks(np.arange(5, max(media_mensal['Distancia mensal']), 10)) 
plt.show()


#Plot line charts for number of runs and average speed, and tendence lines

# Set graphics size

plt.figure(figsize=(12,6))

#Number of runs chart

plt.subplot(2,1,1)

slope, intercept, r_value, p_value, std_err = linregress(range(len(media_mensal['Número de Corridas'])), media_mensal['Número de Corridas'])
line = slope*np.array(range(len(media_mensal['Número de Corridas']))) + intercept

plt.title('Corridas por mês',fontsize=16)
plt.plot(media_mensal['Número de Corridas'], 'o--', label = 'Número de Corridas', color='green')
for (i,valor) in enumerate(media_mensal['Número de Corridas'], start=0):
    plt.text(x=media_mensal.index[i],
    y = valor+0.5,
    s=f'{valor}')
plt.ylim(bottom=0)
plt.yticks(np.arange(0, max(media_mensal['Número de Corridas'])+4, 2))
plt.tick_params(axis='both', labelsize=11)
plt.plot(media_mensal.index, line,color = 'blue', alpha=0.1, label='tendência')
plt.legend()

# Average speed chart

plt.subplot(2,1,2)

slope, intercept, r_value, p_value, std_err = linregress(range(len(media_mensal['Km/h médio'])), media_mensal['Km/h médio'])
line = slope*np.array(range(len(media_mensal['Km/h médio']))) + intercept

plt.title('Velocidade média por mês',fontsize=16)
plt.plot(media_mensal['Km/h médio'], 'o--', label = 'Km/h médio', color='red')

for (i,valor) in enumerate(media_mensal['Km/h médio'], start=0):
    plt.text(x=media_mensal.index[i],
    y = valor+0.5,
    s=f'{valor}')
plt.yticks(np.arange(6, max(media_mensal['Número de Corridas'])+2, 1))
plt.ylim(top=15)
plt.tick_params(axis='both', labelsize=11)
plt.plot(media_mensal.index, line,color = 'darkslateblue', alpha=0.1, label='tendência')
plt.legend()
plt.show()

#Bar chart for monthly distance

plt.figure(figsize=(12,5))
plt.bar(media_mensal.index, media_mensal['Distancia mensal'], color= 'steelblue',width=15)
for (i,valor) in enumerate(media_mensal['Km/h médio'], start=0):
    plt.text(x=media_mensal.index[i],
    y = valor+50,
    s=f'{valor}')
plt.show()


# Pearson correlation test

corr = df_corrida['1km'].corr(df_corrida['Ritmo médio'])

# Correlation coeficient
coef_corr = np.corrcoef(df_corrida['1km'], df_corrida['Ritmo médio'])[0, 1]

# Shows results
print(f"A correlação de Pearson entre as variáveis é: {corr}")
print(f"O coeficiente de correlação é: {coef_corr}")




