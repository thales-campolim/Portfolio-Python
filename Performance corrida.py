#!/usr/bin/env python
# coding: utf-8

# In[1]:


#carregando bibliotecas

import datetime
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy.stats import linregress


# In[22]:


#carregando base

df_corrida = pd.read_csv('Performancecorrida.csv',parse_dates=['Data'],dayfirst=True, delimiter=';')

#define a coluna data como indice

df_corrida.set_index('Data', inplace=True)

#converte coluna ritmo médio para númerico

df_corrida['Ritmo médio'] = pd.to_numeric(df_corrida['Ritmo médio'], errors='coerce').astype(float)
df_corrida['Distancia(km)'] = pd.to_numeric(df_corrida['Distancia(km)'], errors='coerce').astype(float)
df_corrida['1km'] = pd.to_numeric(df_corrida['1km'], errors='coerce').astype(float)


# In[17]:


#cria através das colunas de duração(s) e distância(km) a velocidade em km/h

df_corrida['tempo_h'] = (df_corrida['Duração(segundos)'] / 3600.0)
df_corrida['Km/h médio'] = (df_corrida['Distancia(km)'] / df_corrida['tempo_h']).round(2)


# In[111]:


#características das variáveis do dataset
#df_corrida.info()

#motra o data set

#df_corrida

#verificar se tem valores nulos, e as posições

#valores_nulos = df_corrida.loc[df_corrida['Distancia(km)'].isnull()]
#valores_nulos



# In[14]:


#calcula a média de ritmo por mês, e o número de corridas no mês

media_mensal = df_corrida.groupby(pd.Grouper(freq='M')).agg({'Km/h médio': ['mean', 'count'],'Distancia(km)': ['sum'],'1km': ['mean']}).round(2)

#renomeia o nome das colunas

media_mensal.columns = ['Km/h médio', 'Número de Corridas','Distancia mensal','1km']


# In[5]:


# muda o estilo gráfico a ser utilizado.

plt.style.use(['ggplot'])


# In[298]:


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


# In[316]:


# traça os graficos de número de corridas e velocidade média mensais

# Set o tamanho das figuras
plt.figure(figsize=(12,6))

# Gráfico de corridas por mês

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

# Gráfico de velocidade média por mês e tenência

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


# In[346]:


plt.figure(figsize=(12,5))
plt.bar(media_mensal.index, media_mensal['Distancia mensal'], color= 'steelblue',width=15)
for (i,valor) in enumerate(media_mensal['Km/h médio'], start=0):
    plt.text(x=media_mensal.index[i],
    y = valor+50,
    s=f'{valor}')
plt.show()


# In[ ]:


# Calcula a correlação de Pearson
corr = df_corrida['1km'].corr(df_corrida['Ritmo médio'])

# Obtém o coeficiente de correlação
coef_corr = np.corrcoef(df_corrida['1km'], df_corrida['Ritmo médio'])[0, 1]

# Exibe o resultado
print(f"A correlação de Pearson entre as variáveis é: {corr}")
print(f"O coeficiente de correlação é: {coef_corr}")


# In[ ]:


plt.figure(figsize=(10,2))

#plt.subplot(1,2,1)
plt.scatter(df_corrida.index,df_corrida['1km'])

