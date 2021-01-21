import re
import nltk
import emoji
import unicodedata
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin, BaseEstimator

def read_chat(file):
    with open(file) as file:
        df = file.read()
    pattern =  r'''
               (?P<Fecha>[^-]+)\s+-\s+
               (?P<Autor>[^:]+):\s+
               (?P<Mensaje>[\s\S]+?)
               (?=(\d{1,2}/\d{1,2}/\d{2})|\Z)
               '''        
    matches = re.finditer(pattern, df, re.MULTILINE | re.VERBOSE)
    df = pd.DataFrame([x.groupdict() for x in matches])
    df['Fecha'] = [''.join(re.findall('(\d{1,2}\/\d{1,2}\/\d{2},\s\d{2}:\d{2})',x)) for x in df['Fecha']]
    df['Fecha'] = pd.to_datetime(df['Fecha'], format = '%m/%d/%y, %H:%M')
    df['Mensaje'] = df['Mensaje'].str.replace('\n',' ')
    return df

def outlier(df, x , p = 0.2):
    var = df[x]
    q1 = var.quantile(p/2)
    q3 = var.quantile(1 - p/2)
    iqr = q3 - q1
    df = df[(var.isnull()) | ((var >= q1 - 1.5*iqr) & (var <= q3 + 1.5*iqr))].copy()
    df.reset_index(drop = True, inplace = True)
    return df


def words(df, cv, autor = 'Autor', texto = 'Mensaje_limpio', n = 11):
    word = pd.DataFrame()
    for x in sorted(np.unique(df[autor])):
        word[x] = pd.DataFrame(df[autor]
                               ).join(pd.DataFrame(cv.fit_transform(df[texto]).todense(),
                                                   index = df.index,
                                                   columns = cv.get_feature_names())
                                     ).pivot_table(columns = df[autor],
                                                   values = cv.get_feature_names(),
                                                   aggfunc = sum
                                                  ).sort_values(by = x,
                                                                ascending = False
                                                               ).head(n).index
    return word

def top_variables(modelo, train, n = 11):
    var = pd.DataFrame()
    var['Modelo'] = [x for x in modelo.named_estimators_]
    var['Top_var'] = [list(pd.DataFrame(sorted(list(zip(modelo.named_estimators_['LogReg']
                                                                    .best_estimator_
                                                                    .coef_[0],
                                                                    train.columns)),
                                               reverse = True)[:n])[1])
                     ] + [list(pd.DataFrame(sorted(list(zip(modelo.named_estimators_[x]
                                                            .best_estimator_
                                                            .feature_importances_,
                                                            train.columns)), reverse = True)[:n])[1]
                              ) for x in var['Modelo'][1:]]

    var = var.join(pd.DataFrame(var['Top_var'].to_list())
                  ).drop('Top_var', axis = 1)
    var.set_index(['Modelo'], inplace = True)
    var = var.transpose()
    return var

class TAD(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self,
                  df,
                  fecha = 'Fecha',
                  autor = 'Autor',
                  mensaje = 'Mensaje'):
        self.df = df
        self.fecha = fecha
        self.autor = autor
        self.mensaje = mensaje
        df = self.df
        fecha = self.fecha 
        autor = self.autor
        mensaje = self.mensaje 
        df[f'{fecha}_anio'] = pd.DatetimeIndex(df[fecha]).year
        df[f'{fecha}_mes'] = pd.DatetimeIndex(df[fecha]).month
        df[f'{fecha}_sem'] = df[fecha].dt.isocalendar().week
        df[f'{fecha}_diasem'] = df[fecha].dt.dayofweek + 1
        df[f'{fecha}_dia'] = df[fecha].dt.day
        df[f'{fecha}_hora'] = df[fecha].dt.hour
        df[f'{fecha}_minute'] = df[fecha].dt.minute
        df['hora_min'] = df[f'{fecha}_hora'] + df[f'{fecha}_minute']/60
        df[fecha] = df[fecha].dt.date
        grupo = [fecha,
                 autor,
                 f'{fecha}_anio',
                 f'{fecha}_mes',
                 f'{fecha}_sem',
                 f'{fecha}_diasem',
                 f'{fecha}_dia']
        df = df.groupby(grupo).agg({'hora_min':[lambda x: np.percentile(x,10),
                                                lambda x: np.percentile(x,25),
                                                lambda x: np.percentile(x,50),
                                                lambda x: np.percentile(x,75),
                                                lambda x: np.percentile(x,90)],
                                    mensaje:[sum,
                                             'count']})
        df.columns = [x + '_' + y for x, y in df.columns]
        df.rename(columns = {'hora_min_<lambda_0>':'hr_min_10',
                             'hora_min_<lambda_1>':'hr_min_25',
                             'hora_min_<lambda_2>':'hr_min_50',
                             'hora_min_<lambda_3>':'hr_min_75', 
                             'hora_min_<lambda_4>':'hr_min_90',
                             'Mensaje_sum':'Mensaje'}, 
                  inplace = True)
        df.reset_index(inplace = True)
        nltk.download('stopwords')
        stop_words = stopwords.words('spanish')
        def clean_text(text, pattern='[^a-zA-Z]'):
            cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
            cleaned_text = re.sub(pattern, ' ', cleaned_text.decode('utf-8'), flags=re.UNICODE)
            cleaned_text = u' '.join(cleaned_text.lower().split())
            return cleaned_text  
        texto = []
        for x in [n.split() for n in [clean_text(x) for x in df[mensaje]]]:
            aux = []
            for word in x:
                if word != 'a':
                    word = re.sub('j*a*(jaja)+j*a*','jaja',word)
                    if word not in stop_words + ['media','omitted','https','www','com']:
                        aux.append(word)
            texto.append(aux)
        df[f'{mensaje}_limpio'] = [' '.join(x) for x in texto]
        df[f'{mensaje}_long'] = df[mensaje].str.len()
        df[f'{mensaje}_n_words'] = df[mensaje].str.split().str.len()
        df[f'{mensaje}_n_letters'] = df[mensaje].map(lambda x:sum(map(str.isalpha, x)))
        df[f'{mensaje}_n_whitespaces'] = df[mensaje].map(lambda x:len(re.findall('\s', x)))
        df[f'{mensaje}_n_media'] = df[mensaje].map(lambda x:len(re.findall('<Media omitted>', x)))
        df[f'{mensaje}_url'] = (df[mensaje].str.contains('http'))*1
        df[f'{mensaje}_n_emojis'] = df[mensaje].map(emoji.emoji_count)
        autores = sorted(np.unique(df[autor]))
        autores = dict(zip(autores,range(1,len(autores)+1)))
        cat = [x for x in df.columns if x.startswith(f'{fecha}_')]
        num = [x for x in df.describe().columns if x not in cat]
        return df,cat,num,autores