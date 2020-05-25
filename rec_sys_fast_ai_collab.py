# Author: Timur Abdualimov, SOVIET team
# Competition: Recommended system, SkillFctory
# First date code: 17.05.2020
# Used: Kaggle notebook, GPU!

from fastai.collab import *
import numpy as np
import pandas as pd 

def open_data():
    """ open datasets"""
    global train, test, sample_submission # объявляем переменные глобальными
    train = pd.read_csv('/kaggle/input/recommendationsv4/train.csv', low_memory = False)
    train = train.drop_duplicates().reset_index(drop = True) # удалим дубликаты, если есть
    test = pd.read_csv('/kaggle/input/recommendationsv4/test_v3.csv', low_memory = False)
    sample_submission = pd.read_csv('/kaggle/input/recommendationsv4/sample_submission.csv')
    
open_data() # открываем все и записываем датасет в переменные

def param_data(data): # посмотрим на данные
    """dataset required parameters """
    param = pd.DataFrame({
              'dtypes': data.dtypes.values,
              'nunique': data.nunique().values,
              'isna': data.isna().sum().values,
              'loc[0]': data.loc[0].values,
              }, 
             index = data.loc[0].index)
    return param

pd.concat([param_data(train), param_data(test)], 
          axis=1, 
          keys = [f'↓ ОБУЧАЮЩАЯ ВЫБОРКА ↓ {train.shape}', f'↓ ТЕСТОВАЯ ВЫБОРКА ↓ {test.shape}'],  
          sort=False)

def viz_na(data):
    """NA visualisation"""
    global cols
    cols = data.columns # запишем названия строки сделаем переменную глобальной
    # определяем цвета 
    # желтый - пропущенные данные, синий - не пропущенные
    colours = ['#000099', '#ffff00'] 
    sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))
    plt.show()


viz_na(train)
viz_na(test)

def stat_na_per_percent(data):
    print(f'{data.shape}')
    for col in data.columns:
        pct_missing = np.mean(data[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))
    print("END", end = '\n\n')
stat_na_per_percent(train)
stat_na_per_percent(test)

def concat_train_test(train , test):
    """
    prepare final data, concat train and test
    """
    global orta
    train['sample'] = 1 # объединяем трейн и тест для совместных правок
    test['sample'] = 0
    test['rating'] = -2
    orta = train.append(test, sort = False).reset_index(drop = True) # закончили объединение и присвоили имя основной перемнной
    orta.drop(['Id'], axis = 1, inplace = True)

concat_train_test(train, test)

# удалим столбцы с пропусками где процент больше 80%
orta = orta.drop(['image', 'vote',  'reviewText', 'summary', 'Unnamed: 0' ], axis = 1)
In [8]:
# создадим отдельные столбцы, которые показывают были ли пропущенные значения в столбцах и заполним пропуски
for i in orta.columns:
    if orta[i].isna().sum() != 0:
        orta[str(i) + '_isNAN'] = pd.isna(orta[i]).astype('uint8')
        orta[i] = orta[i].fillna('[]')

# удалим столбцы style, reviewerName, reviewTime за ненадобностью
orta = orta.drop(['style', 'reviewTime', 'reviewerName'], axis = 1)

# перевдем столбец verified в числовой
orta['verified'] = orta['verified'].astype(int)

# переведем столбец с unixtime в нормальные часики    
orta['unixReviewTime'] = pd.to_datetime(orta['unixReviewTime'], unit='s')
# сколько прошло дней
orta['DaysPassed'] = (pd.datetime.now() - orta['unixReviewTime']).dt.days
# день недели в который оставили отзыв
orta['reviewWeekday'] = orta['unixReviewTime'].dt.weekday
# удалим столбец со временем за ненадобностью
orta = orta.drop(['unixReviewTime'], axis = 1)

train_data = orta.query('sample == 1').drop(['sample'], axis = 1)
test_data = orta.query('sample == 0').drop(['sample', 'rating'], axis = 1)

train_data = pd.DataFrame({
    'userid': train_data['userid'],
    'itemid': train_data['itemid'],
    'rating': train_data['rating']
})

test_data = pd.DataFrame({
    'userid': test_data['userid'],
    'itemid': test_data['itemid'],
})

data = CollabDataBunch.from_df(train_data, test=test_data, seed=42)

y_range = [0, 1]

learn = collab_learner(data, use_nn = True, emb_szs = {'userid': 40, 'itemid': 40}, layers = [256, 128], n_factors=50, y_range=y_range)

learn.fit_one_cycle(4, 5e-3)

preds, y = learn.get_preds(DatasetType.Test)

pr = np.array(preds)
b = (pr > 0.5) *1

sample_submission['rating'] = b
sample_submission.to_csv('submission_2.csv', index=False)
sample_submission.head(3)