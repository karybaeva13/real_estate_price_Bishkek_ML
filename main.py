
''' # Preprocessing '''

''' ## Импорт данных '''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")

sns.set(style="darkgrid")


df = pd.read_csv('train.csv')



'''## Подготовка данных'''

'''### Форматирование данных и удаление неполных признаков'''

class ApartmentPreprocessor:

    def __init__(self):
        self.drop_columns = [
            'address',
            'added',
            'upped',
            'Тип предложения'
        ]


    def clean_floor(self, df):
        df['Этаж'] = (df['Этаж'].str.split().str[0]
            .replace({
                'цоколь': '0.5',
                'подвал': '0'
            }))
        return df


    def clean_area(self, df):
        df['Площадь'] = (df['Площадь'].str.split('м2').str[0].str.strip().astype(float))
        return df


    def clean_rooms(self, df):

        df['main'] = df['main'].str.split('-комн').str[0]
        df.loc[df['main'].str.startswith('6 и более комнат кв.', na=False),'main'] = '6'

        mask = df['main'].str.startswith('свободная планировка',na=False)

        area = df.loc[mask, 'Площадь']

        conditions = [
            (area > 20) & (area < 40),
            (area >= 40) & (area < 70),
            (area >= 70) & (area < 80),
            (area >= 80) & (area < 120),
            (area >= 120) & (area < 180),
            (area >= 180)]

        values = ['1', '2', '3', '4', '5', '6']

        df.loc[mask, 'main'] = np.select(
            conditions,
            values,
            default=df.loc[mask, 'main'])

        df.rename(columns={'main': 'Комнаты'}, inplace=True)
        return df


    def clean_building_info(self, df):

        df['Год'] = (df['Дом'].str.split().str[1].str.strip())
        df.loc[df['Год'].isin(['0', '2']),'Год'] = 'NotGiven'

        df['Дом'] = (df['Дом'].str.split(',').str[0].str.strip())
        return df


    def drop_unused_columns(self, df):

        df.dropna(thresh=int(round(df.shape[0] * 0.7)), axis=1, inplace=True)
        df.drop(columns=self.drop_columns, inplace=True, errors='ignore')
        return df


    def transform(self, df):

        df = self.clean_floor(df)
        df = self.clean_area(df)
        df = self.clean_rooms(df)
        df = self.clean_building_info(df)
        df = self.drop_unused_columns(df)
        return df

prep = ApartmentPreprocessor()
df = prep.transform(df)


mask = ((df.lat > 42.7) & (df.lat < 43.0)) & ((df.lon > 74.3) & (df.lon <  74.8))

df = df[mask]



''' ## Отбор полезных признаков 

Проверка: разделяют ли признаки данные на отдельные группы (делают ли они их более различимыми) '''

''' ### квартили, выбросы '''

fig, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.violinplot(data=df, x='Дом', y='usd_price', ax=ax[0])
sns.violinplot(data=df, x='Состояние', y='usd_price', ax=ax[2])
ax[1].tick_params(axis='x', rotation=45)
sns.violinplot(data=df, x='Отопление', y='usd_price', ax=ax[1])
ax[2].tick_params(axis='x', rotation=45)

'''### распределение'''

fig, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.kdeplot(data=df, x='usd_price', hue='Дом', fill=True, common_norm=False, ax=ax[0])
sns.kdeplot(data=df, x='usd_price', hue='Состояние', fill=True, common_norm=False, ax=ax[2])
sns.kdeplot(data=df, x='usd_price', hue='Отопление', fill=True, common_norm=False, ax=ax[1])

'''### цены различных категорий признака при равной площади'''

fig, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.scatterplot(data=df, x='Площадь', y='usd_price', hue='Дом', alpha=0.6, ax=ax[0])
sns.scatterplot(data=df, x='Площадь', y='usd_price', hue='Состояние', alpha=0.6, ax=ax[2])
sns.scatterplot(data=df, x='Площадь', y='usd_price', hue='Отопление', alpha=0.6, ax=ax[1])



''' ## Удаление незначимых признаков '''

df.drop(['Дом', 'Отопление', 'Год', 'view_count'], axis=1, inplace=True)


'''### Преобразование категориальных признаков'''

'''Состояние'''

from sklearn.preprocessing import OrdinalEncoder

order = ['не достроено', 'под самоотделку (псо)', 'среднее', 'хорошее', 'евроремонт']

enc = OrdinalEncoder(
    categories=[order],
    encoded_missing_value=np.nan,
    handle_unknown='use_encoded_value',
    unknown_value=np.nan
)

df[['Состояние']] = enc.fit_transform(df[['Состояние']])


'''Серия'''

plt.figure(figsize=(20, 6))
sns.boxplot(data=df, x='Серия', y='usd_price')
plt.xticks(rotation=45)

pd.DataFrame(df.groupby(['Серия'])['usd_price'].median().sort_values())

seria_mapping = {
    'малосемейка': 1,
    'хрущевка': 2,
    '105 серия улучшенная': 2,
    '108 серия': 3,
    '106 серия улучшенная': 3,
    '104 серия': 3,
    '104 серия улучшенная': 3,
    '105 серия': 3,
    'сталинка': 4,
    '106 серия': 4,
    'индивид. планировка': 4,
    '107 серия': 4,
    'элитка': 5,
    'пентхаус': 6
}

df['Серия'] = df['Серия'].map(seria_mapping)


'''### Заполнение пропусков'''

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

X = df.drop(columns=['usd_price'])
y = df['usd_price']

# сплит
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = IterativeImputer(max_iter=10, random_state=42)

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns, index=X_train.index
)
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns, index=X_test.index
)

# print(X_train.index.equals(y_train.index))
# print(X_test.index.equals(y_test.index))

X_train['Этаж'] = X_train['Этаж'].astype(int)
X_train['Комнаты'] = X_train['Комнаты'].astype(int)


'''# Линейная регрессия'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso


'''l2 регуляризация'''
from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('regression', Ridge())
])

model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("TEST R2:\t %.4f" % r2_score(y_test, y_pred))
print("TEST MAPE:\t %.4f" % mean_absolute_percentage_error(y_test, y_pred))
print('-----------------')
print("TEST RMSE:\t %.4f" % mean_squared_error(y_test, y_pred)**0.5)
print("TEST MAE:\t %.4f" % mean_absolute_error(y_test, y_pred))


model['regression'].coef_


'''l1 регуляризация (зануляем коэффициенты)'''
'''подбор best alpha'''

lasso_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('regression', Lasso())
])

alphas = np.logspace(-2, 5, 20)

param_dict = {"regression__alpha": alphas}
searcher = GridSearchCV(lasso_pipeline, param_dict,
                        scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
searcher.fit(X_train, y_train)

best_alpha = searcher.best_params_["regression__alpha"]
print("Best alpha = %.4f" % best_alpha)

plt.plot(alphas, -searcher.cv_results_["mean_test_score"])
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("CV score")

lasso_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('regression', Lasso(best_alpha))
])

model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("TEST R2:\t %.4f" % r2_score(y_test, y_pred))
print("TEST MAPE:\t %.4f" % mean_absolute_percentage_error(y_test, y_pred))
print('-----------------')
print("TEST RMSE:\t %.4f" % mean_squared_error(y_test, y_pred)**.5)
print("TEST MAE:\t %.4f" % mean_absolute_error(y_test, y_pred))

model['regression'].coef_


'''распределение остатков'''

error = (y_train - model.predict(X_train)) ** 2
sns.distplot(error, color='black')

"""убираем большие ошиибки"""

mask = (error < np.quantile(error, 0.95))

lasso_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('regression', Lasso(best_alpha))
])

model = pipeline.fit(X_train[mask], y_train[mask])
y_pred = model.predict(X_test)

print("TEST R2:\t %.4f" % r2_score(y_test, y_pred))
print("TEST MAPE:\t %.4f" % mean_absolute_percentage_error(y_test, y_pred))
print('-----------------')
print("TEST RMSE:\t %.4f" % mean_squared_error(y_test, y_pred)**.5)
print("TEST MAE:\t %.4f" % mean_absolute_error(y_test, y_pred))


'''# Градиент (SGD)'''

from sklearn.linear_model import SGDRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")

sns.set(style="darkgrid")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import Pipeline


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

'''## Поиск параметров'''

model = SGDRegressor(loss='huber')

param_grid = {'penalty': ['l2', 'l1'], # тип регуляризации
              'alpha': np.logspace(-5, 1, 15), # коэффициент регуляризации
              'eta0': [.15, .2, .25], # начальный learning rate
              'power_t': [.4, .5, .6], # параметр уменьшения learning rate
               'epsilon': [200, 250, 300]} # задаёт порог: где MSE превращается в MAE-подобное поведение (только для Huber loss)

searcher = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", cv=4, verbose=2) # verbose - уровень подробностии вывода данных в консоль
searcher.fit(X_train, y_train)

searcher.best_params_

searcher.best_score_


'''## Обучение модели'''
SGD = SGDRegressor(loss='huber', alpha = 0.0005179474679231213, epsilon = 200, eta0 = 0.15, penalty = 'l1', power_t = 0.4)

model = Pipeline([('scaling', StandardScaler()),
                  ('regressor', SGD)])

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

print("TEST R2:\t %.4f" % r2_score(y_test, y_pred))
print("TEST MAPE:\t %.4f" % mean_absolute_percentage_error(y_test, y_pred))
print('-----------------')
print("TEST RMSE:\t %.4f" % mean_squared_error(y_test, y_pred))
print("TEST MAE:\t %.4f" % mean_absolute_error(y_test, y_pred))

print("TRAIN R2:\t %.4f" % r2_score(y_train, y_train_pred))
print("TRAIN MAPE:\t %.4f" % mean_absolute_percentage_error(y_train, y_train_pred))
print('-----------------')
print("TRAIN RMSE:\t %.4f" % mean_squared_error(y_train, y_train_pred))
print("TRAIN MAE:\t %.4f" % mean_absolute_error(y_train, y_train_pred))


'''## Ошибки'''

sns.boxplot(y_train_pred - y_train, color='black')

fig, ax = plt.subplots(1, 2, figsize=(15, 8))

sns.scatterplot(x=y_train, y=y_train_pred - y_train, color='black', ax=ax[0])
ax[0].set_ylabel('ERROR')

sns.scatterplot(x=X_train.Площадь, y=y_train_pred - y_train, color='black', ax=ax[1])
ax[1].set_ylabel('ERROR')

mask = y_train_pred - y_train > 300000
X_train_clean = X_train[~mask]
y_train_clean = y_train[~mask]

model.fit(X_train_clean, y_train_clean)
y_train_pred = model.predict(X_train_clean)
y_pred = model.predict(X_test)

print("TEST R2:\t %.4f" % r2_score(y_test, y_pred))
print("TEST MAPE:\t %.4f" % mean_absolute_percentage_error(y_test, y_pred))
print('-----------------')
print("TEST RMSE:\t %.4f" % mean_squared_error(y_test, y_pred)**.5)
print("TEST MAE:\t %.4f" % mean_absolute_error(y_test, y_pred))
print()

print("TRAIN R2:\t %.4f" % r2_score(y_train_clean, y_train_pred))
print("TRAIN MAPE:\t %.4f" % mean_absolute_percentage_error(y_train_clean, y_train_pred))
print('-----------------')
print("TRAIN RMSE:\t %.4f" % mean_squared_error(y_train_clean, y_train_pred)**.5)
print("TRAIN MAE:\t %.4f" % mean_absolute_error(y_train_clean, y_train_pred))

sns.distplot(model['regressor'].coef_)


'''# Random forest'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV


'''подбор гиперпараметров с GridSearch'''

kf = KFold(n_splits=5, shuffle=True, random_state=42)

parameters = {'max_features': [2, 3, 4],
              'min_samples_leaf': [1, 3, 5],
              'max_depth': [10, 15, 20, 25]}

rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
gcv = GridSearchCV(rfr, parameters, n_jobs=-1, cv=kf, scoring='neg_mean_absolute_percentage_error')
gcv.fit(X_train, y_train)

print("Лучшие параметры:", gcv.best_params_)
print("Лучшее CV MAPE: {:.2f}".format(-gcv.best_score_))

'''подбор гиперпараметров с Optuna'''

import optuna
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

    # Кросс-валидация по метрике MAPE
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1
    )

    return scores.mean()


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Лучшие параметры:", study.best_params)
print("Лучший CV MAPE:", -study.best_value)

best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("Test MAPE:", mean_absolute_percentage_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))

from sklearn.metrics import r2_score, mean_absolute_percentage_error

'''график скора'''
df_trials_rf = study.trials_dataframe()

plt.figure(figsize=(8, 5))
plt.plot(df_trials_rf['number'], df_trials_rf['value']*(-1), marker='o', markersize=3)
plt.xlabel('Номер trial')
plt.ylabel('MAPE')
plt.title('Изменение скоринга по мере поиска гиперпараметров для RandomForest')
plt.tight_layout()
plt.show()

'''важность признаков'''
forest = RandomForestRegressor(n_estimators=443, max_depth=21, min_samples_split=2, min_samples_leaf=1, max_features = 0.3183850552445241, bootstrap=False,
                             random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
importances = pd.Series(forest.feature_importances_,
                        index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
importances.plot.barh(ax=ax)
ax.set_title("Важность признаков (RandomForest)")
ax.set_xlabel("важность")
plt.tight_layout()

'''# Бустинг'''

X_train['Серия'] = X_train['Серия'].astype(int).astype(str)
X_test['Серия'] = X_test['Серия'].astype(int).astype(str)

X_train['Состояние'] = X_train['Состояние'].astype(int).astype(str)
X_test['Состояние'] = X_test['Состояние'].astype(int).astype(str)

X_train['Этаж'] = X_train['Этаж'].astype(int).astype(str)
X_test['Этаж'] = X_test['Этаж'].astype(int).astype(str)

X_train['Комнаты'] = X_train['Комнаты'].astype(int).astype(str)
X_test['Комнаты'] = X_test['Комнаты'].astype(int).astype(str)

from catboost import CatBoostRegressor


'''Catboost из коробки'''

model = CatBoostRegressor(
    loss_function='RMSE',
    cat_features=['Состояние', 'Этаж', 'Комнаты', 'Серия'],
    verbose=100
)

model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_percentage_error

y_pred = model.predict(X_test)


mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE без подбора гиперпараметров: {mape:.4f}")


'''Catboost + optuna'''

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import optuna

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'loss_function': 'RMSE',
        'eval_metric': 'MAPE',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50,
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        cat_features=['Состояние', 'Этаж', 'Комнаты', 'Серия']
    )

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)  # оптимизируем именно MAPE
    return mape

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Лучшие параметры:", study.best_params)
print("Лучший MAPE:", study.best_value)

best_params = study.best_params
best_params.update({
    'loss_function': 'RMSE',
    'eval_metric': 'MAPE',
    'random_seed': 42,
    'verbose': False,
})

best_model = CatBoostRegressor(**best_params)
best_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

'''Графики'''
df_trials_cb = study.trials_dataframe()
df_trials_cb.head()

plt.figure(figsize=(8, 5))
plt.plot(df_trials_cb['number'], df_trials_cb['value'], marker='o', markersize=3)
plt.xlabel('Номер trial')
plt.ylabel('MAPE')
plt.title('Изменение скоринга по мере поиска гиперпараметров для Catboost')
plt.tight_layout()
plt.show()

'''Графики Optuna visualization'''
import optuna.visualization as vis

# 1. Как менялся скоринг по трайлам (с отметкой лучшего значения на каждый момент)
vis.plot_optimization_history(study).show()

# 2. Как выглядит "важность" каждого гиперпараметра для итогового скоринга
vis.plot_param_importances(study).show()

# 3. Как менялся конкретный гиперпараметр в зависимости от номера trial
vis.plot_slice(study, params=['learning_rate', 'depth']).show()

# 4. Параллельные координаты — сразу видно, какие сочетания параметров дают лучший скор
vis.plot_parallel_coordinate(study).show()

# 5. Контурный график для пары параметров (где скор выше — теплее цвет)
vis.plot_contour(study, params=['learning_rate', 'depth']).show()


'''Mape на выборке test (которая была выделена из файла train.csv)'''
preds = best_model.predict(X_test)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
print("Test MAPE:", mean_absolute_percentage_error(y_test, preds))
print("Test R2:", r2_score(y_test, preds))

'''Важность признаков (на сколько упадет скор если перемешать признаки)'''

importances = model.get_feature_importance()
feature_names = X_train.columns

order = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.barh(np.array(feature_names)[order], importances[order])
plt.xlabel('Важность признака')
plt.title('Важность признаков — CatBoostRegressor')
plt.tight_layout()
plt.show()


'''# Сохранение модели для Hugging Face'''

best_model.save_model('model.cbm')

print(f'Признаки модели: {best_model.feature_names_}')
