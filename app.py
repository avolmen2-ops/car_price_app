import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

#даем название
st.set_page_config(page_title='Прогноз цены на автомобиль',layout='centered')
st.title('Прогноз цены автомобиля')

#Предобработка(взяли из ноутбука)

#обрабатываем torque
#создаем функцию для обарботки torque
def torque_new(x):
#проверяем пропуски,и если они есть,то воззвращем 2 пропуска(сразу для обоих столбцов)
    if pd.isna(x):
        return np.nan, np.nan
#преобразуем значение в строку и приводим к нижнему регистру
    x = str(x).lower()
#ищем число в значении перед nm или kgm(шаблон:'[\d.]+(?=\s*(?:nm|kgm))' для re)
    torque_find = re.findall(r'[\d.]+(?=\s*(?:nm|kgm))', x)
#если torque_find не является пустым,то мы берем 0 элемент(само число) и преобразуем его во float
    torque = float(torque_find[0]) if torque_find else np.nan
#Формула: 1 kgm = 9.8 nm. если 'kgm' была в x(в строке) и значение не пустое, то переводим kgm в nm,умножая на 9,8
    if 'kgm' in x and not pd.isna(torque):
        torque*=9.8
#делим строку по @ и берем последнюю часть, и в ней ищем числа по шаблону('[\d]+')
    rpm_find = re.findall(r'[\d]+', x.split('@')[-1])
#если нашли числа,то проходимся по каждому числу(если их больше 1) и добавляем в список(преобразуя в целое)
    if rpm_find:
        rpm_list = []
        for i in rpm_find:
            rpm_list.append(int(i))
#чтобы найти rpm скалдываем числа из списка и делим на длину списка(находим среднее)
        rpm = sum(rpm_list)/len(rpm_list)
#если чисел не нашли,то возвращаем np.nan
    else:
        rpm = np.nan
#в итоге возвращаем 2 числа torque и rpm
    return torque, rpm

#собираем вместе нашу предобработку данных
def prepare_data(df):
#копируем датасет
    df = df.copy()
#убираем дубликаты,если есть целевая переменная, обновляем индексы, чтобы они шли от 0 без пропусков
    if 'selling_price' in df.columns:
        df = df.drop_duplicates(subset=df.drop(columns=['selling_price']).columns,keep='first').reset_index(drop=True)
#вставляем нашу функцию,которая достает числа, немного ее улучшили в конце
    def extract_n(x):
        if pd.isnull(x):
            return np.nan
        nums = re.findall(r"[\d.]+", str(x))
        return float(nums[0]) if nums else np.nan
#применяем функцию к необходимым столбцам
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(extract_n)
#из 4 шага выбрали признак, который больше всего улучшил модель(повышаем качество нашей модели)
    current_year = 2026
    df['age'] = current_year - df['year']
    df['power_age'] = df['max_power']*df['age']
#применяем нашу функцию torque_new(x), которую расписали в самом начале, к необходимым столбцам(делим столбец на 2 столбца)
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(lambda x: pd.Series(torque_new(x)))
#работаем с колонкой name, достаем из нее бренд
    df['name'] = df['name'].apply(lambda x: x.split()[0])
#приводим столбцы к типу float
    df['engine'] = df['engine'].astype(float)
    df['seats'] = df['seats'].astype(float)
    return df

#Загружаем модель

#декоратор streamlit, который загрузит модель и будет хранить в памяти,чтобы быстрее работало
@st.cache_resource
#загружаем модель, которую заранее скачали, и используем ее
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

#Загружаем csv

uploaded_file = st.file_uploader("Загрузите CSV-файл",type=["csv"])
#проверяем загружен ли файл, читаем его
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
#показываем наши данные
    st.subheader("Наши данные")
    st.dataframe(data.head())
#показываем размер наших данных
    st.subheader("Размер данных")
    st.write(data.shape)

#EDA
#вставляем графики из ноутбука,но проверяем сначала, чтобы признак был в столбцах

    if 'selling_price' in data.columns:
        st.subheader("Распределение цены")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data['selling_price'], bins=50, ax=ax)
        ax.set_title("Распределение цены")
#чтобы графики не были огромными
        st.pyplot(fig, use_container_width=False)

    if 'year' in data.columns:
        st.subheader("Распределение по году выпуска автомобиля")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data['year'],bins=30,ax=ax)
        ax.set_title("Распределение по году выпуска")
        ax.set_xlabel("Год")
        ax.set_ylabel("Количество")
        st.pyplot(fig, use_container_width=False)

    if 'fuel' in data.columns:
        st.subheader("Количество автомобилей по типу топлива")
        fuel_counts = data['fuel'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(fuel_counts.index, fuel_counts.values)
        ax.set_title("Количество автомобилей по типу топлива")
        ax.set_xlabel("Тип топлива")
        ax.set_ylabel("Количество")
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=False)

    if 'year' in data.columns and 'selling_price' in data.columns:
        st.subheader("Средняя цена по году выпуска")
        price_year = data.groupby('year')['selling_price'].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(price_year.index, price_year.values, marker='o')
        ax.set_title("Средняя цена по году выпуска")
        ax.set_xlabel("Год")
        ax.set_ylabel("Средняя цена")
        st.pyplot(fig, use_container_width=False)
#берем только 5 признаков, иначе график слишком большой и его трудно воспринимать
    num_columns = data.select_dtypes(include=['float64', 'int64']).columns[:5]
    if len(num_columns) > 1:
        fig = sns.pairplot(data[num_columns])
        st.pyplot(fig, use_container_width=False)

    st.subheader("Тепловая карта корреляций Пирсона")
    corr_pirs = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_pirs,cmap="coolwarm",annot=True,center=0,ax=ax)
    ax.set_title("Тепловая карта корреляций")
    st.pyplot(fig, use_container_width=False)

#Предсказание

    st.header("Предсказание цены")
    data_for_prediction =data.copy()
#убираем цену,если она есть
    if 'selling_price' in data_for_prediction.columns:
        data_for_prediction = data_for_prediction.drop(columns=['selling_price'])
#применяем предобработку по нашей функции
    data_for_prediction=prepare_data(data_for_prediction)
#предсказываем
    predictions = model.predict(data_for_prediction)
    result = data.copy()
    result['predicted_price'] = predictions
    st.subheader("Предсказание")
    st.dataframe(result.head(20))

#Визуализация весов модели

    st.header("Веса обученной модели")
#достаем препроцессор и модель
    preprocessor = model.named_steps['preprocessor']
    ridge_model = model.named_steps['model']
#получаем названия всех признаков
    feature_names = preprocessor.get_feature_names_out()
#достаем коэффициенты(веса)
    coefficients = ridge_model.coef_
#создаем таблицу с признаками и весами
    coef_df = pd.DataFrame({'feature': feature_names,'coefficient': coefficients})
#достаем именно абсолютные значения весов
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
#сортируем в порядке убывания
    coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
#выводим топ признаков
    st.subheader("Первые 20 признаков по влиянию")
    st.dataframe(coef_df.head(20))
#сохраняем топ 20 коэффициентов
    top_coef = coef_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_coef['feature'],top_coef['coefficient'])
    ax.set_title("Первые 20 коэффициентов")
    ax.set_xlabel("Коэффициент")
    ax.set_ylabel("Признак")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=False)
#выделяем отдельно числовые признаки(берем все те,что не начинаются с cat)
    num_coef = coef_df[~coef_df['feature'].str.startswith('cat_')]
    st.subheader("Числовые признаки по влиянию")
    num_coef_sorted = num_coef.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    #выводим числовые признаки
    st.dataframe(num_coef_sorted)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(num_coef_sorted['feature'], num_coef_sorted['coefficient'])
    ax.set_title("Коэффициенты числовых признаков")
    ax.set_xlabel("Коэффициент")
    ax.set_ylabel("Признак")
    st.pyplot(fig, use_container_width=False)

