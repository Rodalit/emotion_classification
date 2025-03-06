# Ссылка на датасет: https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset/data

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Загрузка данных
df = pd.read_csv('emotions.csv')

# Проверка на пропущенные значения и дубликаты
print(df.isnull().sum())
print(f"Number of duplicates: {df.duplicated().sum()}")

# Удаление дубликатов
df = df.drop_duplicates()

# Разделение на признаки
X = df['text']
Y = df['label']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Создание конвеера
pipeline = Pipeline([
	('Vectorizer', TfidfVectorizer(stop_words='english')),
	('Model', SGDClassifier())
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Делаем предсказание на тестовых данных
y_pred = pipeline.predict(X_test)

# Оцениваем производительность модели на тестовой выборке
print(f"Report: {classification_report(y_test, y_pred)}")

# Визуализация матрицы ошибок
fig, ax = plt.subplots(figsize=(10, 7))
ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, display_labels=pipeline.classes_, cmap='viridis', ax=ax)
plt.title('Confusion Matrix')
plt.show()