import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Загрузка данных
data = pd.read_csv('train.csv', encoding_errors='ignore')

# Определение входных и выходных переменных
X = data['comment']
Y = data[["TOXICITY", "SEVERE_TOXICITY", "SEXUALLY_EXPLICIT", "THREAT", "INSULT", "IDENTITY_ATTACK"]]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Создание пайплайна с TfidfVectorizer и MultiOutputClassifier с внутренним LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(LogisticRegression()))
])

# Обучение модели
pipeline.fit(X_train, Y_train)

# Оценка модели
Y_pred = pipeline.predict(X_test)

# Поскольку у нас есть несколько меток, мы вычислим точность для каждой
for i, label in enumerate(Y.columns):
    accuracy = accuracy_score(Y_test[label], Y_pred[:, i])
    print(f'Accuracy for {label}: {accuracy}')

# Сохранение обученной модели
joblib.dump(pipeline, 'multilabel_toxicity_model.pkl')

print("The multilabel model has been saved.")
