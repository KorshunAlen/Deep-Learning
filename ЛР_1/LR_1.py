# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('ЛР_#1 -Логистическая_регрессия-Предсказание_ухода_клиента_из_банка.csv')

# Просмотр основных характеристик данных
print("Размер данных:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистическое описание:")
print(df.describe())
print("\nПропущенные значения:")
print(df.isnull().sum())

# Предобработка данных
# Удаление ненужных столбцов
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Кодирование категориальных переменных
label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Проверка баланса классов
print("\nРаспределение целевой переменной:")
print(df['Exited'].value_counts())
print("Доля ушедших клиентов:", df['Exited'].mean())

# Разделение на признаки и целевую переменную
X = df.drop('Exited', axis=1)
y = df['Exited']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Построение модели логистической регрессии
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Предсказания на тестовой выборке
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# Оценка качества модели
print("\n" + "="*50)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
print("="*50)

# Точность
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность (Accuracy): {accuracy:.4f}")

# AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc_roc:.4f}")

# Матрица ошибок
print("\nМатрица ошибок:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Лояльный', 'Ушедший'],
            yticklabels=['Лояльный', 'Ушедший'])
plt.title('Матрица ошибок логистической регрессии')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.show()

# Отчет классификации
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred, target_names=['Лояльный', 'Ушедший']))

# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Logistic Regression (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.grid(True)
plt.show()

# Анализ важности признаков
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_[0],
    'Absolute_Coefficient': np.abs(logreg.coef_[0])
})
feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

print("\n" + "="*50)
print("ВАЖНОСТЬ ПРИЗНАКОВ")
print("="*50)
print(feature_importance)

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Absolute_Coefficient', y='Feature', palette='viridis')
plt.title('Важность признаков в логистической регрессии')
plt.xlabel('Абсолютное значение коэффициента')
plt.ylabel('Признак')
plt.show()

# Анализ результатов
print("\n" + "="*50)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*50)
print("1. Наиболее важные признаки для предсказания оттока:")
for i, row in feature_importance.head(3).iterrows():
    direction = "положительная" if row['Coefficient'] > 0 else "отрицательная"
    print(f"   - {row['Feature']}: {direction} связь")

print("\n2. Интерпретация коэффициентов:")
print("   - Положительные коэффициенты увеличивают вероятность оттока")
print("   - Отрицательные коэффициенты уменьшают вероятность оттока")

print(f"\n3. Качество модели:")
print(f"   - Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   - AUC-ROC: {auc_roc:.4f}")

# Дополнительный анализ: пороги классификации
print("\n4. Анализ различных порогов классификации:")
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in thresholds_to_test:
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    print(f"   Порог {threshold}: точность = {custom_accuracy:.4f}")