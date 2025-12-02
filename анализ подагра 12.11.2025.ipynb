import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Пример данных — замените на реальный датасет с диагнозами
data = {
    'Белок': [70, 80, 76.9, 90],
    'Билирубин': [6, 7, 6.8, 8],
    'АЛТ': [20, 25, 22.9, 30],
    'АСТ': [22, 27, 24.2, 28],
    'Мочевая_кислота': [300, 400, 442, 350],
    'Холестерин': [5, 5.5, 5.35, 6],
    'Триглицериды': [1.5, 2.5, 2.38, 3],
    'Диагноз': [0, 1, 1, 0]  # 1 - подагра, 0 - нет
}

df = pd.DataFrame(data)

X = df.drop('Диагноз', axis=1)
y = df['Диагноз']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Тест на ваших данных
your_data = np.array([[76.9, 6.8, 22.9, 24.2, 442, 5.35, 2.38]])
prediction = model.predict(your_data)
print("Вероятность подагры:", "Да" if prediction[0] == 1 else "Нет")
import matplotlib 
import matplotlib.pyplot as plt

# Ваши показатели и их значения
labels = [
    'Мочевая кислота', 'Коэффициент атерогенности', 'Триглицериды',
    'Холестерин', 'Ревматоидный фактор', 'Белок', 'Билирубин',
    'АЛТ', 'АСТ'
]
values = [442, 5.2, 2.38, 5.35, 9.0, 76.9, 6.8, 22.9, 24.2]

# Определяем, какие показатели выходят за норму (условно)
risks = {
  'Мочевая кислота': 'Повышенный риск подагры и почечных камней',
    'Коэффициент атерогенности': 'Высокий риск сердечно-сосудистых заболеваний',
    'Триглицериды': 'Повышенный риск атеросклероза',
    'Холестерин': 'Повышенный риск атеросклероза',
    'Ревматоидный фактор': 'Возможное воспалительное заболевание',
    'Белок': 'В пределах нормы',
    'Билирубин': 'В пределах нормы',
    'АЛТ': 'В пределах нормы',
    'АСТ': 'В пределах нормы'
}
# Цвета: красный для рисков, зеленый для нормальных значений
colors = ['red' if key in ['Мочевая кислота', 'Коэффициент атерогенности', 'Триглицериды', 'Холестерин', 'Ревматоидный фактор'] else 'green' for key in labels]

# Построение кольцевой диаграмма
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts = ax.pie(values, labels=labels, colors=colors, startangle=90, wedgeprops=dict(width=0.3))

# Добавляем заголовок
plt.title('Анализ показателей крови и риски заболеваний')

# Выводим легенду с описанием рисков
plt.figtext(0.1, 0.1, "Риски и отклонения:\n" + "\n".join([f"{k}: {v}" for k, v in risks.items()]), fontsize=10, ha='left')

plt.show()

mport matplotlib.pyplot as plt
import pandas as pd

# Данные по рекомендуемым продуктам и их доле в рационе (условно)
recommended_products = {
    'Овощи и зелень': 35,
    'Фрукты': 20,
    'Цельнозерновые': 15,
    'Нежирные белки (рыба, птица)': 15,
    'Молочные продукты с низким содержанием жира': 10,
    'Вода': 5
}

# Круговая диаграмма рекомендуемых продуктов
labels = recommended_products.keys()
sizes = recommended_products.values()

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Рекомендуемые продукты при подагре и повышенном холестерине')
plt.axis('equal')  # Круглая диаграмма
plt.show()

# Таблица запрещённых продуктов
forbidden_products = [
      "Красное мясо (говядина, свинина)",
    "Субпродукты (печень, почки)",
    "Морепродукты (креветки, мидии, сардины)",
    "Алкоголь (особенно пиво)",
    "Газированные сладкие напитки",
    "Жирные и жареные блюда",
    "Сладости и кондитерские изделия с высоким содержанием сахара"
]

print("Категорически запрещённые продукты при подагре и повышенном холестерине:\n")
for item in forbidden_products:
    print(f" - {item}")
