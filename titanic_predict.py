import pandas as pd
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('/workspaces/titanic_prediction/dataset/train.csv')

# data.info() # 데이터 칼럼 설명

# print(data.describe())
# print("\n",data.isnull().sum())
data.isnull().sum().plot.barh(figsize=(10, 9))  # 결측치 가로 막대 그래프

# Age의 결측값은 평균으로 채우기
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Embarked의 결측값은 최빈값으로 채우기
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Cabin의 결측값은 많기에 삭제
data.drop(['Cabin'], axis=1, inplace=True)

# 범주형 변수 get_dummies로 변경 - 이렇게 숫자형으로 변경해야 머신러닝이 처리 가능
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# features와 targets 나누기
X = data[['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']

# 학습용/검증용 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_predict = model.predict(X_test)

# 결과 평가
print(classification_report(y_test, y_predict))
print("Accuracy :", accuracy_score(y_test, y_predict))

# 랜덤 포레스트 모델
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# 튜닝할 파라미터
param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [5, 10, 15],
    'min_samples_split' : [2, 5, 10]
}

# GridSearchCV 실행
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv = 5, scoring='f1')
grid_search.fit(X_train, y_train)

# 최적의 모델 출력
print("최적 파라미터 :", grid_search.best_params_)

# 테스트 데이터로 성능 평가
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# 모델 평가
print("\n 튜닝된 모델 평가")
print(classification_report(y_test, y_pred_rf))
print("Accuracy :", accuracy_score(y_test, y_pred_rf))

rf_model = RandomForestClassifier(n_estimators=5, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
prediction = rf_model.predict(X)
print(classification_report(y, prediction))
print("Accuracy :", accuracy_score(y, prediction))

# 예측값 (로지스틱)
rg_cm = confusion_matrix(y_test, y_predict)

# 예측값 (랜덤 포레스트)
rf_cm = confusion_matrix(y, prediction)

# 시각화
#rg_disp = ConfusionMatrixDisplay(confusion_matrix=rg_cm, display_labels=[0, 1]) # 로지스틱 
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=[0, 1]) # 랜덤 포레스트

#rg_disp.plot(cmap='Blues')
rf_disp.plot(cmap='Reds')

plt.title("Confusion Matrix")
# plt.savefig("Rogistic")
plt.savefig("Random Forest")