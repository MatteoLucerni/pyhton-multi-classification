import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import seaborn as sns

digits = load_digits()

X = digits.data 
Y = digits.target

X.shape
np.unique(Y)

for i in range(10):
    pics = X[Y==i][0].reshape([8,8])
    plt.imshow(pics, cmap="gray")
    # plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# normalizzo i dati in modo da avere un range più adatto (0-255 -> 0-1 per i colori)
mms = MinMaxScaler()

X_train = mms.fit_transform(X_train)  
X_test = mms.transform(X_test)  

lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

acc = accuracy_score(Y_test, Y_pred)
lloss = log_loss(Y_test, Y_pred_proba)
conf = confusion_matrix(Y_test, Y_pred)

print(conf)

print(f'Accuracy: {acc}, Loss: {lloss}')

# analisi visiva della matrice di confusione
plt.figure(figsize=(9,9))
sns.heatmap(conf, annot=True, cmap="Blues_r", linewidths=0.5, square=True)
plt.xlabel('Corretta')
plt.ylabel('Predetta')
plt.show()

# classe dedicata OvR (stesso risultato)

from sklearn.multiclass import OneVsRestClassifier

ovr = OneVsRestClassifier()

ovr.fit(X_train, Y_train)
Y_pred = ovr.predict(X_test)
Y_pred_proba = ovr.predict_proba(X_test)

acc = accuracy_score(Y_test, Y_pred)
lloss = log_loss(Y_test, Y_pred_proba)
conf = confusion_matrix(Y_test, Y_pred)

print(conf)

print(f'Accuracy: {acc}, Loss: {lloss}')