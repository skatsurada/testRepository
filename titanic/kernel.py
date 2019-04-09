# ブランチ１を切った後にmastae　での変更


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.5f' % x)

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10, gamma=0.1)

#データファイルからdataをデータフレームとして読み込み
data_test  = pd.read_csv('data/test.csv' , delimiter=',')
data_train = pd.read_csv('data/train.csv', delimiter=',')

#データのサイズ確認
print('データのサイズ(test, train)')
print(data_test.shape, data_train.shape,"\n")

#データの中身確認
print(data_train.head(3))

#データの統計量出力
print('統計量')
print(data_train.describe(include='all').transpose())

#欠損値の処理
print(data_train.isnull().sum())
train = data_train.dropna(subset=['Age'])
test  = data_test
print(train.isnull().sum())

#x 予測に用いるデータ  y ターゲット
x_train, y_train = train.drop(['PassengerId','Survived','Name','Sex','Ticket','Cabin','Embarked'],axis=1), train.Survived
x_test           = test.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)

# モデルの作成
## ランダムフォレスト
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)

## SVM
clf.fit(x_train, y_train)

# スコアを表示
print(m.score(x_train, y_train))
print(clf.score(x_train, y_train))

print('予測値　年齢にヌルあり')
print(x_test.isnull().sum())
x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())
x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].mean())
print('予測値　年齢にヌル無し')
print(x_test.isnull().sum())
print(x_test.head(3))

# 予測実行
m_pre = m.predict(x_test)
clf_pre = clf.predict(x_test)

# 結果出力
#print('ランダムフォレスト',m_pre)
#print('SVM',clf_pre)

out_data = pd.concat([test['PassengerId'], pd.DataFrame(clf_pre)], axis=1, join='inner')
out_data.columns = ['PassengerId', 'Survived']
out_data.to_csv('submission.csv', index=False, header=True)

#git test用変更
#branch1　開発中
#branch1 開発中2回目変更