from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("spam.csv", encoding="latin-1") 
	#讀取垃圾訊息的文本
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	#將多餘的Unnamed:colume刪除
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	#設定標籤，將垃圾郵件標為1,非垃圾文件標為0
	X = df['message']
	y = df['label']

	cv = CountVectorizer()
	X = cv.fit_transform(X) 
	#拟合模型，并返回文本矩阵
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#train_data：所要划分的样本特征集,train_target：所要划分的样本结果,test_size：样本占比，如果是整数的话就是样本的数量,random_state：是随机数的种子
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	#MultinomialNB, 其適用特徵是離散型資料, 如出現次數;GaussianNB, 其適用特徵是連續型資料, 是常態分配, 如身高, 體重, 成績.
	clf.fit(X_train,y_train)
	#拟合数据
	clf.score(X_test,y_test)
	#查看準確度


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)