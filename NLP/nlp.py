import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#quoting=3 ignores the qutes | delimiter is that splitting the data 
df = pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

# re is using to simplfy 
import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(1000):
    review= re.sub('[^a-zA-z]',' ',df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review= [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# print(corpus[2])

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X= cv.fit_transform(corpus).toarray()

Y= df.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
gnb=GaussianNB()
print("Gaussian NB is training")
gnb.fit(x_train,y_train)
# making predictions
y_pred_gnb=gnb.predict(x_test)
print(confusion_matrix(y_test,y_pred_gnb))
print(accuracy_score(y_test,y_pred_gnb))
