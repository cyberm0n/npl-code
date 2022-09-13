import time
start_time = time.time()

import pandas as pd
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


asc = "i love you"
data_frame = pd.read_csv("dataset2.csv",index_col=False)

data_frame['Clean_Text'] = data_frame['Text'].apply(nfx.remove_userhandles)
data_frame['Clean_Text'] = data_frame['Clean_Text'].apply(nfx.remove_stopwords)
Xfeatures = data_frame['Clean_Text']
ylabels = data_frame['Emotion']

x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)

from sklearn.pipeline import Pipeline
pipe_lr = Pipeline(steps=[('lr',LogisticRegression(n_jobs=-1))])
pipe_lr.fit(x_train,y_train)

print(model.predict([asc])[0])
print("--- %s seconds ---" % (time.time() - start_time))
