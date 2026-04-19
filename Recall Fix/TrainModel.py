import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/manve/Desktop/New model/Recall Fix/bmi.csv')

def ConvertFeatures(df):
    label_encoder = LabelEncoder()
    df['EncodedBmiClass'] = label_encoder.fit_transform(df['BmiClass'])
    print(label_encoder.classes_)
    return df, label_encoder

def TrainModel(df):
    #split it into input and output
    X = df.drop(['EncodedBmiClass', 'BmiClass'], axis=1)
    Y = df['EncodedBmiClass']
    #split the data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42, stratify=Y)
    #train the model
    model = LogisticRegression(max_iter=10000, class_weight='balanced')
    model.fit(X_train, Y_train)

    #test the model
    test_predictions = model.predict(X_test)
    print("Model Evaluation: ")
    #print classification report
    print(classification_report(Y_test, test_predictions))
    
    #save the model
    joblib.dump(model, '/Users/manve/Desktop/New model/Recall Fix/trained_model.pkl')
    print("model saved successfully as \'trained_model.pkl\'")
df, label_encoder = ConvertFeatures(df)
TrainModel(df)