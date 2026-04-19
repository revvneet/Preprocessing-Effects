import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('C:/Users/manve/Desktop/New model/missing values simulated/bmi.csv')

def ConvertFeatures(df):
    label_encoder = LabelEncoder()
    df['EncodedBmiClass'] = label_encoder.fit_transform(df['BmiClass'])
    print(label_encoder.classes_)
    return df, label_encoder

def InjectMissingValues(df, missing_fraction=0.2):
    df_missing = df.copy()
    np.random.seed(42)

    for col in df_missing.select_dtypes(include=[np.number]).columns:
        mask = np.random.rand(len(df_missing)) < missing_fraction
        df_missing.loc[mask, col] = np.nan

    print(f"Injected missing values into {missing_fraction*100:.0f}% of numeric features.")
    print(df_missing.isnull().sum())
    return df_missing

def TrainModel(df):
    #split it into input and output
    X = df.drop(['BmiClass', 'EncodedBmiClass'], axis=1)
    scaler = StandardScaler()
    X_Scaled = scaler.fit_transform(X)
    Y = df['EncodedBmiClass']
    sm = SMOTE(random_state=42)
    X_resampled, Y_resampled = sm.fit_resample(X_Scaled, Y)
    print("Before SMOTE:", dict(pd.Series(Y).value_counts()))
    print("After SMOTE:", dict(pd.Series(Y_resampled).value_counts()))
    #split the data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X_resampled,Y_resampled, test_size=0.2, random_state=42, stratify=Y_resampled)
    #train the model
    model = LogisticRegression(max_iter=10000, class_weight='balanced')
    model.fit(X_train, Y_train)

    #test the model
    test_predictions = model.predict(X_test)
    print("Model Evaluation: ")
    #print classification report
    print(classification_report(Y_test, test_predictions))
    
    #save the model
    joblib.dump(model, '/Users/manve/Desktop/New model/missing values simulated/trained_model.pkl')
    print("model saved successfully as \'trained_model.pkl\'")
df, label_encoder = ConvertFeatures(df)
df_missing = InjectMissingValues(df)

df_missing.to_excel("C:/Users/manve/Desktop/New model/missing values simulated/bmi_missing.xlsx", index=False)
print("Missing value dataset saved as 'bmi_missing.xlsx'")
df_missing.to_csv("C:/Users/manve/Desktop/New model/missing values simulated/bmi_missing.csv", index=False)
TrainModel(df_missing)