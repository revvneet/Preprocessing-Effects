import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer

df = pd.read_csv('C:/Users/manve/Desktop/New model/missing values simulated/bmi_missing.csv')

def HandleMissingValues(df_missing):
    numeric_cols = ['Age', 'Height', 'Weight', 'Bmi']
    
    imputer = KNNImputer(n_neighbors=5)
    df_missing[numeric_cols] = imputer.fit_transform(df_missing[numeric_cols])
    
    # For categorical (encoded) class column
    if df_missing['EncodedBmiClass'].isnull().sum() > 0:
        mode_val = df_missing['EncodedBmiClass'].mode()[0]
        df_missing['EncodedBmiClass'] = df_missing['EncodedBmiClass'].fillna(mode_val)
    
    return df_missing

def TrainModel(df):
    df = df.dropna()
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
df_fixed = HandleMissingValues(df)
TrainModel(df_fixed)