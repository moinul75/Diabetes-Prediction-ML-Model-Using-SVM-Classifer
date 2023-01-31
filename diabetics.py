import numpy as np 
import pandas as pd 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time


print("\n\nReading Data form CSV....\n\n")
time.sleep(2)

df = pd.read_csv('./diabetes.csv')
print("Done...\n\n")
time.sleep(1)

# print(df.head())
# print(df.shape)
# print(df.describe())

#Data Cleaning
print("Process your Data...\n\n")
time.sleep(2) 
X = df.drop(columns='Outcome',axis=1)
Y = df['Outcome']
#Data Standardization 
sclar = StandardScaler()
standardization = sclar.fit(X)
scaler_data = sclar.transform(X)
# print(scaler_data)
#train test Data 
print("Data Processed Done...\n\n")
time.sleep(1)
print("Trained the model...\n\n")
time.sleep(2)
X_train,X_test,Y_train,Y_test = train_test_split(scaler_data,Y,test_size = 0.2,stratify= Y,random_state=1)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)
#SVM Fiting 
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
predict_data = classifier.predict(X_test)

#accurecy Check 
accurecy_check = accuracy_score(predict_data,Y_test)
# print("Accurecy: ",accurecy_check)

#now Check the Model given and input 
print("="*10+"Diabetics Prediction"+"="*10+"\n\n")
Pregnancies = float(input("Enter Pragnancies Report: "))
Glucose = float(input("Enter Glucose  Report: ")) 
BloodPressure = float(input("Enter BloodPressure Report: "))
SkinThickness = float(input("Enter SkinThickness Report: "))
Insulin = float(input("Enter Insulin Report: "))
BMI  = float(input("Enter BMI  Report: "))
DiabetesPedigreeFunction  = float(input("Enter DiabetesPedigreeFunction Report: "))       
Age = float(input("Enter Age Report: "))

print("Processing your Data \n\n")
time.sleep(2)

input_data = ( Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
#convert this to np array 
convert_np = np.asarray(input_data)
#reshaped the data 1,-1
reshaped_data = convert_np.reshape(1,-1)
#fit the data
stadard_data = sclar.transform(reshaped_data)
predict_Data = classifier.predict(stadard_data)
print("\nPredict The Result....\n\n")
time.sleep(2)

if predict_Data[0] == 1:
    print("Yes ,You Have Diabetics\n\n")
else:
    print("NO, You are Safe\n\n")
    







