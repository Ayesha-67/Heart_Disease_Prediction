import numpy as np
import pandas as pd

heart_df = pd.read_csv("framingham.csv")

#Replacing null values with mean value
heart_df["education"].fillna(heart_df["education"].mode()[0],inplace=True)
heart_df["cigsPerDay"].fillna(heart_df["cigsPerDay"].mean(), inplace=True)
heart_df["BPMeds"].fillna(heart_df["BPMeds"].mode()[0],inplace=True)
heart_df["totChol"].fillna(heart_df["totChol"].mean(), inplace=True)
heart_df["BMI"].fillna(heart_df["BMI"].mean(),inplace=True)
heart_df["heartRate"].fillna(heart_df["heartRate"].mean(),inplace=True)
heart_df["glucose"].fillna(heart_df["glucose"].mean(),inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#defining variables x,y
x= heart_df.drop("TenYearCHD",axis=1)
y = heart_df["TenYearCHD"]


print("columns in x:", x.columns)
print("y:",y)

print("shape of x:", x.shape)
print("Shape of y:", y.shape[0])

col_names = heart_df.columns


for c in col_names:
    heart_df = heart_df.replace("?", np.NaN)
heart_df = heart_df.apply(lambda x: x.fillna(x.value_counts().index[0]))

category_col = [ i for i in heart_df.columns if heart_df[i].dtypes=='object']

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
mapping_dict = {}
for col in category_col:
    heart_df[col] = labelencoder.fit_transform(heart_df[col])
    
    le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    mapping_dict[col] = le_name_mapping
print(mapping_dict)

#splitting data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)

from sklearn.utils import resample

# Upsample minority class
df_majority = heart_df[heart_df.TenYearCHD == 0]
df_minority = heart_df[heart_df.TenYearCHD == 1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,   
                                 n_samples=len(df_majority), 
                                 random_state=42)

heart_df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Train-test split
x = heart_df_balanced.drop("TenYearCHD", axis=1)
y = heart_df_balanced["TenYearCHD"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


#scaling the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train = pd.DataFrame(x_train,columns=x.columns)
x_test = pd.DataFrame(x_test, columns=x.columns)

#training the model using Logistic Regression
regressor = LogisticRegression()
regressor.fit(x_train,y_train)

#save trained model
import pickle
file_name = 'heart_prediction.pkl'
with open(file_name, 'wb') as f:
    pickle.dump(regressor, f)
    
# Save the scaler  
scaler_filename = "scaler.pkl"
with open(scaler_filename, 'wb') as f:
    pickle.dump(sc, f)

#load the model to verify
heart_prediction = pickle.load(open('heart_prediction.pkl','rb'))

print("Model training complete & saved succesfully!")