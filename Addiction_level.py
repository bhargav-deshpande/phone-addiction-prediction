import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("teen_phone_addiction_dataset.csv")
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# sns.histplot(df['Addiction_Level'],bins=50,kde=True)
# plt.figure(figsize=(15,10))
# sns.heatmap(df.corr(numeric_only=True),annot=True,fmt=".2f")
# sns.boxplot(data=df,x='Gender',y='Addiction_Level') 
# sns.boxplot(data=df,x='School_Grade',y='Addiction_Level') 
# sns.scatterplot(data=df,x='Daily_Usage_Hours',y='Addiction_Level') 
# sns.scatterplot(data=df,x='Sleep_Hours',y='Addiction_Level')
# sns.scatterplot(data=df,x='Phone_Checks_Per_Day',y='Addiction_Level') 
# sns.scatterplot(data=df,x='Apps_Used_Daily',y='Addiction_Level') 
# sns.scatterplot(data=df,x='Time_on_Social_Media',y='Addiction_Level') 
# sns.scatterplot(data=df,x='Time_on_Gaming',y='Addiction_Level') 
# sns.boxplot(data=df,x='Phone_Usage_Purpose',y='Addiction_Level') 

x=df.drop(columns=['ID','Addiction_Level','Name','Location'])
y=df['Addiction_Level']
x_one_encoded=pd.get_dummies(x,
                           columns=['Gender','School_Grade','Phone_Usage_Purpose'],
                           drop_first=True,
                           dtype=int)

x_train, x_test, y_train, y_test = train_test_split(x_one_encoded, y, test_size=0.20, random_state=42)
x_train_lr=x_train.copy()
x_test_lr=x_test.copy()

numeric_col=['Age','Daily_Usage_Hours','Sleep_Hours','Phone_Checks_Per_Day','Apps_Used_Daily',
             'Time_on_Social_Media','Time_on_Gaming','Time_on_Education','Weekend_Usage_Hours',
             'Screen_Time_Before_Bed','Exercise_Hours','Social_Interactions','Academic_Performance',
             'Family_Communication','Anxiety_Level','Depression_Level','Self_Esteem','Parental_Control']

scaler=StandardScaler()
x_train_lr[numeric_col]=scaler.fit_transform(x_train_lr[numeric_col])
x_test_lr[numeric_col]=scaler.transform(x_test_lr[numeric_col])

model_lr=LinearRegression()
model_lr.fit(x_train_lr,y_train)
y_pred_lr=model_lr.predict(x_test_lr)
r2=r2_score(y_test,y_pred_lr)
n=x_test.shape[0]
p=x_test.shape[1]
adjusted_r2=1-((1-r2)*(n-1)/(n-p-1))
mae_lr=mean_absolute_error(y_test,y_pred_lr)
rmse_lr=np.sqrt(mean_squared_error(y_test,y_pred_lr))
print("R2: ",adjusted_r2,"\nMAE: ",mae_lr,"\nRMSE: ",rmse_lr)

model_rf=RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
model_rf.fit(x_train,y_train)
y_pred_rf=model_rf.predict(x_test)
r2_rf=r2_score(y_test,y_pred_rf)
mae=mean_absolute_error(y_test,y_pred_rf)
rmse=np.sqrt(mean_squared_error(y_test,y_pred_rf))
print("R2: ",r2_rf,"\nMAE: ",mae,"\nRMSE: ",rmse)

model_xgb=XGBRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,
                        eval_metric='rmse',random_state=42)
model_xgb.fit(x_train,y_train)
y_pred_xgb=model_xgb.predict(x_test)
r2_xgb=r2_score(y_test,y_pred_xgb)
mae_xgb=mean_absolute_error(y_test,y_pred_xgb)
rmse_xgb=np.sqrt(mean_squared_error(y_test,y_pred_xgb))
print("R2: ",r2_xgb,"\nMAE: ",mae_xgb,"\nRMSE: ",rmse_xgb)