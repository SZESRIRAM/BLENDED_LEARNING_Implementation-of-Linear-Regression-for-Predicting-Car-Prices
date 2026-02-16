# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Loading the Dataset
df=pd.read_csv("CarPrice_Assignment.csv")
df.head()

x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

#Split Data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Feature Scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#Train Model
model=LinearRegression()
model.fit(x_train_scaled,y_train)

#Prediciton

y_pred=model.predict(x_test_scaled)

print("Name:Ponsriram P")
print("Reg.No:25011113")
print("Model Coefficients:")
for feature, coef in zip(x.columns,model.coef_):
    print(f"{feature}: {coef}")
print(f"('Intercept'): {model.intercept_}")

print("\nModel Performance:")
print(f"{'MSE'}: {mean_squared_error(y_test,y_pred)}")
print(f"{'RMSE'}: {np.sqrt(mean_squared_error(y_test,y_pred))}")
print(f"{'R-Squared'}: {r2_score(y_test,y_pred)}")
print(f"{'MAE'}: {mean_absolute_error(y_test,y_pred)}")

#1.Linearity Check
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

#2. Independence (Durbin-Watson)
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
     "\n(Values close to 2 indicate no autocorrelation)")

#3.Homoscedasticity
plt.figure(figsize=(10,9))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title('Homoscedasticity Check:Residuals vs Predicted')
plt.xlabel("Predicted Price($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

#4.Normality of residuals
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
```

## Output:
<img width="418" height="340" alt="image" src="https://github.com/user-attachments/assets/c7a4c9b5-ac8f-41dd-84fd-06852492e3c2" />
<img width="1219" height="570" alt="image" src="https://github.com/user-attachments/assets/53d71f60-1a39-4a66-b236-ddd2ecb149ad" />
<img width="1279" height="754" alt="image" src="https://github.com/user-attachments/assets/d80f7c0c-9fc8-486c-8383-8112c6b223e4" />
<img width="1369" height="668" alt="image" src="https://github.com/user-attachments/assets/dafed30b-76a8-46ee-83aa-de58ecc4c246" />





## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
