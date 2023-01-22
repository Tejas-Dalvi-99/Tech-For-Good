# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
Bank = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\TimePass\Hackathon\Data.csv")

Bank.describe()

#Graphical Representation
Bank.dropna()
import matplotlib.pyplot as plt # mostly used for visualization purposes


plt.bar(height = Bank.Dependents, x = np.arange(1, 100, 2))
plt.hist(Bank.Dependents) #histogram
plt.boxplot(Bank.Dependents) #boxplot

plt.bar(height = Bank.ApplicantIncome, x = np.arange(1, 2199, 1))
plt.hist(Bank.ApplicantIncome) #histogram
plt.boxplot(Bank.ApplicantIncome) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=Bank['Credit_History'], y=Bank['ApplicantIncome'])


# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(Bank.ApplicantIncome, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Bank.iloc[:, :])
                             
# Correlation matrix
Bank.corr()
cars_new1.corr()
sns.pairplot(cars_new1.iloc[:, :])

         
ml1 = smf.ols('ApplicantIncome ~ CoapplicantIncome + LoanAmount + Credit_History ', data = Bank).fit() # regression model

# Summary
ml1.summary()


# Checking whether data has any influential values
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 581 is showing high influence so we can exclude that entire row

Bank_new = Bank.drop(Bank.index[[581]])

# Preparing model                  
ml_new = smf.ols('ApplicantIncome ~ CoapplicantIncome + LoanAmount + Credit_History ', data = Bank_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_appInc = smf.ols('ApplicantIncome ~ CoapplicantIncome + LoanAmount + Credit_History ', data = Bank).fit().rsquared  
vif_appInc = 1/(1 - rsq_appInc)

rsq_coInc = smf.ols('CoapplicantIncome ~ ApplicantIncome + LoanAmount + Credit_History ', data = Bank).fit().rsquared  
vif_coInc = 1/(1 - rsq_coInc)

rsq_LonAmt = smf.ols('LoanAmount ~ CoapplicantIncome + ApplicantIncome  + Credit_History ', data = Bank).fit().rsquared  
vif_LonAmt = 1/(1 - rsq_LonAmt)

rsq_crHis = smf.ols('Credit_History ~ CoapplicantIncome + LoanAmount +ApplicantIncome ', data = Bank).fit().rsquared  
vif_crHis = 1/(1 - rsq_crHis)

# Storing vif values in a data frame
d1 = {'Variables':['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History'], 'VIF':[vif_appInc,vif_coInc, vif_LonAmt, vif_crHis]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('ApplicantIncome ~ CoapplicantIncome + LoanAmount + Credit_History ', data = Bank).fit()
final_ml.summary()

# Prediction
pred = final_ml.predict(Bank)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = Bank.ApplicantIncome, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


### Splitting the data into train and test data
from sklearn.model_selection import train_test_split
Bank_train, Bank_test = train_test_split(Bank_new, test_size = 0.2) # 20% test data

# preparing the model on train data
model_train = smf.ols('ApplicantIncome ~ CoapplicantIncome + LoanAmount + Credit_History ', data = Bank_train).fit()

# prediction on test data set
test_pred = model_train.predict(Bank_test)

# test residual values
test_resid = test_pred - Bank_test.ApplicantIncome
# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(Bank_train)

# train residual values
train_resid  = train_pred - Bank_train.ApplicantIncome
# RMSE value for train data
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
