# -*- coding: utf-8 -*-
# [Team Carlton]
# [MMA]
# [2021 Winter]
# [Analytics for Financial Markets]
# [09/29/2020]


###
### INSTALL LIBRARIES AND SET SYSTEM OPTIONS
###

#Basic imports
import os
# The OS module in Python provides functions for creating and removing a directory (folder),
# fetching its contents, changing and identifying the current directory, etc.
from tqdm import tqdm # progress bar

import numpy as np

import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport

#Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline



from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Ignore warnings
import sys
import warnings
# warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter('ignore')

pd.options.display.width = 100
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 1000)



import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn import datasets
from sklearn import preprocessing

#General Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#Import scikit-learn metrics module for accuracy calculation of MSE & RMSE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

# from yellowbrick.classifier import ROCAUC

#Classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#Hyperopt imports
import hyperopt as hp
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK, space_eval
from hyperopt.pyll.base import scope


from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer # explicitly require this experimental feature
from sklearn.impute import IterativeImputer # now can import IterativeImputer normally from sklearn.impute

from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold

from scipy import stats
from scipy.special import boxcox1p


###
### IMPORT DATASET
###

# EPS – Earnings Per Share
# Liquidity – Working Capital/Total Assets
# Profitability – Retained Earnings/Total Assets
# Productivity – EBIT/Total Assets
# Leverage Ratio – (Total Long-term debt + Debt in Current liabilities)/Stockholders Equity
# Asset Turnover – Sales/ Total Assets
# Operational Margin – EBIT/Sales
# Market Book Ratio – (Price Close Annual Fiscal * Common Shares Outstanding)/Book Value Per Share
# Asset Growth – Change in assets from previous year
# Sales Growth – Change in sales from previous year
# Employee Growth – Change in employees from previous year
# Tobin’s Q – (Total market value of company + liabilities)/ (Total asset or book value + liabilities)
# BK – Company bankrupt or not


#Read the full dataset containing both train and test
df = pd.read_excel("Bankruptcy_data_Final.xlsx")


#Get info
df.shape
df.info()
df.head()



###
### PROFILE THE DATA
###

pandas_profiling.ProfileReport(df)



###
### EDA
###

#Explore target variable
total_len = len(df['BK'])
percentage_labels = (df['BK'].value_counts()/total_len)*100
percentage_labels
sns.set()
sns.countplot(df['BK']).set_title('Bankruptcy Classification')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100*(height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=3)
ax.set_xlabel("BK")
ax.set_ylabel("Count")
plt.show()



sns.set(style="ticks")

sns.pairplot(df, hue="BK", dropna=True);



#Check for duplicates
df[df.duplicated()].sort_values(["Data Year - Fiscal", "Tobin's Q"])
# No exact matches found



#Correlation matrix
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True)
plt.show()



df.boxplot(figsize=(35,20), grid=False)



df.hist(bins=30, figsize=(20,20))



###
### HANDLE MISSING DATA / NAs
###

#Summarize missing values
df.isnull().sum()



# creating surrogates for missing data
for col in df:
    if df[col].isna().sum() != 0: 
        df[col + '_surrogate'] = df[col].isna().astype(int)
        
#check new columns created
df.head()



#Impute missing values
num_nulls = pd.DataFrame({"Number of Nulls":df.isnull().sum()}) 
impute_cols = list(num_nulls[num_nulls["Number of Nulls"]!=0].index)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(missing_values=np.nan, max_iter=10, verbose=0, random_state=100)
df[impute_cols] = imp.fit_transform(df[impute_cols])
df
df.describe()



#Check for missing values
df.isnull().sum()
df.info()
df.describe()



###
### HANDLE SKEWED DATA
###

#Get float cols
float_cols = df.select_dtypes(include=[np.float]).columns
float_cols

#PowerTransdformer Scaler to deal with the skewness
scaler = PowerTransformer(method='yeo-johnson')
df[float_cols] = scaler.fit_transform(df[float_cols])#only numeric features should be scaled
df.shape
df.columns
df.head()



df.hist(bins=30, figsize=(20,20))



# Profile the data after the Standard Scaling
pandas_profiling.ProfileReport(df)



###
### FEATURE SELECTION - LOW VARIANCE
###

#Pick the numeric columns
X = df.copy()
feat = [feat for feat in X.columns if feat != "BK"]

#Remove columns with Zero or Low Variance
sel = VarianceThreshold(threshold=(0.1))
X = sel.fit_transform(X[feat])
X.shape

#Keep the True values
var_bols = list(sel.get_support())
A = pd.DataFrame({"All Cols":feat, "var_bols":var_bols})

#Get the chosen columns
T = A[A["var_bols"] == True]
chosen_cols = T["All Cols"].to_list()
print(chosen_cols)

#Surrogate features have zero variance and will therefore be excluded from the model


#Select only columns that do not have low variance
df1 = df[['Data Year - Fiscal', "Tobin's Q", 'EPS', 'Liquidity', 
          'Profitability', 'Productivity', 'Leverage Ratio', 
          'Asset Turnover', 'Operational Margin', 'Return on Equity', 
          'Market Book Ratio', 'Assets Growth', 'Sales Growth', 'Employee Growth', "BK"]]

df1.info()



###
### SPLIT THE DATA
###


#Split the data into train and test.
#80% of the data will be for training, model picking and Hyperparameter Tuning.
#20% will be for testing the finalized model.
#The train set will be used in K Fold cross validation.

#Translate slice ranged objects to concatenation along the first axis.
X_train, X_test, y_train, y_test = train_test_split(df1.iloc[:,:-1], 
                                                    df1.iloc[:,-1], 
                                                    test_size=0.2, 
                                                    random_state=100)


#Take a look at the resulting data sets after the split to ensure it was done well.

print("X_train")
X_train.shape
X_train.head() 

print("\n X_test")
X_test.shape
X_test.head()

print("\n y_train")
y_train.shape
y_train.head() 

print("\n y_test")
y_test.shape
y_test.head()



#Observe distribution

print("Train")
y_train.value_counts()

print("\n Test")
y_test.value_counts()




###
### IMBALANCED DATA - SMOTE
###

#Use SMOTE to handle imbalance
oversample = SMOTE(random_state=0)
X_resampled1, y_resampled1 = oversample.fit_resample(X_train, y_train)

print("X_Resampled1:")
X_resampled1.shape
X_resampled1.head()

print("\n y_Resampled1:")
y_resampled1.shape
y_resampled1.head()

y_resampled1.value_counts()



###
### IMBALANCED DATA - UNDERSAMPLING
###

#Undersamplinng to handle imbalance
ros = RandomUnderSampler(random_state=0)
X_resampled2, y_resampled2 = ros.fit_resample(X_train, y_train)

print("X_Resampled2:")
X_resampled2.shape
X_resampled2.head()

print("\n y_Resampled2:")
y_resampled2.shape
y_resampled2.head()

y_resampled2.value_counts()


###
### MODELING
###

# First we define a set of functions to compute the metrics of the model

# ROC curve
def plot_roc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1, drop_intermediate = False)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.001, 1.001])
    plt.ylim([-0.001, 1.001])
    plt.xlabel('1-Specificity (False Negative Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

# Confusion Matrix returns in the format: cm[0,0], cm[0,1], cm[1,0], cm[1,1]: tn, fp, fn, tp

# Sensitivity
def custom_sensitivity_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tp/(tp+fn))

# Specificity
def custom_specificity_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tn/(tn+fp))

# Positive Predictive Value
def custom_ppv_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tp/(tp+fp))

# Negative Predictive Value
def custom_npv_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tn/(tn+fn))

# Accuracy
def custom_accuracy_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return ((tn+tp)/(tn+tp+fn+fp))



###
### MODEL 1: Logistic Regression
###

# Grid search with 5 cross validation for Logisitic regression model

# List of hyperparameters for tuning
lr_grid_param = {'C': [0.001,0.009,0.01,0.09,1,5,25,50,100], "penalty":["l1","l2"]}# l1 lasso l2 ridge

# Run the GridSearch on the X_train and y_train data processed with SMOTE
clf_lr = LogisticRegression(solver='liblinear')
lr_grid = GridSearchCV(estimator=clf_lr,
                       param_grid=lr_grid_param,
                       scoring = 'roc_auc',
                       cv = 5) # 5 cross validation
lr_grid.fit(X_resampled1, y_resampled1)

# Print the model hyperparameter set with the best accuracy
print("tuned hpyerparameters :(best parameters) ",lr_grid.best_params_)
print("accuracy :",lr_grid.best_score_)



# Run the best Logisitc regression with {'C': 100, 'penalty': 'l1'}

LRM = LogisticRegression(solver='liblinear', random_state=100, C = 100, penalty = 'l1')

# Fit the train data into the LRM model
LRM.fit(X_resampled1, y_resampled1)



# The intercept and coefficients of the LRM model
LRM.intercept_

LRM.coef_



# Model evaluation on the TEST set
print(confusion_matrix(y_test, LRM.predict(X_test)))

print(classification_report(y_test, LRM.predict(X_test)))

# AUC score
auc_score_lr = roc_auc_score(y_test,LRM.predict(X_test))
round(float(auc_score_lr), 2 )

# Other performance metrics
print("Accuracy = {:.2f}".format(accuracy_score(y_test, LRM.predict(X_test))))
print("AUC score = {:.2f}".format(auc_score_lr))
print("F1 Score = {:.2f}".format(f1_score(y_test, LRM.predict(X_test))))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, LRM.predict(X_test))))
print("Log Loss = {:.2f}".format(log_loss(y_test, LRM.predict(X_test))))
print("Matthews Corrcoef = {:.2f}".format(matthews_corrcoef(y_test, LRM.predict(X_test))))



cm1 = confusion_matrix(y_test, LRM.predict(X_test))

TN1 = cm1[0][0]
FN1 = cm1[1][0]
TP1 = cm1[1][1]
FP1 = cm1[0][1]

# Sensitivity, hit rate, recall, or true positive rate
TPR1 = TP1/(TP1+FN1)
# Specificity or true negative rate
TNR1 = TN1/(TN1+FP1) 
# Precision or positive predictive value
PPV1 = TP1/(TP1+FP1)
# Negative predictive value
NPV1 = TN1/(TN1+FN1)
# Fall out or false positive rate
FPR1 = FP1/(FP1+TN1)
# False negative rate
FNR1 = FN1/(TP1+FN1)
# False discovery rate
FDR1 = FP1/(TP1+FP1)

print("True Negative = {:.0f}".format(TN1))
print("False Negative = {:.0f}".format(FN1))
print("True Positive = {:.0f}".format(TP1))
print("False Positive = {:.0f}".format(FP1))

print("Sensitivity = {:.2f}".format(TPR1))
print("Specificity = {:.2f}".format(TNR1))
print("Precision = {:.2f}".format(PPV1))
print("Negative predictive value = {:.4f}".format(NPV1))
print("False positive rate = {:.2f}".format(FPR1))
print("False negative rate = {:.2f}".format(FNR1))
print("False discovery rate = {:.2f}".format(FDR1))



#Summary of Tuned Random Forest
plt.figure(figsize=(15,6))

## CONFUSION MATRIX
plt.subplot(121)
# Set up the labels for in the confusion matrix
lr_cm = confusion_matrix(y_test, LRM.predict(X_test))
names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
lr_counts = ['{0:0.0f}'.format(value) for value in lr_cm.flatten()]
lr_percentages = ['{0:.2%}'.format(value) for value in lr_cm.flatten()/np.sum(lr_cm)]
lr_labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, lr_percentages)]
lr_labels = np.asarray(lr_labels).reshape(2,2)
ticklabels = ['No Bankruptcy', 'Bankruptcy']

# Create confusion matrix as heatmap
sns.set(font_scale = 1.4)
lr_ax = sns.heatmap(lr_cm, annot=lr_labels, fmt='', cmap='YlGnBu', xticklabels=ticklabels, yticklabels=ticklabels )
plt.xticks(size=12)
plt.yticks(size=12)
plt.title("Confusion Matrix") #plt.title("Confusion Matrix\n", fontsize=10)
plt.xlabel("Predicted", size=14)
plt.ylabel("Actual", size=14) 
#plt.savefig('lr_cm.png', transparent=True) 

## ROC CURVE
plt.subplot(122)
fpr1, tpr1, thresholds1 = roc_curve(y_test, LRM.predict(X_test))
auc1 = roc_auc_score(y_test, LRM.predict(X_test))
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = "Random (AUC = 50%)")
plt.plot(fpr1, tpr1, label='Logistics (AUC = {:.2f}%)'.format(auc1*100))
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend(loc='best')
#plt.savefig('roc.png', bbox_inches='tight', pad_inches=1)

## END PLOTS
plt.tight_layout()

## Summary Statistics
TN1, FP1, FN1, TP1 = lr_cm.ravel() # cm[0,0], cm[0, 1], cm[1, 0], cm[1, 1]
acc1 = (TP1 + TN1) / np.sum(lr_cm) # % positive out of all predicted positives
precision1 = TP1 / (TP1+FP1) # % positive out of all predicted positives
recall1 =  TP1 / (TP1+FN1) # % positive out of all supposed to be positives
specificity1 = TN1 / (TN1+FP1) # % negative out of all supposed to be negatives
lr_f1 = 2*precision1*recall1 / (precision1 + recall1)
MCC1 = matthews_corrcoef(y_test, LRM.predict(X_test))
stats_summary1 = '[Summary Statistics]\nF1 Score = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Accuracy = {:.2%} | Specificity = {:.2%} | MCC = {:.2%}'.format(lr_f1, precision1, recall1, acc1, specificity1, MCC1)
print(stats_summary1)



class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm1), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



# Plot the ROC Curve
fpr1, tpr1, threshold1 = roc_curve(y_test,LRM.predict(X_test),drop_intermediate=False)
roc_auc1 = metrics.auc(fpr1, tpr1)
 
plt.title('ROC Curve')
plt.plot(fpr1, tpr1, 'b', label='ROC curve (area = %0.2f)' % auc_score_lr)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



#Validate number of estimators parameter using f1 score 
lr_num_est = (0.001,0.009,0.01,0.09,1,5,25,50,100)

lr_train_score, lr_test_score = validation_curve(LogisticRegression(solver='liblinear', penalty = 'l1'), 
                                                 X = X_resampled1, y = y_resampled1, 
                                                 param_name = 'C', 
                                                 param_range = lr_num_est, 
                                                 cv = 3, 
                                                 scoring = "f1")

# Calculating mean and standard deviation of training score 
lr_mean_train_score = np.mean(lr_train_score, axis = 1) 
lr_std_train_score = np.std(lr_train_score, axis = 1) 
  
# Calculating mean and standard deviation of testing score 
lr_mean_test_score = np.mean(lr_test_score, axis = 1) 
lr_std_test_score = np.std(lr_test_score, axis = 1) 
  
# Plot mean accuracy scores for training and testing scores 
plt.plot(lr_num_est, lr_mean_train_score, 
         label = "Training Score", color = 'b') 
plt.plot(lr_num_est, lr_mean_test_score, 
         label = "Cross Validation Score", color = 'g') 
  
# Creating the plot 
plt.title("Validation Curve") 
plt.xlabel("Value of C") 
plt.ylabel("f1") 
plt.tight_layout() 
plt.legend(loc = 'best') 
plt.show()



#Plot Precision-Recall Curve

precision1, recall1, thresholds1 = precision_recall_curve(y_test, LRM.predict(X_test))
plt.figure(figsize = (20,10))
plt.plot(recall1, precision1)
plt.plot([0, 1], [0.5, 0.5], linestyle = '--')
plt.xlabel('Recall', fontsize = 16)
plt.ylabel('Precision', fontsize = 16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('Logistics Precision-Recall Curve', fontsize = 28)
plt.show();



###
### MODEL 2: Random Forest
###

###
### Hyperparameter Exploration: Validation Curves for Various Parameters
###

#Validate number of estimators parameter using f1 score 
rf_num_est = np.arange(10, 60, 10)

rf_train_score, rf_test_score = validation_curve(RandomForestClassifier(), 
                                                 X = X_resampled1, y = y_resampled1, 
                                                 param_name = 'n_estimators', 
                                                 param_range = rf_num_est, 
                                                 cv = 3, 
                                                 scoring = "roc_auc")

# Calculating mean and standard deviation of training score 
rf_mean_train_score = np.mean(rf_train_score, axis = 1) 
rf_std_train_score = np.std(rf_train_score, axis = 1) 
  
# Calculating mean and standard deviation of testing score 
rf_mean_test_score = np.mean(rf_test_score, axis = 1) 
rf_std_test_score = np.std(rf_test_score, axis = 1) 
  
# Plot mean accuracy scores for training and testing scores 
plt.plot(rf_num_est, rf_mean_train_score, 
         label = "Training Score", color = 'b') 
plt.plot(rf_num_est, rf_mean_test_score, 
         label = "Cross Validation Score", color = 'g') 
  
# Creating the plot 
plt.title("Validation Curve") 
plt.xlabel("Number of Estimators") 
plt.ylabel("AUC") 
plt.tight_layout() 
plt.legend(loc = 'best') 
plt.show()



#Validate max depth parameter using f1 score 
rf_max_depth = np.arange(5, 30, 5)

rf_train_score, rf_test_score = validation_curve(RandomForestClassifier(),
                                                 X = X_resampled1, y = y_resampled1, 
                                                 param_name = 'max_depth', 
                                                 param_range = rf_max_depth, 
                                                 cv = 3, 
                                                 scoring = "roc_auc")

# Calculating mean and standard deviation of training score 
rf_mean_train_score = np.mean(rf_train_score, axis = 1) 
rf_std_train_score = np.std(rf_train_score, axis = 1) 
  
# Calculating mean and standard deviation of testing score 
rf_mean_test_score = np.mean(rf_test_score, axis = 1) 
rf_std_test_score = np.std(rf_test_score, axis = 1) 
  
# Plot mean accuracy scores for training and testing scores 
plt.plot(rf_max_depth, rf_mean_train_score, 
         label = "Training Score", color = 'b') 
plt.plot(rf_max_depth, rf_mean_test_score, 
         label = "Cross Validation Score", color = 'g') 
  
# Creating the plot 
plt.title("Validation Curve") 
plt.xlabel("max_depth") 
plt.ylabel("AUC") 
plt.tight_layout() 
plt.legend(loc = 'best') 
plt.show()



# Define a random forest model and call it classifier_RF
clf_rf = RandomForestClassifier(random_state=100)

# Define parameter grid
rf_grid_param = { 
    'n_estimators': [35, 40, 60],
    'min_samples_split': [2, 5],
    'max_depth' : [15, 25, 30 ]
}

# Define scoring parameters
scoring = ['roc_auc', 'recall', 'f1', 'f1_macro', 'balanced_accuracy']

# Define grid search
rf_grid = GridSearchCV(estimator=clf_rf, 
                       param_grid=rf_grid_param, 
                       scoring=scoring, 
                       refit='roc_auc', 
                       cv=3, 
                       n_jobs=-1, 
                       verbose=3)

# Train the model classifier_RF on the training data
rf_grid.fit(X_resampled1, y_resampled1)



# Print the model hyperparameter set with the best accuracy
print("tuned hpyerparameters :(best parameters) ", rf_grid.best_params_)
print("accuracy :", rf_grid.best_score_)

# Get best parameters results from Grid Search
rf_cv_results = pd.DataFrame(rf_grid.cv_results_)
rf_cv_results = rf_cv_results.sort_values(by='mean_test_f1_macro', ascending=False)
rf_cv_results[['mean_test_recall', 'mean_test_roc_auc', 
               'mean_test_f1','mean_test_balanced_accuracy', 
               'param_n_estimators', 'param_max_depth', 'param_min_samples_split']].round(6).head()



# Use optimal parameters to predict test
RF = RandomForestClassifier(n_estimators=40,
                             min_samples_split=2,
                             max_depth=30,
                             max_features='auto',
                             random_state=100,
                             n_jobs=-1)

RF.fit(X_resampled1,y_resampled1)



# Use the trained model to predict testing data
class_threshold = 0.07
rf_grid_y_pred_prob = RF.predict_proba(np.array(X_test))[:,1]
rf_grid_y_pred = np.where(rf_grid_y_pred_prob > class_threshold, 1, 0) # classification



#threshold to optimize F1
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# define thresholds
thresholds2 = np.arange(0, 1, 0.001)
# evaluate each threshold
scores2 = [f1_score(y_test, to_labels(rf_grid_y_pred_prob, t)) for t in thresholds2]
# get best threshold
ix = np.argmax(scores2)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds2[ix], scores2[ix]))



#threshold to optimize auc
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# define thresholds
thresholds2 = np.arange(0, 1, 0.001)
# evaluate each threshold
scores2 = [roc_auc_score(y_test, to_labels(rf_grid_y_pred_prob, t)) for t in thresholds2]
# get best threshold
ix = np.argmax(scores2)
print('Threshold=%.3f, roc_auc_score=%.5f' % (thresholds2[ix], scores2[ix]))



#threshold to optimize average_precision_score
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# define thresholds
thresholds2 = np.arange(0, 1, 0.001)
# evaluate each threshold
scores2 = [average_precision_score(y_test, to_labels(rf_grid_y_pred_prob, t)) for t in thresholds2]
# get best threshold
ix = np.argmax(scores2)
print('Threshold=%.3f, average_precision_score=%.5f' % (thresholds2[ix], scores2[ix]))



cm2 = confusion_matrix(y_test, rf_grid_y_pred)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm2), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual Bankruptcy')
plt.xlabel('Predicted Bankruptcy')



# Lets look at the model metrics

print('Metrics of the Random Forest model: \n')

cm2 = np.transpose(confusion_matrix(y_test, rf_grid_y_pred))
print("Confusion matrix: \n" + str(cm2))

print("                                   Accuracy: " + str(custom_accuracy_score(y_test, rf_grid_y_pred))) 
print("                   SENSITIVITY (aka RECALL): " + str(custom_sensitivity_score(y_test, rf_grid_y_pred)))
print("                 SPECIFICITY (aka FALL-OUT): " + str(custom_specificity_score(y_test, rf_grid_y_pred)))
print(" POSITIVE PREDICTIVE VALUE, (aka PRECISION): " + str(custom_ppv_score(y_test, rf_grid_y_pred)))
print("                 NEGATIVE PREDICTIVE VALUE): " + str(custom_npv_score(y_test, rf_grid_y_pred)))

plot_roc(y_test, rf_grid_y_pred_prob)
print(" AUC: " + str(roc_auc_score(y_test, rf_grid_y_pred_prob)))
print(" MCC: " + str(matthews_corrcoef(y_test, rf_grid_y_pred)))



#View Classification Report
print(classification_report(y_test,rf_grid_y_pred))



#Plot Precision-Recall Curve

precision2, recall2, thresholds2 = precision_recall_curve(y_test, rf_grid_y_pred_prob)
plt.figure(figsize = (20,10))
plt.plot(recall2, precision2)
plt.plot([0, 1], [0.5, 0.5], linestyle = '--')
plt.xlabel('Recall', fontsize = 16)
plt.ylabel('Precision', fontsize = 16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('Random Forest Precision-Recall Curve', fontsize = 28)
plt.show();



#Summary of Tuned Random Forest
plt.figure(figsize=(15,6))

## CONFUSION MATRIX
plt.subplot(121)
# Set up the labels for in the confusion matrix
rf_cm = confusion_matrix(y_test, rf_grid_y_pred)
names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
rf_counts = ['{0:0.0f}'.format(value) for value in rf_cm.flatten()]
rf_percentages = ['{0:.2%}'.format(value) for value in rf_cm.flatten()/np.sum(rf_cm)]
rf_labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, rf_percentages)]
rf_labels = np.asarray(rf_labels).reshape(2,2)
ticklabels = ['No Bankruptcy', 'Bankruptcy']

# Create confusion matrix as heatmap
sns.set(font_scale = 1.4)
rf_ax = sns.heatmap(rf_cm, annot=rf_labels, fmt='', cmap='YlGnBu', xticklabels=ticklabels, yticklabels=ticklabels )
plt.xticks(size=12)
plt.yticks(size=12)
plt.title("Confusion Matrix") #plt.title("Confusion Matrix\n", fontsize=10)
plt.xlabel("Predicted", size=14)
plt.ylabel("Actual", size=14) 
#plt.savefig('rf_cm.png', transparent=True) 

## ROC CURVE
plt.subplot(122)
fpr2, tpr2, thresholds2 = roc_curve(y_test, rf_grid_y_pred)
auc2 = roc_auc_score(y_test, rf_grid_y_pred)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = "Random (AUC = 50%)")
plt.plot(fpr2, tpr2, label='Random Forest (AUC = {:.2f}%)'.format(auc2*100))
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend(loc='best')
#plt.savefig('roc.png', bbox_inches='tight', pad_inches=1)

## END PLOTS
plt.tight_layout()

## Summary Statistics
TN2, FP2, FN2, TP2 = rf_cm.ravel() # cm[0,0], cm[0, 1], cm[1, 0], cm[1, 1]
acc2 = (TP2 + TN2) / np.sum(rf_cm) # % positive out of all predicted positives
precision2 = TP2 / (TP2+FP2) # % positive out of all predicted positives
recall2 =  TP2 / (TP2+FN2) # % positive out of all supposed to be positives
specificity2 = TN2 / (TN2+FP2) # % negative out of all supposed to be negatives
rf_f1 = 2*precision2*recall2 / (precision2 + recall2)
MCC2 = matthews_corrcoef(y_test, rf_grid_y_pred)
stats_summary2 = '[Summary Statistics]\nF1 Score = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Accuracy = {:.2%} | Specificity = {:.2%} | MCC = {:.2%}'.format(rf_f1, precision2, recall2, acc2, specificity2, MCC2)
print(stats_summary2)



###
### MODEL 3: Naive Bayes - Gaussian
###

###
### Hyperparameter Exploration: Validation Curves for Various Parameters
###

#Validate number of estimators parameter using f1 score 
var_smooth_param = np.arange(0.000000001, 1, 0.2)

nb_train_score, nb_test_score = validation_curve(GaussianNB(), 
                                                 X = X_resampled1, y = y_resampled1, 
                                                 param_name = 'var_smoothing', 
                                                 param_range = var_smooth_param, 
                                                 cv = 3, 
                                                 scoring = "roc_auc")

# Calculating mean and standard deviation of training score 
nb_mean_train_score = np.mean(nb_train_score, axis = 1) 
nb_std_train_score = np.std(nb_train_score, axis = 1) 
  
# Calculating mean and standard deviation of testing score 
nb_mean_test_score = np.mean(nb_test_score, axis = 1) 
nb_std_test_score = np.std(nb_test_score, axis = 1) 
  
# Plot mean accuracy scores for training and testing scores 
plt.plot(var_smooth_param, nb_mean_train_score, 
         label = "Training Score", color = 'b') 
plt.plot(var_smooth_param, nb_mean_test_score, 
         label = "Cross Validation Score", color = 'g') 
  
# Creating the plot 
plt.title("Validation Curve") 
plt.xlabel("var_smoothing") 
plt.ylabel("AUC") 
plt.tight_layout() 
plt.legend(loc = 'best') 
plt.show()



# Define a Gaussian NB model and call it gnb
clf_gnb = GaussianNB()

# Define parameter grid
gnb_grid_param = {
    'var_smoothing': np.arange(0, 1, 0.2)
}

# Define scoring parameters
scoring = ['roc_auc', 'recall', 'f1', 'f1_macro', 'balanced_accuracy']

# Define grid search
gnb_grid = GridSearchCV(estimator=clf_gnb, 
                        param_grid=gnb_grid_param, 
                        scoring=scoring, 
                        refit='roc_auc', 
                        cv=3, 
                        n_jobs=-1, 
                        verbose=3)


# Train the model clf_gnb on the training data
gnb_grid.fit(X_resampled1, y_resampled1)



# Get best parameters results
print(gnb_grid.best_score_)
print(gnb_grid.best_params_)

gnb_grid_results = pd.DataFrame(gnb_grid.cv_results_)
gnb_grid_results = gnb_grid_results.sort_values(by='mean_test_f1_macro', ascending=False)
gnb_grid_results[['mean_test_recall', 'mean_test_roc_auc', 
                'mean_test_f1', 'mean_test_f1_macro','mean_test_balanced_accuracy', 
                'param_var_smoothing']].round(6).head()



GNB = GaussianNB(var_smoothing= 0.4)
GNB.fit(X_resampled1, y_resampled1)
gnb_y_pred = GNB.predict(X_test)
print(classification_report(y_test,gnb_y_pred))



cm3 = confusion_matrix(y_test, gnb_y_pred)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm3), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



# Lets look at the model metrics

print('Metrics of the Gaussian Naive Bayes model: \n')

cm = np.transpose(confusion_matrix(y_test, gnb_y_pred))
print("Confusion matrix: \n" + str(cm))

print("                                   Accuracy: " + str(custom_accuracy_score(y_test, gnb_y_pred))) 
print("                   SENSITIVITY (aka RECALL): " + str(custom_sensitivity_score(y_test, gnb_y_pred)))
print("                 SPECIFICITY (aka FALL-OUT): " + str(custom_specificity_score(y_test, gnb_y_pred)))
print(" POSITIVE PREDICTIVE VALUE, (aka PRECISION): " + str(custom_ppv_score(y_test, gnb_y_pred)))
print("                 NEGATIVE PREDICTIVE VALUE): " + str(custom_npv_score(y_test, gnb_y_pred)))

plot_roc(y_test, gnb_y_pred)
print(" AUC: " + str(roc_auc_score(y_test, gnb_y_pred)))
print(" MCC: " + str(matthews_corrcoef(y_test, gnb_y_pred)))



#View Classification Report
print(classification_report(y_test,gnb_y_pred))



#Plot Precision-Recall Curve

precision3, recall3, thresholds3 = precision_recall_curve(y_test, gnb_y_pred)
plt.figure(figsize = (20,10))
plt.plot(recall3, precision3)
plt.plot([0, 1], [0.5, 0.5], linestyle = '--')
plt.xlabel('Recall', fontsize = 16)
plt.ylabel('Precision', fontsize = 16)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.title('Gaussian Naive Bayes Precision-Recall Curve', fontsize = 28)
plt.show();



#Summary of Tuned Naive Bayes
plt.figure(figsize=(15,6))

## CONFUSION MATRIX
plt.subplot(121)
# Set up the labels for in the confusion matrix
gnb_cm = confusion_matrix(y_test, gnb_y_pred)
names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
gnb_counts = ['{0:0.0f}'.format(value) for value in gnb_cm.flatten()]
percentages3 = ['{0:.2%}'.format(value) for value in gnb_cm.flatten()/np.sum(gnb_cm)]
labels3 = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages3)]
labels3 = np.asarray(labels3).reshape(2,2)
ticklabels = ['No Bankruptcy', 'Bankruptcy']

# Create confusion matrix as heatmap
sns.set(font_scale = 1.4)
gnb_ax = sns.heatmap(gnb_cm, annot=labels3, fmt='', cmap='YlGnBu', xticklabels=ticklabels, yticklabels=ticklabels )
plt.xticks(size=12)
plt.yticks(size=12)
plt.title("Confusion Matrix") #plt.title("Confusion Matrix\n", fontsize=10)
plt.xlabel("Predicted", size=14)
plt.ylabel("Actual", size=14) 
#plt.savefig('gnb_cm.png', transparent=True) 

## ROC CURVE
plt.subplot(122)
fpr3, tpr3, thresholds = roc_curve(y_test, gnb_y_pred)
auc3 = roc_auc_score(y_test, gnb_y_pred)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = "Random (AUC = 50%)")
plt.plot(fpr3, tpr3, label='Naive Bayes (AUC = {:.2f}%)'.format(auc3*100))
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend(loc='best')
#plt.savefig('roc.png', bbox_inches='tight', pad_inches=1)

## END PLOTS
plt.tight_layout()

## Summary Statistics
TN3, FP3, FN3, TP3 = gnb_cm.ravel() # cm[0,0], cm[0, 1], cm[1, 0], cm[1, 1]
acc3 = (TP3 + TN3) / np.sum(gnb_cm) # % positive out of all predicted positives
precision3 = TP3 / (TP3+FP3) # % positive out of all predicted positives
recall3 =  TP3 / (TP3+FN3) # % positive out of all supposed to be positives
specificity3 = TN3 / (TN3+FP3) # % negative out of all supposed to be negatives
gnb_f1 = 2*precision3*recall3 / (precision3 + recall3)
MCC3 = matthews_corrcoef(y_test, gnb_y_pred)
stats_summary3 = '[Summary Statistics]\nF1 Score = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Accuracy = {:.2%} | Specificity = {:.2%} | MCC = {:.2%}'.format(gnb_f1, precision3, recall3, acc3, specificity3, MCC3)
print(stats_summary3)



###
### MODEL 4: Support Vector Machine
###

# train the model on train set 
clf_svm = SVC() 
clf_svm.fit(X_resampled2, y_resampled2) 
  
# print prediction results 
svm_y_predict = clf_svm.predict(X_test) 
print(classification_report(y_test, svm_y_predict)) 



# defining parameter range 
svm_grid_param = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'kernel': ['rbf']}  
  
svm_grid = GridSearchCV(SVC(), svm_grid_param, refit = True, verbose = 3) 
  
# fitting the model for grid search 
svm_grid.fit(X_resampled2, y_resampled2) 



# print best model hyperparameter set after tuning 
print("tuned hpyerparameters (best parameters) ",svm_grid.best_params_)

# print how our model looks after hyper-parameter tuning 
print(svm_grid.best_estimator_)
print("accuracy :",svm_grid.best_score_)



svm_grid_y_predict = svm_grid.predict(X_test) 

# print classification report 
print(classification_report(y_test, svm_grid_y_predict))



#### Summary of Tuned SVM
plt.figure(figsize=(15,6))

## CONFUSION MATRIX
plt.subplot(121)
# Set up the labels for in the confusion matrix
svm_cm = confusion_matrix(y_test, svm_grid_y_predict)
names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
svm_counts = ['{0:0.0f}'.format(value) for value in svm_cm.flatten()]
svm_percentages = ['{0:.2%}'.format(value) for value in svm_cm.flatten()/np.sum(svm_cm)]
svm_labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, svm_percentages)]
svm_labels = np.asarray(svm_labels).reshape(2,2)
ticklabels = ['No Bankruptcy', 'Bankruptcy']

# Create confusion matrix as heatmap
sns.set(font_scale = 1.4)
svm_ax = sns.heatmap(svm_cm, annot=svm_labels, fmt='', cmap='YlGnBu', xticklabels=ticklabels, yticklabels=ticklabels )
plt.xticks(size=12)
plt.yticks(size=12)
plt.title("Confusion Matrix") #plt.title("Confusion Matrix\n", fontsize=10)
plt.xlabel("Predicted", size=14)
plt.ylabel("Actual", size=14) 
#plt.savefig('svm_cm.png', transparent=True) 

## ROC CURVE
plt.subplot(122)
fpr4, tpr4, thresholds4 = roc_curve(y_test, svm_grid_y_predict)
auc4 = roc_auc_score(y_test, svm_grid_y_predict)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = "Random (AUC = 50%)")
plt.plot(fpr4, tpr4, label='Support Vector Machine (AUC = {:.2f}%)'.format(auc4*100))
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend(loc='best')
#plt.savefig('roc.png', bbox_inches='tight', pad_inches=1)

## END PLOTS
plt.tight_layout()

## Summary Statistics
TN4, FP4, FN4, TP4 = svm_cm.ravel() # cm[0,0], cm[0, 1], cm[1, 0], cm[1, 1]
acc4 = (TP4 + TN4) / np.sum(svm_cm) # % positive out of all predicted positives
precision4 = TP4 / (TP4+FP4) # % positive out of all predicted positives
recall4 =  TP4 / (TP4+FN4) # % positive out of all supposed to be positives
specificity4 = TN4 / (TN4+FP4) # % negative out of all supposed to be negatives
svm_f1 = 2*precision4*recall4 / (precision4 + recall4)
MCC4 = matthews_corrcoef(y_test, svm_grid_y_predict)
stats_summary4 = '[Summary Statistics]\nF1 Score = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Accuracy = {:.2%} | Specificity = {:.2%} | MCC = {:.2%}'.format(svm_f1, precision4, recall4, acc4, specificity4, MCC4)
print(stats_summary4)



###
### MODEL 5: AdaBoost with RuS
###

# RUS RANDOM UNDER SAMPLING
X_full = X_train.copy()
X_full['BK'] = y_train
X_maj = X_full[X_full.BK==0]
X_min = X_full[X_full.BK==1]
X_maj_rus = resample(X_maj,replace=False,n_samples=len(X_min),random_state=100)
X_rus = pd.concat([X_maj_rus, X_min])
X_train_rus = X_rus.drop(['BK'], axis=1)
y_train_rus = X_rus.BK


# AdaBoost with RuS
# Hyper Parameters Tuning
ada_grid_param = {"base_estimator__splitter" : ["best", "random"], 
                  "n_estimators": [1,50,75,85,95,100],
                  "base_estimator__max_depth":[1,2,3,4,5] 
                 }
# Decision Tree and Ada Boost
DTC = DecisionTreeClassifier(random_state = 100, max_features = "auto", class_weight = "balanced", max_depth = None)
ABC = AdaBoostClassifier(base_estimator = DTC)
# Gridsearch with Cross validation to 5
ada_grid =GridSearchCV(ABC, param_grid=ada_grid_param, scoring = 'roc_auc',cv=5, return_train_score=False)
ada_grid.fit(X_train_rus, y_train_rus.values.reshape(-1,))
# print best model hyperparameter set after tuning 
print("tuned hpyerparameters (best parameters) ",ada_grid.best_params_)


# Running Adaboost based on the best parameters from GridSearch
ADA = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=5, min_samples_split=2,splitter='best'), 
    n_estimators=95, random_state=100)
ADA.fit(X_train_rus, y_train_rus)

# Making Prediction for Test
y_pred_ada=ADA.predict(X_test)


#Confusion matrix for the test
ada_feature_names = X_train_rus.columns
ada_class_names = [str(x) for x in ADA.classes_]
print("CONFUSION MATRIX FOR TEST:")
confusion_matrix(y_test, y_pred_ada)

print(classification_report(y_test, y_pred_ada, target_names=ada_class_names))
# Calculating Accuracy, F1 Score, log loss n MCC 
score_str = "acc={:.2f}, kappa={:.2f}, f1={:.2f},Log Loss={:.2f},Mathews_Coefficient={:.2f}".format(
        accuracy_score(y_test, y_pred_ada),
        cohen_kappa_score(y_test, y_pred_ada),
        f1_score(y_test, y_pred_ada),
log_loss(y_test, y_pred_ada),
matthews_corrcoef(y_test, y_pred_ada))
score_str
# # Calculating AUC ROC for Adaboost
# visualizer = ROCAUC(ADA, classes=ada_class_names)
# visualizer.fit(X_train, y_train)
# visualizer.score(X_test, y_test)
# g = visualizer.poof()


#### Summary of Tuned SVM
plt.figure(figsize=(15,6))

## CONFUSION MATRIX
plt.subplot(121)
# Set up the labels for in the confusion matrix
ada_cm = confusion_matrix(y_test, y_pred_ada)
names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
ada_counts = ['{0:0.0f}'.format(value) for value in ada_cm.flatten()]
ada_percentages = ['{0:.2%}'.format(value) for value in ada_cm.flatten()/np.sum(ada_cm)]
ada_labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, ada_percentages)]
ada_labels = np.asarray(ada_labels).reshape(2,2)
ticklabels = ['No Bankruptcy', 'Bankruptcy']

# Create confusion matrix as heatmap
sns.set(font_scale = 1.4)
ada_ax = sns.heatmap(ada_cm, annot=ada_labels, fmt='', cmap='YlGnBu', xticklabels=ticklabels, yticklabels=ticklabels )
plt.xticks(size=12)
plt.yticks(size=12)
plt.title("Confusion Matrix") #plt.title("Confusion Matrix\n", fontsize=10)
plt.xlabel("Predicted", size=14)
plt.ylabel("Actual", size=14) 
#plt.savefig('ada_cm.png', transparent=True) 

## ROC CURVE
plt.subplot(122)
fpr5, tpr5, thresholds5 = roc_curve(y_test, y_pred_ada)
auc5 = roc_auc_score(y_test, y_pred_ada)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = "Random (AUC = 50%)")
plt.plot(fpr5, tpr5, label='AdaBoost (AUC = {:.2f}%)'.format(auc5*100))
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend(loc='best')
#plt.savefig('roc.png', bbox_inches='tight', pad_inches=1)

## END PLOTS
plt.tight_layout()

## Summary Statistics
TN5, FP5, FN5, TP5 = ada_cm.ravel() # cm[0,0], cm[0, 1], cm[1, 0], cm[1, 1]
acc5 = (TP5 + TN5) / np.sum(ada_cm) # % positive out of all predicted positives
precision5 = TP5 / (TP5+FP5) # % positive out of all predicted positives
recall5 =  TP5 / (TP5+FN5) # % positive out of all supposed to be positives
specificity5 = TN5 / (TN5+FP5) # % negative out of all supposed to be negatives
ada_f1 = 2*precision5*recall5 / (precision5 + recall5)
MCC5 = matthews_corrcoef(y_test, y_pred_ada)
stats_summary5 = '[Summary Statistics]\nF1 Score = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Accuracy = {:.2%} | Specificity = {:.2%} | MCC = {:.2%}'.format(ada_f1, precision5, recall5, acc5, specificity5, MCC5)
print(stats_summary5)