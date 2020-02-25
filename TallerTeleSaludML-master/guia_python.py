

import pandas as pd
import numpy as np

path = "G:\\My Drive\\..."
path = "G:\\My Drive\\Maritzinha\\Unicauca\\Doctorado\\taller IA\\datasets\\hepatitis"


#outliers
filename = path + "\\pima-indians-diabetes.data.csv"
names = ['Pregnancies', 'Glucose', 'BloodPress', 'SkinThick', 'Insulin', 'BMI', 'DPF', 'Age', 'Diabetic']
data = pd.read_csv(filename, names = names)
data.boxplot(column= names, figsize = (20,20))
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(20,20))


#analisis de outliers
mask = data['Pregnancies']>12
print (data[mask])

#reemplazo
mask = data['BMI']==0
print (data[mask])
BMI = data[~mask]['BMI'].mean()
BMI = data[~mask]['BMI'].max()
BMI = data[~mask]['BMI'].min()


print (data[mask]['BMI'] )
ind = data[mask].index.values
for i in list(ind):
    data.loc[i,'BMI']=BMI



#valores perdidos

names = ["Class", "AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER_BIG", "LIVER_FIRM", "SPLEEN_PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK_PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]

data = pd.read_csv(path + "\\hepatitis.csv", names = names)
#data = data.replace ({"?":-9})
data = data.replace ({"?":np.NaN})
#data = data.fillna(-9)
datos = data.dropna()

###reemplazo en variable categorica
mask = data["LIVER_BIG"].isnull()
moda = data[~mask]["LIVER_BIG"].mode()[0]
data["LIVER_BIG"].fillna(moda)


###trasnformación
data['SEX'] = data['SEX'].replace({"female":1, "male":2})




#Formato 
mask = data["ALBUMIN"].isnull()
data['ALBUMIN'] = data['ALBUMIN'] .astype('float32')

media = data[~mask]["ALBUMIN"].mean()
data["ALBUMIN"].fillna(media)


####################selección de caracteristicas  RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
dataframe = pd.read_csv(path+"\\heart_complete.csv", header=None)


"método: recursive feature elimination"
X = dataframe[dataframe.columns[0:dataframe.shape[1]-1]]
Y = dataframe[len (dataframe.columns)-1]    


model = RandomForestClassifier()
K = 5
rfe = RFE(model, K)
fit = rfe.fit(X, Y)

print("Num Features: %d", fit.n_features_)
print("Selected Features: %s", fit.support_)

selected_feature = []
for i in range (len(fit.support_)):
    if (fit.support_[i]):
        selected_feature.append(dataframe.columns[i])
        
print (selected_feature)
        
        
   
##############seleccion de caracterisicas SelectKBest     

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
dataframe = pd.read_csv(path+"\\heart_complete.csv", header=None)
        
X = dataframe[dataframe.columns[0:dataframe.shape[1]-1]]
Y = dataframe[len (dataframe.columns)-1]    



# feature extraction
test = SelectKBest(score_func=chi2, k=5)

fit = test.fit(X, Y)

selected_feature = []
for i in range (len(fit.get_support())):
    if (fit.get_support()[i]):
        selected_feature.append(dataframe.columns[i])
        
print (selected_feature)



#########################clasificación


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                            X[selected_feature], Y, test_size=0.33)

modelo = RandomForestClassifier()

modelo.fit(X_train, y_train)
    #            print('')
print("RandomForestClassifier",modelo.score(X_test, y_test))

modelo2 = LogisticRegression()
modelo2.fit(X_train, y_train)
print("LogisticRegression",modelo2.score(X_test, y_test))


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

modelo3= DecisionTreeClassifier()
modelo3.fit(X_train, y_train)
print("DecisionTreeClassifier",modelo3.score(X_test, y_test))


modelo3= SVC()
modelo3.fit(X_train, y_train)
print("SVC",modelo3.score(X_test, y_test))






####### analisis de EEG


dataframe_eeg = pd.read_csv(path+"\\dataset_taller_test_data_nan.csv", header=None)

dataframe_eeg.isnull().sum()

for i in dataframe_eeg.columns:
    mask = dataframe_eeg[i].isnull()
#    data['ALBUMIN'] = data['ALBUMIN'] .astype('float32')
    mean = dataframe_eeg[~mask][i].mean()
    dataframe_eeg [i] = dataframe_eeg[i].fillna(mean)
    
X = dataframe_eeg[dataframe_eeg.columns [0:len (dataframe_eeg.columns)-1]]
Y = dataframe_eeg[dataframe_eeg.columns [len (dataframe_eeg.columns)-1]]

    
X_train, X_test, y_train, y_test = train_test_split(
                            X, Y, test_size=0.33)

modeloEEG = LogisticRegression()
modeloEEG.fit(X_train, y_train)
print("LogisticRegression",modeloEEG.score(X_test, y_test))




        




