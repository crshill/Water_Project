
"""
Here are a set of supervised machine learning algorithms:
    Random Forest Classifier
    Random Forest Gradient Boosting Models
    Random Forest Feature Selection
    Suport Vector Machine
    
"""

#import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import scale
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm

# noinspection PyDeprecation
# from sklearn.cross_validation import train_test_split


class S_Models(object) :

    """
    This is a routine that utilizes a random forest classifier to predict the rating of all given WUSs by cycling through
    the entire set of municipalities.
    """
    
    def classifier_step(self, file, mdl) :

        data = pd.read_csv(file)#, index_col = 'Munic&Year')
        features = list(data.columns[3:-1])

        out = pd.Series()
        datay = data["Rating"]
        datax = data[features]

        
        scaler = StandardScaler().fit(datax)
        scalx = scaler.transform(datax)
        normedx = normalize(scalx)
        norm = pd.DataFrame(data=normedx, columns = features, 
                            index = data.index)
        
        all_names = data['Munic'].drop_duplicates().values


        norm_data = pd.merge(norm, datay, how = 'right', left_index=True, 
                             right_index=True, sort = False)
        
        prog = 0
        
        for nn in all_names :


            test = norm_data.loc[data['Munic'] == nn]
            train = norm_data.loc[data['Munic'] != nn]


            y = train["Rating"]
            x = train[features]
            
            x_test = test[features]
                #y_test = test['Rating']

            #print(str(nn+1) + " of " + str(NN))
            prog += len(test)
            print(str(round(prog/len(data)*100,2)) + '% Complete')

            names = data.loc[test.index, 'Munic&Year']
            rates = data.loc[test.index, 'Rating']

            if mdl == 1 :
            
                clf = RandomForestClassifier(n_estimators=500)
                model = clf.fit(x, y)

                
            elif mdl == 2 :
                
                clf = RandomForestClassifier(n_estimators = 500)
                abc = AdaBoostClassifier(n_estimators = 100,
                            base_estimator = clf, learning_rate = 1)
                model = abc.fit(x, y)
                
            elif mdl == 3 :
                
                gb = GradientBoostingClassifier(n_estimators=500, learning_rate = 1,
                                        warm_start=True)
                model = gb.fit(x, y)
            
            #if nn % 10 == 0 : 
            if prog % 7 < 2 :
                print(sorted(zip(map(lambda x: round(x, 4), 
                        model.feature_importances_), features), reverse=True))

            est = model.predict(x_test)
                
                
            aa = {'Munic&Year':names, 'model': est, 'Given': rates}
            add= pd.DataFrame(aa)

            out = out.append(add)
                
        out.to_csv("./Model/AllMunics_Modeled.csv")

        #return out
        
        
####################################################################
        ###########################################################
        
    def regression_step(self, file, mdl) :

        data = pd.read_csv(file)#, index_col = 'Munic&Year')
        features = list(data.columns[3:-1])
        print(features)
        out = pd.Series()
        NN = len(data)/step
        NN = int(NN)
        datay = data.loc[:,"NextYear"]
        datax = data[features]

        names = data['Munic'].drop_duplicates().values
        
        scaler = Normalizer().fit(datax)
        #scalx = scaler.transform(datax)
        #normedx = normalize(scalx)
        normedx = scaler.transform(datax)
        norm = pd.DataFrame(data=normedx, columns = features, 
                            index = data.index)


        norm_data = pd.merge(norm, datay, how = 'right', left_index=True, 
                             right_index=True, sort = False)
        
        
        prog = 0
       
        for nn in names:
            
            test = norm_data.loc[data['Munic'] == nn]
            train = norm_data.loc[data['Munic'] != nn]

           
            y = train["NextYear"]
            x = train[features]
            
            x_test = test[features]
                

            names = data.loc[test.index, 'Munic&Year']
            rates = data.loc[test.index, 'NextYear']

            if mdl == 1 :
            
                clf = RandomForestRegressor(n_estimators=200)
                model = clf.fit(x, y)

                
            elif mdl == 2 :
                
                clf = RandomForestRegressor(n_estimators = 500)
                abc = AdaBoostClassifier(n_estimators = 100,
                            base_estimator = clf, learning_rate = 1)
                model = abc.fit(x, y)
                
            elif mdl == 3 :
                
                gb = GradientBoostingRegressor(n_estimators=500, learning_rate = 1,
                                        warm_start=True)
                model = gb.fit(x, y)
            
           # if nn % 10 == 0 : 
            if prog % 7 < 2 :
                print(sorted(zip(map(lambda x: round(x, 4), 
                        model.feature_importances_), features), reverse=True))

            est = model.predict(x_test)
                
                
            aa = {'Munic&Year':names, 'model': est, 'R/E': rates}
            add= pd.DataFrame(aa)

            out = out.append(add)
            prog += len(test)
            print(str(round(prog/len(data)*100,2)) + '% Complete')
            
            
        out.to_csv("./Model/AllYears_Regressed.csv")

#file = "./Data/Final_Rating_Data_D.csv"
####################################################################
####################################################################
# df - designated file. Similar to the previous routine, this uses a random forest classifier to predict the ratings of
# a given set of WUSs. This routine, however, takes two files as input, using file1 as the data to train the model, and
# file2 as the test data whos ratings will be predicted.


    def voter(self, file, step) :

        data = pd.read_csv(file)#, index_col = 'Munic&Year')
        features = list(data.columns[1:-1])

        out = pd.Series()
        NN = len(data)/step
        NN = int(NN)
        datay = data["Rating"]
        datax = data[features]

        names = data['Munic'].drop_duplicates().values

        
        scaler = StandardScaler().fit(datax)
        scalx = scaler.transform(datax)
        normedx = normalize(scalx)
        norm = pd.DataFrame(data=normedx, columns = features, 
                            index = data.index)


        norm_data = pd.merge(norm, datay, how = 'right', left_index=True, 
                             right_index=True, sort = False)
        

        prog = 0
       
        for nn in names:

            test = norm_data.loc[data['Munic'] == nn]
            train = norm_data.loc[data['Munic'] != nn]

            y = train["Rating"]
            x = train[features]
            
            x_test = test[features]
                #y_test = test['Rating']

            print(str(nn+1) + " of " + str(NN))

            names = data.loc[test.index, 'Munic&Year']
            rates = data.loc[test.index, 'Rating']


            
            clf = RandomForestClassifier(n_estimators=500)
            gb = GradientBoostingClassifier(n_estimators=500, learning_rate = 1,
                                        warm_start=True)
            C = 1.0
            svc = svm.SVC(kernel='rbf', C=C)
            
            vote_model = VotingClassifier(estimators = [('rf', clf), 
                            ('gb', gb), ('svm',svc)])#, voting = 'soft')
    
            model = vote_model.fit(x, y)

            est = model.predict(x_test)
                
                
            aa = {'Munic&Year':names, 'model': est, 'Given': rates}
            add= pd.DataFrame(aa)

            out = out.append(add)
                
        out.to_csv("./Model/AllYears_Voted.csv")

    #########################################################################
        ################################################################
        
    """
    This is a routine that utilizes an adaboosting random forest classifier to predict the rating of all given WUSs by cycling through
    the entire set of municipalities.
    """
    def classifier_ADA(self, file) :

        data = pd.read_csv(file)
        features = list(data.columns[1:-1])
        print(features)
        #this = {"Municipality" : [], "Given" : [], "Model" : []}
        
        out = pd.Series()
        NN = len(data)/step
        NN = int(NN)
        
        datay = data["Rating"]
        datax = data[features]
        scaler = StandardScaler().fit(datax)
        scalx = scaler.transform(datax)
        normedx = normalize(scalx)
        norm = pd.DataFrame(data=normedx,columns = features, index=datay.index)

        norm_data = pd.merge(norm, datay, left_index=True, right_index = True)
                
        names = data['Munic'].drop_duplicates().values
        
        prog = 0
       
        for nn in names:

            test = norm_data.loc[data['Munic'] == nn]
            train = norm_data.loc[data['Munic'] != nn]
        
            y = train["Rating"]
            x = train[features]
            
            x_test = test[features]
            #y_test = test["Rating"]
            names = data.loc[test.index, 'Munic&Year']

            #clf = RandomForestClassifier(n_estimators=200), base_estimator = clf
            abc = AdaBoostClassifier(n_estimators = 500, learning_rate = 1)
            model = abc.fit(x, y)
            
            if nn % 10 == 0 : 
                print(sorted(zip(map(lambda x: round(x, 4), 
                                 model.feature_importances_), features), reverse=True))
            print(str(nn+1) + " of " + str(NN+1))

            est = model.predict(x_test)
            
            aa = {'model': est, 'Munic&Year': names}
            add= pd.DataFrame(aa)

            out = out.append(add)
        
        out = pd.merge(out, datay, left_index = True, right_index = True)

        out.to_csv("./Model/AllYears_ADA_Ratings.csv")
        
   

    #########################################################################
        ################################################################
        
    """
    This is a routine that utilizes a gradient boosting random forest classifier to predict the rating of all given WUSs using a gradient boosting random forest method.
    """
    def classifier_Gradient(self, file) :

        data = pd.read_csv(file)
        features = list(data.columns[1:-1])
        print(features)
        #this = {"Municipality" : [], "Given" : [], "Model" : []}
        
        out = pd.Series()
        
        datay = data["Rating"]
        datax = data[features]
        scaler = StandardScaler().fit(datax)
        scalx = scaler.transform(datax)
        normedx = normalize(scalx)
        norm = pd.DataFrame(data=normedx,columns = features, index=datay.index)

        norm_data = pd.merge(norm, datay, left_index=True, right_index = True)
        
        names = data['Munic'].drop_duplicates().values
        
        prog = 0
       
        for nn in names:

            test = norm_data.loc[data['Munic'] == nn]
            train = norm_data.loc[data['Munic'] != nn]

            y = train["Rating"]
            x = train[features]
            
            x_test = test[features]
            #y_test = test["Rating"]
            names = data.loc[test.index, 'Munic&Year']

            #clf = RandomForestClassifier(n_estimators=200), base_estimator = clf
            abc = AdaBoostClassifier(n_estimators = 500, learning_rate = 1)
            model = abc.fit(x, y)
            
            if nn % 10 == 0 : 
                print(sorted(zip(map(lambda x: round(x, 4), 
                                 model.feature_importances_), features), reverse=True))
            print(str(nn+1) + " of " + str(NN+1))

            est = model.predict(x_test)
            
            aa = {'model': est, 'Munic&Year': names}
            add= pd.DataFrame(aa)

            out = out.append(add)
        
        out = pd.merge(out, datay, left_index = True, right_index = True)

        out.to_csv("./Model/AllYears_ADA_Ratings.csv")



########################################################################
########################################################################
# Feature Selection routine that will return an normalized, ordered list of the fields that correlate most with
    # the independent variable. For instance, population has the largest correlation with WUS rating with a coefficient
    # of approx. ~0.35. Adding all coefficients will return 1 (normalized).


    def feature_select(self, file1) :


        #normed = pd.read_csv(file2)
        data = pd.read_csv(file1)

        #data = pd.merge(normed, reg, how='outer', on='Municipality')
        #print(data.head())

        features = list(data.columns[1:-1])
        print(features)

        x = data[features]
        y = data["Rating"]

        rf = RandomForestClassifier(n_estimators=5000)
        rf.fit(x, y)
        print("Features sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features), reverse=True))


########################################################################
########################################################################
    # SVM - Support Vector Machine. This routine attempts to classify each municipality relative to the rating system,
    # by grouping municipalities based on their position in phase space according to the data.
    # Types of svm: RBF, linear, poly (polynomial), sigmoid

    def svm_class(self, file, step) :

        data = pd.read_csv(file, index_col = 'Munic&Year')
        #outy = {"Munic&Year": [], 'Rating': []}
        out = pd.Series()

        features = list(data.columns[0:-1])
        print(features)
        
        names = data['Munic'].drop_duplicates().values
        
        prog = 0
       
        for nn in names:

            test = norm_data.loc[data['Munic'] == nn]
            train = norm_data.loc[data['Munic'] != nn]
                
            x_test = test[features]
            y_test = test['Rating']
           # test = data.iloc[b:e]
           # train = pd.merge(data.iloc[:b], data.iloc[e:], how='outer')

            Y = train["Rating"]
            X = train[features]
            C = 1.0
            svc = svm.SVC(kernel='rbf', C=C).fit(X, Y)

        #for m in data.index :
            #print(m)
            #munic_data = data.loc[m][features]
            #munic_data = munic_data.reshape(1,-1)
            #print(munic_data)
            

        
            r = svc.predict(x_test)
            print(str(nn+1) + " of " + str(NN))
            print("Accuracy:",metrics.accuracy_score(y_test, r))

        
            #print(m, ", ", r)
            aa = {'model': r, 'rating' : y_test}
            add= pd.DataFrame(aa)

            #out = pd.concat([out, add])
            out = out.append(add)
        #data.rename(columns = {'Rating':'Given'})
        #out = pd.concat([out, data['Rating']], axis = 1)

        out.to_csv('./Model/SVM_Rating_allYears.csv')



