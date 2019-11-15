# -*- coding: utf-8 -*-
'''
Basic Logger
'''
def getSHLogger(name='stream_handler',level=20):
    """
    name: Get a Logger
    
    Args:
        param1 (int): Log level
    
    Returns:
        Returns a logger object
    
    """
    import logging
    #print(level)
    loglevel = None
    if level == 10: # DEBUG
        loglevel = logging.DEBUG;
    elif level == 20: # INFO
        loglevel = logging.INFO;
    elif level == 30: # WARNING
        loglevel = logging.WARNING;
    elif level == 40: # ERROR
        loglevel = logging.ERROR;
    elif level == 50: # CRITICAL
        loglevel = logging.CRITICAL;
    else:
        loglevel = logging.DEBUG;
    
    
    isSimpleOutput = True
    l = logging.getLogger(name)
    
    if not l.hasHandlers():
        f = None
        l.setLevel(loglevel)
        h = logging.StreamHandler()
        if isSimpleOutput:
            f = logging.Formatter('%(message)s')
        else:
            f = logging.Formatter('Date Time: %(asctime)s | Level: %(levelname)s | Message: %(message)s')
        
        h.setFormatter(f)
        l.addHandler(h)
        l.setLevel(loglevel)
        l.handler_set = True
        
    return l

'''

'''
def getFileLogger(logDir, logFileName, name='file_logger', level=20):
    import logging
    #print(level)
    loglevel = None
    if level == 10: # DEBUG
        loglevel = logging.DEBUG;
    elif level == 20: # INFO
        loglevel = logging.INFO;
    elif level == 30: # WARNING
        loglevel = logging.WARNING;
    elif level == 40: # ERROR
        loglevel = logging.ERROR;
    elif level == 50: # CRITICAL
        loglevel = logging.CRITICAL;
    else:
        loglevel = logging.DEBUG;
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # logging file handler
    fh = logging.FileHandler(f'{logDir}{logFileName}.log')
    fh.setLevel(logging.DEBUG)
    
    # create console handler with higher level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # create formatter and add it to the handlers
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    
    fh.setFormatter(file_formatter)
    ch.setFormatter(console_formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # add the handler to the root logger
    #logging.getLogger('').addHandler(console)
    
    return logger
    

    
    


'''
Find NaN values in dataframe
'''
def getNaNCount(df):
    
    totNaNCnt = df.isnull().sum().sum()
    nanRowsCnt = len(df[df.isnull().T.any().T])
    
    #print("Total NaN Cnt {0}".format(totNaNCnt))
    #print("Total NaN Rows Cnt {0}".format(nanRowsCnt))
    
    return totNaNCnt, nanRowsCnt
    
'''

'''
def findColumnsNaN(df,rowIndex=True):
    naCols = []
    for col in list(df.columns):
        #print(coachesDf[col].isnull().sum().sum())
        if df[col].isnull().sum().sum() > 0:
            print("Column: {0} has: {1} NaN values".format(col,df[col].isnull().sum().sum()))
            if rowIndex: print("{0}: {1}\n".format(col,getNaNIndexes(df,col)))
   
'''

'''  
def getColumnsNaNCnts(df,rowIndex=True):
    naCols = []
    for col in list(df.columns):
        #print(coachesDf[col].isnull().sum().sum())
        if df[col].isnull().sum().sum() > 0:
            naCols.append((col,df[col].isnull().sum().sum()))
    
    #print(len(naCols))
    if not len(naCols) == 0:
        return(naCols)
    else:
        return(0)
           
'''

'''
def getNaNIndexes(df,att):
    import numpy as np
    n = np.where(df[att].isnull()==True)
    return list(n[0])


'''
# find missing data
'''
def missing_data(data):
    import pandas as pd
    import numpy as np
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

'''
convert NaN values to means of the column values
'''
def nan2Mean(df):
    for x in list(df.columns.values):
        #print("___________________"+x)
        #print(df[x].isna().sum())
        df[x] = df[x].fillna(df[x].mean())
       #print("Mean-"+str(df[x].mean()))
    return df

'''
# function used for clocking processing time to build/run models
'''
from contextlib import contextmanager
from timeit import default_timer
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start



'''
perform label encoding
'''
def labelEncoding(train_df, test_df):
    from sklearn.preprocessing import LabelEncoder
    # Label Encoding
    for f in train_df.columns:
        if  train_df[f].dtype=='object': 
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))  
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    return(train_df,test_df)



'''
# Confusion matrix
'''
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, saveImgAs, normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues):
    import numpy as np
    import itertools
    #from sklearn.metrics import confusion_matrix
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(saveImgAs, dpi=300)
    plt.show()   
   
   
'''
Visualize different variations of training image
'''
def visualize_image_variations(X_train, y_train, vis_img ,numImgsToVis, imgShape, saveImgAs):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(nrows=5, ncols=4, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(numImgsToVis):
        img = X_train[y_train == vis_img][i].reshape(imgShape[0],imgShape[1])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(saveImgAs, dpi=300)
    plt.show()
    
    
'''
Addapted from IST 718 - W8
'''
def plot_nn_training_curve(history,imageName):
    import matplotlib.pyplot as plt
    #%matplotlib inline
    
    colors = ['#e66101','#fdb863','#b2abd2','#5e3c99']
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    
    with plt.style.context("ggplot"):
        plt.figure(figsize=(8, 8/1.618))
        plt.plot(epochs, accuracy, marker='o', c=colors[3], label='Training accuracy')
        plt.plot(epochs, val_accuracy, c=colors[0], label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure(figsize=(8, 8/1.618))
        plt.plot(epochs, loss, marker='o', c=colors[3], label='Training loss')
        plt.plot(epochs, val_loss, c=colors[0], label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
        plt.savefig(f'{imageName}.png', dpi=300)
        plt.show()


'''
# Sample some images in the dataset
'''
def plot_sample_images(instances, imageSize=28, images_per_row=10, **options):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt                # used for 2D plotting
    from matplotlib.pyplot import figure
    
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    
    size = imageSize
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    
    image = np.concatenate(row_images, axis=0)
    
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    plt.show()


'''
Prints out the Computational Execution Time
'''
def show_time(diff):
   m, s = divmod(diff, 60)
   h, m = divmod(m, 60)
   s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
   print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))
   
   
'''
# Takes in a classifier, calculates the training + prediction times and accuracy score, returns a model
'''
def train_score_predict(clf, X, y, X_predict, y_predict, record_performance, sh_logger, type='classification'):
    import time
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error as rmse

    # Train
    start = time.time()
    model = clf.fit(X,y)
    end = time.time()
    if sh_logger.debug: print('Training time: ')
    if sh_logger.debug: show_time(end - start)
    
    # save performance variables
    record_performance['TrainTime'].append(end - start)

    # Predict
    start = time.time()
    if(type=='classification'):
        record_performance['PredictAccuracyScore'].append(accuracy_score(y_predict, model.predict(X_predict)))
    else:
        record_performance['PredictAccuracyScore'].append(rmse(y_test, model.predict(X_test)))
        
    end = time.time()
    record_performance['PredictTime'].append(end - start)
    
    if sh_logger.debug: print('\nPrediction time: ')
    if sh_logger.debug: show_time(end - start)
    
    return model

'''
# Takes in a classifier, calculates the training + prediction times and accuracy score, returns a model
'''
def train_validate_classifier(clf, modelName, X, y, X_predict, y_predict, record_performance, sh_logger, modelDir, n_classes, labels,type='classification'):
    from datetime import datetime
    import time
    import pickle
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics import mean_squared_error as rmse
        
    # Train the model
    record_performance['ModelName'].append(modelName)
    start = time.time()
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model Training at time: {datetime.now().time()}')
        model_fit = clf.fit(X,y)
        t = elapsed()
        if sh_logger.debug: print(f'Training Complete - Execution Time: [{t}]')
        record_performance['TrainTime'].append(t)
    
    # save model's to file
    #save model to file
    with open(f'{modelDir}{modelName}','wb') as f:
        pickle.dump(clf,f)

    with open(f'{modelDir}{modelName}_fit','wb') as f:
        pickle.dump(model_fit,f)
    
    #validate prediction
    start = time.time()
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model Prediction at time: {datetime.now().time()}')
        clf_val_pred = clf.predict(X_predict)
        t = elapsed()
        if sh_logger.debug: print(f'Predicting on Validation dataset Complete - Execution Time: [{t}]')
        record_performance['TestTime'].append(t)
    #save model to file
    with open(f'{modelDir}{modelName}_val_predictions','wb') as f:
        pickle.dump(clf_val_pred,f)
    
    #score validation prediction accuracy
    start = time.time()
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model Accuracy Scoring at time: {datetime.now().time()}')
        clf_val_score = accuracy_score(y_predict,clf_val_pred)
        if sh_logger.debug: print(f'Scoring Complete - Execution Time: [{t}]')
        record_performance['ScoreTime'].append(t)
        record_performance['TestAccuracyScore'].append(clf_val_score)
    
    #print classification report table
    targetNames = ["Class{}".format(i) for i in range(n_classes)]
    if sh_logger.info: print(f'\n{classification_report(y_predict, clf_val_pred, target_names=targetNames)}')  

    #print confusion matrix report
    cm = confusion_matrix(y_predict,clf_val_pred, labels=labels)
    if sh_logger.info: print(f'\nConfusion Matrix Report:\n{cm}')

    # plot confusion matrix evaluation
    if sh_logger.info: plot_confusion_matrix(cm,classes=labels)
    
    
    return clf

'''
# Takes in a classifier, calculates the training + prediction times and accuracy score, returns a model
'''
def grid_search(clf, modelName, params, X, y, X_predict, y_predict, record_performance, sh_logger, modelDir, n_classes, labels, type='classification'):
    from datetime import datetime
    import time
    import pickle
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error as rmse
    
    # Train the model
    record_performance['ModelName'].append(modelName)
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model GridSearch Training at time: {datetime.now().time()}')
        model = GridSearchCV(clf, params, scoring='accuracy', n_jobs=-1, cv=5).fit(X,y).best_estimator_
        t = elapsed()
        if sh_logger.debug: print(f'Training Complete - Execution Time: [{t}]')
        record_performance['TrainTime'].append(t)
    
    #save model to file
    with open(f'{modelDir}{modelName}','wb') as f:
        pickle.dump(model,f)

    #validate prediction
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model GridSearch Prediction at time: {datetime.now().time()}')
        model_pred = model.predict(X_predict)
        t = elapsed()
        if sh_logger.debug: print(f'Predicting on Validation dataset Complete - Execution Time: [{t}]')
        record_performance['TestTime'].append(t)
    #save model to file
    with open(f'{modelDir}{modelName}_val_predictions','wb') as f:
        pickle.dump(model_pred,f)
    
    #score validation prediction accuracy
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model Accuracy Scoring at time: {datetime.now().time()}')
        model_score = accuracy_score(y_predict,model_pred)
        if sh_logger.debug: print(f'Scoring Complete - Execution Time: [{t}]')
        record_performance['ScoreTime'].append(t)
        record_performance['TestAccuracyScore'].append(model_score)
    
    #print classification report table
    targetNames = ["Class{}".format(i) for i in range(n_classes)]
    if sh_logger.info: print(f'\n{classification_report(y_predict, model_pred, target_names=targetNames)}')  

    #print confusion matrix report
    cm = confusion_matrix(y_predict,model_pred, labels=labels)
    if sh_logger.info: print(f'\nConfusion Matrix Report:\n{cm}')

    # plot confusion matrix evaluation
    if sh_logger.info: plot_confusion_matrix(cm,classes=labels)
    
    return model
    

'''
Returns best fit model
'''
def trainWithCrossValidate(clf, modelName, X, y, record_performance, sh_logger, modelDir, cv=5):
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import recall_score
    from datetime import datetime
    import time
    import pickle
    
    # score validation dataset with cross validation
    cv = cv
    scoring = ['precision_macro', 'recall_macro']
    start = time.time()
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model Cross Validate Training at time: {datetime.now().time()}\n')
        clf_cv = cross_validate(clf, X, y, scoring=scoring, cv=cv, 
                                return_train_score=True, return_estimator=True) 
        t = elapsed()
        if sh_logger.debug: print(f'Cross Validate Complete - Execution Time: [{t}]\n')
    
    if sh_logger.debug: print(f'Scorer Names: {sorted(clf_cv.keys())}')
    if sh_logger.info: print(f'Fit Time:               {clf_cv["fit_time"]}')
    if sh_logger.info: print(f'Score Time:             {clf_cv["score_time"]}')
    if sh_logger.info: print(f'Test Recall Scores:     {clf_cv["test_recall_macro"]}')
    if sh_logger.info: print(f'Test Precision Scores:  {clf_cv["test_precision_macro"]}')
    if sh_logger.info: print(f'Train Recall Scores:    {clf_cv["train_recall_macro"]}')
    if sh_logger.info: print(f'Train Precision Scores: {clf_cv["train_precision_macro"]}') 
    
    #save model to file
    with open(f'{modelDir}{modelName}_cv','wb') as f:
        pickle.dump(clf,f)
    
    #model prediction
    bestFit = clf_cv["test_precision_macro"].argmax()
    clf_cv['estimator'][bestFit]

    record_performance['ModelName'].append(modelName)
    record_performance['TrainTime'].append(clf_cv["fit_time"][bestFit])
    record_performance['ScoreTime'].append(clf_cv["score_time"][bestFit])
    record_performance['TestAccuracyScore'].append(clf_cv["test_precision_macro"][bestFit])
    
    #return best predictor
    return clf_cv['estimator'][bestFit]

'''

'''
def predict_classifier(clf, modelName, X_test, y_true, record_performance, sh_logger, n_classes, labels, modelDir):
    import pandas as pd
    from datetime import datetime
    import numpy as np
    import time
    import pickle
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    t = 0.0
    with elapsed_timer() as elapsed:
        if sh_logger.debug: print(f'Starting Model Prediction at time: {datetime.now().time()}\n')
        clf_pred = clf.predict(X_test)
        t = elapsed()
        if sh_logger.debug: print(f'Complete Model Prediction - Execution Time: [{t}]\n')
        record_performance['PredictTime'].append(t)
    
    #save model to file
    with open(f'{modelDir}{modelName}_predictions','wb') as f:
        pickle.dump(clf,f)
     
    #correct and inccorrect
    correct = np.nonzero(clf_pred==y_true)[0]
    incorrect = np.nonzero(clf_pred!=y_true)[0]
    
    d = {'Label':y_true, 'Prediction':clf_pred}
    predictionsDf = pd.DataFrame(data=d)
    if sh_logger.debug: print(f'Classification DF Shape: {predictionsDf.shape}\n Head:\n{predictionsDf.head()}')
    
    # which test observations were miss classified
    missClassified_DT = predictionsDf[(predictionsDf['Label'] != predictionsDf['Prediction'])]
    
    if sh_logger.debug: print(f'Miss Classified DF Shape: {missClassified_DT.shape}')
    if sh_logger.debug: print(f'Miss Classified Percent: {missClassified_DT.shape[0]/y_true.size}')
    if sh_logger.info: print(f'Total Number of points: [{X_test.shape[0]}]  Mislabeled Points: [{(y_true != clf_pred).sum()}]')
    
    misLabeled = (y_true != clf_pred).sum()/X_test.shape[0]
    accuractelyLabeled = 1-misLabeled
    if sh_logger.info: print(f'Percent Accurately Labeled: [{accuractelyLabeled}]')
    
    clf_pred_score = accuracy_score(y_true,clf_pred)
    if sh_logger.info: print(f'Accuracy Score: [{clf_pred_score}]')
    
    record_performance['PredictAccuracyScore'].append(clf_pred_score)
    
    #print classification report table
    targetNames = ["Class{}".format(i) for i in range(n_classes)]
    if sh_logger.info: print(f'\n{classification_report(y_true, clf_pred, target_names=targetNames)}')  

    #print confusion matrix report
    cm = confusion_matrix(y_true,clf_pred, labels=labels)
    if sh_logger.info: print(f'\nConfusion Matrix Report:\n{cm}')

    # plot confusion matrix evaluation
    #if sh_logger.info: plot_confusion_matrix(cm,labels,f'{}.png')
    
    return clf_pred


'''
# Takes in model scores and plots them on a bar graph
'''
def plot_metric(model_scores, score='Accuracy'):
    #import time
    import matplotlib.pyplot as plt
    from matplotlib.pylab import rcParams
    
    rcParams['figure.figsize'] = 7,5
    plt.bar(model_scores['Model'], height=model_scores[score])
    xlocs, xlabs = plt.xticks()
    xlocs=[i for i in range(0,6)]
    xlabs=[i for i in range(0,6)]
    
    if(score != 'Prediction Times'):
        for i, v in enumerate(model_scores[score]):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
            
    plt.xlabel('Model')
    plt.ylabel(score)
    plt.xticks(rotation=45)
    plt.show()
    
'''
# Takes in training data and a model, and plots a bar graph of the model's feature importances
'''
def feature_importances(features, model, model_name, sh_logger, max_num_features=10):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    feature_importances = pd.DataFrame(columns = ['feature', 'importance'])
    feature_importances['feature'] = features
    feature_importances['importance'] = model.feature_importances_
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    feature_importances = feature_importances[:max_num_features]
    if sh_logger.debug: print(feature_importances)
    
    plt.figure(figsize=(12, 6));
    sns.barplot(x="importance", y="feature", data=feature_importances);
    plt.title(model_name+' features importance:');

'''
# Takes in training data and a model, and plots a bar graph of SHAP values
# see anaconda for more details on shap: https://anaconda.org/conda-forge/shap
'''
def shap_values(df, model, model_name):
    import shap # SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model
    
    shap_values = shap.TreeExplainer(model).shap_values(df)
    shap_values[:5]
    shap.summary_plot(shap_values, df.iloc[:1000,:])

