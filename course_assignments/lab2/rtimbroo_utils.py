# -*- coding: utf-8 -*-
# RTIMBROO UTIL FUNCTIONS

def getLogger(level=20):
    """
    name: Get a Logger
    
    Args:
        param1 (int): Log level
    
    Returns:
        Returns a logger object
    
    """
    import logging
    print(level)
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
    l = logging.getLogger(__name__)
    
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

def getLogger2(configFile):
    """
    Args:
    
    Returns: 
    """
    import logging
    import logging.config
    import yaml
    #import coloredlogs

    with open(configFile, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        #coloredlogs.install()

        logger = logging.getLogger(__name__)

        logger.debug('This is a debug message')

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
def getNaNIndexes(df,att):
    import numpy as np
    n = np.where(df[att].isnull()==True)
    return list(n[0])
##---------------------------------------------

'''
-----------------------------------------------
Facebook Prophet functions
'''
# function to format and create a prophet model
def beProphet(label,components,predPeriods):
    from fbprophet import Prophet
    model={}
    # restructure the dataframe to fit prophet
    df = pd.DataFrame(components)
    df = df.reset_index()
    logValue = 'log_'+label
    df[logValue] = np.log(df[label])
    dfProphet = df.rename(index=str, columns={logValue:'y','index':'ds'})
    dfProphet = dfProphet.loc[:,['y','ds']]
    
    # setting uncertainty interval to 95%
    zipModel = Prophet(interval_width=0.95)
    zipModel_fit = zipModel.fit(dfProphet)
    model['model_fit'] = zipModel_fit
    # make future dates dataframe
    future_dates = zipModel.make_future_dataframe(periods=predPeriods, freq='M', include_history=True)
    
    # model
    forecast = zipModel.predict(future_dates)
    model['model_forecast'] = forecast
    return(model)

'''

'''
# Facebook Prophet requires columns to be in a specific format
def dfTransformForProphet(df,cols,index):
    from fbprophet import Prophet
    df = df.drop(columns=cols)
    df = df.set_index(index)
    df = df.T
    df.index = pd.to_datetime(df.index)
    return (df)



'''
Calculate annual price value change by zipcode
'''
def calcPriceDelta(df,dateSeries):
    years = [str(i) for i in range(1997,2020)]
    dateCols = {}
    for y in years:
        dateCols[y] = getDateColumns(dateSeries,y)
    
    yearAvg = {}
    for d in dateCols:
        subSet = m1[m1.Date.isin(dateCols[d])]
        anualAvg = subSet.iloc[:,1].mean()
        yearAvg[d] = anualAvg
        #break
    #print('Yearly Price Averages: {0}\n'.format(yearAvg))

    thisYear = ''
    priorYear = ''
    i = 0
    #while(i<=len(yearAvg)):
    prior_years = []
    priceDelta = {}
    for i, year in enumerate(yearAvg):
        thisYear = year
        
        if not i == 0:
            priorYear = prior_years[i-1] # not first year
        else:
            priorYear = year # is first year
        #print('This Year: {0}'.format(thisYear))
        #print('Prior Year: {0}\n'.format(priorYear))
        
        # set prior year list
        prior_years.append(thisYear)
    
        # calculate delta between years
        #print('This Year Average Price: {0}'.format(yearAvg[thisYear]))
        #print('Prior Year Average Price: {0}\n'.format(yearAvg[priorYear]))
        
        delta = yearAvg[thisYear] - yearAvg[priorYear]
        #print('This Year: {0}, Prior Year: {1}, Price Delta: {2}\n'.format(thisYear,priorYear,delta))
        
        priceDelta[thisYear] = delta
    #print('Yearly Price Change:{0}\n'.format(priceDelta))
    return(pd.DataFrame([yearAvg,priceDelta], index=['Yearly_Price_Avg','Yearly_Price_Delta']))
    
    
    
    

'''
util function for getting values for dates
'''
def getDateColumns(series,d):   
    return([i for i in series if d in i])

'''
----------------------------------------------
Random Grid Search Functions
Refe
-----------------------------------------------
'''
'''

'''
# Utility function to report best scores
def rgs_reportBestScores(results, n_top=3):
    import numpy as np
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



