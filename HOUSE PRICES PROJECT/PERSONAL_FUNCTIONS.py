def catconsep(a):
    cat = list(a.columns[a.dtypes=='object'])
    con = list(a.columns[a.dtypes!='object'])
    return cat , con


def replacer(a):
    cat,con = catconsep(a)
    
    for i in a.columns:
        if i in cat:
            mode=a[i].mode()[0]
            a[i] = a[i].fillna(mode)
        
        else:
            mean=a[i].mean()
            a[i] = a[i].fillna(mean)
            
    print('missing values replaced in dataframe')
    


def evaluate_model(xtrain, ytrain, xtest, ytest, model):
    # Predict train and test results
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    # Get metrics 
    from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
    tr_mse = mean_squared_error(ytrain, ypred_tr)
    tr_rmse = tr_mse**(1/2)
    tr_mae = mean_absolute_error(ytrain, ypred_tr)
    tr_r2 = r2_score(ytrain, ypred_tr)
    # Print all train results
    print('Training Results : ')
    print(f'MSE : {tr_mse:.2f}')
    print(f'RMSE: {tr_rmse:.2f}')
    print(f'MAE : {tr_mae:.2f}')
    print(f'R2  : {tr_r2:.4f}')
    # Testing results
    ts_mse = mean_squared_error(ytest, ypred_ts)
    ts_rmse = ts_mse**(1/2)
    ts_mae = mean_absolute_error(ytest, ypred_ts)
    ts_r2 = r2_score(ytest, ypred_ts)
    # Print all train results
    print('\n====================================\n')
    print('Testing Results : ')
    print(f'MSE : {ts_mse:.2f}')
    print(f'RMSE: {ts_rmse:.2f}')
    print(f'MAE : {ts_mae:.2f}')
    print(f'R2  : {ts_r2:.4f}')
    
    
def r2_adj(xtrain, ytrain, model):
    r2 = model.score(xtrain, ytrain)
    N = xtrain.shape[0]
    P = xtrain.shape[1]
    num = (1-r2)*(N-1)
    den = N-p-1
    res = 1 - num/den
    return res