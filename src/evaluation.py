import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def evaluation_metrics(real, pred):
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = mean_squared_error(real, pred, squared=False)
    mape = 100 * mean_absolute_percentage_error(real, pred)
    return {'mae': round(mae, 2), 'mse': round(mse, 2), 'rmse': round(rmse, 2), 'mape': round(mape, 2)}


def mape_sku(mape_name, cluster_test_lags, model, scaler, threshold=25):
    df_mape = pd.DataFrame(columns=['Country_Code', 'Material_Code', mape_name])
    sku_list = cluster_test_lags[['Country_Code', 'Material_Code']].drop_duplicates().to_numpy()
    for sku in sku_list:
        test_data = cluster_test_lags[(cluster_test_lags['Material_Code'] == sku[1]) & (cluster_test_lags['Country_Code'] == sku[0])]
        target = test_data['Demand']
        features = test_data.loc[:, test_data.columns != 'Demand']
        features = scaler.transform(features)

        pred = model.predict(features)
        e = evaluation_metrics(pred, target)
        if threshold is not None:
            if e['mape'] <= threshold:
                vals = pd.DataFrame([{'Country_Code': sku[0], 'Material_Code': sku[1], mape_name: e['mape']}])
                df_mape = df_mape.append(vals, ignore_index=True)
            else: 
                pass
        else:
            vals = pd.DataFrame([{'Country_Code': sku[0], 'Material_Code': sku[1], mape_name: e['mape']}])
            df_mape = df_mape.append(vals, ignore_index=True)
            
    return df_mape.sort_values(by=mape_name), df_mape[mape_name].mean(), df_mape[mape_name].median()


def mape_sku_client_forecast(mape_name, cluster_test_lags, target, model, scaler):
    # cluster_test_lags = cluster_test_lags.drop(columns=['Date'])
    df_mape = pd.DataFrame(columns=['Country_Code', 'Material_Code', mape_name])
    sku_list = cluster_test_lags[['Country_Code', 'Material_Code']].drop_duplicates().to_numpy()
    for sku in sku_list:
        test_data = cluster_test_lags[(cluster_test_lags['Material_Code'] == sku[1]) & (cluster_test_lags['Country_Code'] == sku[0])]
        Index = 'country_' + str(sku[0]) + '_material_' + str(sku[1])
        try:
            _target = target.loc[Index]['Demand']
            features = test_data.loc[:, test_data.columns != 'Demand']
            features = scaler.transform(features)
            pred = model.predict(features)
            e = evaluation_metrics(pred, _target)
            vals = pd.DataFrame([{'Country_Code': sku[0], 'Material_Code': sku[1], mape_name: e['mape']}])
            df_mape = df_mape.append(vals, ignore_index=True)
        except:
            pass

    return df_mape.sort_values(by=mape_name), df_mape[mape_name].mean(), df_mape[mape_name].median()
