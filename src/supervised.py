import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from evaluation import evaluation_metrics, mape_sku
from cluster_data import Cluster_Data
from datetime import timedelta


def multi_step_mape(df_cluster, model, mape_name, scaler):
    """ 16 and 8 step forecast evaluation """
    mape_name_8 = '8 step MAPE ' + str(mape_name)
    # mape_name_16 = '16 step MAPE ' + str(mape_name)
    df_multi_step_mape = pd.DataFrame(columns=['SKU', mape_name_8])

    sku_list = df_cluster['sku_list']
    # print(sku_list)
    for sku in sku_list:
        # print(sku)

        sku_info = df_cluster.copy()
        sku_data = df_cluster['data'].loc[sku]
        sku_info['data'] = sku_data

        # cluster data
        c_data = Cluster_Data(cluster=sku_info)

        # data split
        cluster_train, cluster_test = c_data.train_test_step_split(split_steps=8)

        df_forecast = pd.DataFrame()
        for i in range(8):

            last_row = cluster_train.tail(1)
            # print(last_row)

            # train and test datasets with lags
            constant_cols = ["Zone", "Country_Code", "Material_Code"]
            dependent_cols = []
            cols_to_drop = []

            cluster_train_lags = c_data.training_features(
                cluster_train,
                constants=constant_cols,
                dependents=dependent_cols,
                to_drop=cols_to_drop,
                lags=8
            )

            # Target/Input Features split
            X_train_to_pred, y_train_to_pred = c_data.target_features_split(cluster_train_lags, target='Demand')
            row_to_pred = X_train_to_pred.tail(1)
            # print(row_to_pred)

            # Standardize
            row_to_pred_scaled = scaler.transform(row_to_pred)

            # Predict
            pred = model.predict(row_to_pred_scaled)

            forecast_row = pd.DataFrame(
                {'Zone': last_row['Zone'],
                 'Material_Code': last_row['Zone'],
                 'Country_Code': last_row['Zone'],
                 'Date': last_row['Date'] + timedelta(days=8),
                 'Demand': pred},
                index=[sku])
            # print(forecast_row)
            df_forecast = df_forecast.append(forecast_row)  # forecasting list

            # add predicted value to the training dataset input
            cluster_train = cluster_train.append(forecast_row)

        # evaluate 8 step forecast
        multistep_forecast_8 = df_forecast['Demand'].head(8).to_numpy()
        test_data_8 = cluster_test['Demand'].head(8).to_numpy()
        eval_8_step_pred = evaluation_metrics(test_data_8, multistep_forecast_8)

        # evaluate 16 step forecast
        # multistep_forecast_16 = df_forecast['Demand'].head(13).to_numpy()
        # test_data_16 = cluster_test['Demand'].head(13).to_numpy()
        # eval_16_step_pred = evaluation_metrics(test_data_16, multistep_forecast_16)

        # Save results per SKU
        result_multi_step = pd.DataFrame({
            'SKU': sku,
            mape_name_8: [eval_8_step_pred['mape']]
            # mape_name_16: [eval_16_step_pred['mape']]
        })
        df_multi_step_mape = df_multi_step_mape.append(result_multi_step, ignore_index=True).sort_values(by=[mape_name_8])

    return df_multi_step_mape, df_multi_step_mape.mean(), df_multi_step_mape.median()


class XGBoost(object):

    def __init__(self, max_depth=None, n_estimators=None, eta=None):
        self.regressor = XGBRegressor() if max_depth is None and n_estimators is None and eta is None else XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, eta=eta)

    def get_model(self):
        return self.regressor

    def train(self, X_train, y_train):
        self.get_model().fit(X_train, y_train)

    def test(self, X_test, y_test):
        pred = self.predict(X_test)
        metrics = evaluation_metrics(y_test, pred)
        return metrics

    def predict(self, data):
        return self.get_model().predict(data)

    def eval_sku(self, test_lags, scaler, threshold):
        metrics = mape_sku("MAPE_XB", test_lags, self.get_model(), scaler, threshold)
        return metrics

    def multistep_eval_sku(self, df_cluster, scaler):
        metrics = multi_step_mape(df_cluster, self.get_model(), 'XB', scaler)
        return metrics

    def mlflow_pipeline(self, X_train, y_train, X_test, y_test, scaler=None, threshold=None, cluster_model_id=None, sku_eval=None, multistep_eval=None, run_name="XGBoost Model Training", model_name="xg_model"):
        # Useful for multiple runs (only doing one run in this sample notebook)
        with mlflow.start_run(run_name=run_name) as run:
            self.train(X_train, y_train)
            metrics = self.test(X_test, y_test)

            # Log parameter, metrics, and model to MLflow
            model_params = self.get_model().get_params()
            mlflow.log_param("max_depth", model_params['max_depth'])
            mlflow.log_param("n_estimators", model_params['n_estimators'])
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("mse", metrics['mse'])
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("mape", metrics['mape'])

            if cluster_model_id is not None:
                mlflow.set_tag("cluster_model_id", cluster_model_id)

            if sku_eval is not None:
                eval_sku = self.eval_sku(sku_eval, scaler, threshold)
                mlflow.log_metric("sku_mape_mean", eval_sku[1])
                mlflow.log_metric("sku_mape_median", eval_sku[2])

            if threshold is not None:
                mlflow.log_metric("sku_mape_mean_threshold", threshold)

            if multistep_eval is not None:
                multistep_eval_sku = self.multistep_eval_sku(multistep_eval, scaler)
                mlflow.log_metric("step_8_mape", multistep_eval_sku[1][0])
                mlflow.log_metric("step_8_median", multistep_eval_sku[2][0])
                # mlflow.log_metric("step_16_mape", multistep_eval_sku[1][1])
                # mlflow.log_metric("step_16_median", multistep_eval_sku[2][1])

            mlflow.sklearn.log_model(self.get_model(), model_name)

        return run


class Random_Forest(object):

    def __init__(self, max_depth=None, n_estimators=None):
        self.regressor = RandomForestRegressor() if max_depth is None and n_estimators is None else RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    def get_model(self):
        return self.regressor

    def train(self, X_train, y_train):
        self.get_model().fit(X_train, y_train)

    def test(self, X_test, y_test):
        pred = self.predict(X_test)
        metrics = evaluation_metrics(y_test, pred)
        return metrics

    def predict(self, data):
        return self.get_model().predict(data)

    def eval_sku(self, test_lags, scaler, threshold):
        metrics = mape_sku("MAPE_RF", test_lags, self.get_model(), scaler, threshold)
        return metrics

    def multistep_eval_sku(self, df_cluster, scaler):
        # print(df_cluster, self.get_model(), 'RF', scaler)
        metrics = multi_step_mape(df_cluster, self.get_model(), 'RF', scaler)
        return metrics

    def mlflow_pipeline(self, X_train, y_train, X_test, y_test, scaler=None, threshold=None, cluster_model_id=None, sku_eval=None, multistep_eval=None, run_name="Random Forest Model Training", model_name="rf_model"):
        # Useful for multiple runs (only doing one run in this sample notebook)
        with mlflow.start_run(run_name=run_name) as run:
            self.train(X_train, y_train)
            metrics = self.test(X_test, y_test)

            # Log parameter, metrics, and model to MLflow
            model_params = self.get_model().get_params()
            mlflow.log_param("max_depth", model_params['max_depth'])
            mlflow.log_param("n_estimators", model_params['n_estimators'])
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("mse", metrics['mse'])
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("mape", metrics['mape'])
            
            if cluster_model_id is not None:
                mlflow.set_tag("cluster_model_id", cluster_model_id)

            if sku_eval is not None:
                eval_sku = self.eval_sku(sku_eval, scaler, threshold)                
                mlflow.log_metric("sku_mape_mean", eval_sku[1])
                mlflow.log_metric("sku_mape_median", eval_sku[2])
                
            if threshold is not None:
                mlflow.log_metric("mape_threshold", threshold)

            if multistep_eval is not None:
                multistep_eval_sku = self.multistep_eval_sku(multistep_eval, scaler)
                mlflow.log_metric("step_8_mape", multistep_eval_sku[1][0])
                mlflow.log_metric("step_8_median", multistep_eval_sku[2][0])
                # mlflow.log_metric("step_16_mape", multistep_eval_sku[1][1])
                # mlflow.log_metric("step_16_median", multistep_eval_sku[2][1])

            mlflow.sklearn.log_model(self.get_model(), model_name)

        return run


class Support_Vector(object):

    def __init__(self, kernel=None):
        self.regressor = SVR() if kernel is None else SVR(kernel=kernel)

    def get_model(self):
        return self.regressor

    def train(self, X_train, y_train):
        self.get_model().fit(X_train, y_train)

    def test(self, X_test, y_test):
        pred = self.predict(X_test)
        metrics = evaluation_metrics(y_test, pred)
        return metrics

    def predict(self, data):
        return self.get_model().predict(data)

    def eval_sku(self, test_lags, scaler, threshold):
        metrics = mape_sku("MAPE_SV", test_lags, self.get_model(), scaler, threshold)
        return metrics

    def multistep_eval_sku(self, df_cluster, scaler):
        metrics = multi_step_mape(df_cluster, self.get_model(), 'SV', scaler)
        return metrics

    def mlflow_pipeline(self, X_train, y_train, X_test, y_test, scaler=None, threshold=None, cluster_model_id=None, sku_eval=None, multistep_eval=None, run_name="SVR Model Training", model_name="sv_model"):
        # Useful for multiple runs (only doing one run in this sample notebook)
        with mlflow.start_run(run_name=run_name) as run:
            self.train(X_train, y_train)
            metrics = self.test(X_test, y_test)

            # Log parameter, metrics, and model to MLflow
            model_params = self.get_model().get_params()
            mlflow.log_param("kernel", model_params['kernel'])
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("mse", metrics['mse'])
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("mape", metrics['mape'])
            
            if cluster_model_id is not None:
                mlflow.set_tag("cluster_model_id", cluster_model_id)

            if sku_eval is not None:
                eval_sku = self.eval_sku(sku_eval, scaler)        
                mlflow.log_metric("sku_mape_mean", eval_sku[1])
                mlflow.log_metric("sku_mape_median", eval_sku[2])
                
            if threshold is not None:
                mlflow.log_metric("mape_threshold", threshold)

            if multistep_eval is not None:
                multistep_eval_sku = self.multistep_eval_sku(multistep_eval, scaler)
                mlflow.log_metric("step_8_mape", multistep_eval_sku[1][0])
                mlflow.log_metric("step_8_median", multistep_eval_sku[2][0])
                # mlflow.log_metric("step_16_mape", multistep_eval_sku[1][1])
                # mlflow.log_metric("step_16_median", multistep_eval_sku[2][1])

            mlflow.sklearn.log_model(self.get_model(), model_name)

        return run
