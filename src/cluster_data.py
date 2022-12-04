import numpy as np
import pandas as pd
from datetime import timedelta
# pip install adtk
from adtk.data import validate_series
from adtk.detector import InterQuartileRangeAD
from sklearn.preprocessing import StandardScaler


def create_lags(df, target_col='target', dependent_cols=[], constant_cols=[], lag_number=8):
    df_lags = pd.DataFrame()
    unique_ids = df.index.unique()

    for identifier in unique_ids:
        single_process_data = df.loc[identifier]  # 1

        data_cols = [target_col] + dependent_cols
        data = pd.DataFrame(single_process_data[data_cols].copy())  # 2
        data.columns = data_cols  # 2

        # last x values of the target variable as "lag" variables (the most recent one is the dependent feature (y))
        for i in range(1, lag_number + 1):  # 3
            data[target_col + '_lag_{}'.format(i)] = data[target_col].shift(i)

        # last x values of the target variable as "dependent_feature" variables
        for col in dependent_cols:
            for j in range(1, lag_number + 1):  # 4
                data[col + '_lag_{}'.format(j)] = data[col].shift(j)

        # rewrite constants
        for col in constant_cols:
            data[col] = single_process_data[col]  # 5

        # the shift operations in the loops create many partial results. They are useless, and we don't want them
        data = data.dropna()
        df_lags = pd.concat([df_lags, data], ignore_index=True)
    return df_lags


class Cluster_Data(object):

    def __init__(self, cluster):
        self.data = cluster['data']
        self.sku_list = cluster['sku_list']
        self.overview = cluster['overview']

    def get_data(self):
        return self.data

    def get_sku_list(self):
        return self.sku_list

    def get_overview(self):
        return self.overview

    def train_test_date_split(self, splitting_date):
        test = self.data[self.data['Date'] >= splitting_date]
        train = self.data[self.data['Date'] < splitting_date]
        return train, test

    def train_test_step_split(self, split_steps):
        max_date = self.data['Date'].max()
        step_date = max_date - timedelta(weeks=split_steps)
        # print(max_date, step_date)
        test = self.data[self.data['Date'] >= step_date]  # test = self.data[split_steps*(-1):]
        train = self.data[self.data['Date'] < step_date]  # train = self.data[:split_steps*(-1)]
        return train, test

    def outliers_interpolation(self, df, skus_list, c=2):
        """ Remove outliers from train dataset """
        for sku in skus_list:
            s = df.loc[sku]

            s = s[['Date', 'Demand']].set_index('Date')
            s = validate_series(s)

            iqr_ad = InterQuartileRangeAD(c=c)
            anomalies = iqr_ad.fit_detect(s)

            anomalies_list = anomalies[anomalies['Demand'] == True].reset_index()['Date'].to_list()
            # print(sku, anomalies_list, s.loc[s.index.isin(anomalies_list)])
            s.loc[s.index.isin(anomalies_list)] = np.NaN
            s = s.interpolate()

            df.at[sku, 'Demand'] = s['Demand'].to_list()
        return df

    def apply_rolling_mean(dataset, sku, mean_smoothing):
        """ Apply rolling mean """
        sku_data = dataset[dataset['SKU'] == sku]
        if mean_smoothing is True:
            sku_data['Demand'] = sku_data['Demand'].rolling(8, center=True).mean()

        s = sku_data[['Date', 'Demand']].set_index('Date')
        s = s.dropna()
        s = validate_series(s)

        return s, sku_data

    def calculate_statistical_features(self, dataframe):
        """ Statistical features """
        df_calc = dataframe['Demand']
        dataframe['mean'] = df_calc.groupby(['SKU']).mean()
        dataframe['std'] = df_calc.groupby(['SKU']).std()
        dataframe['skewness'] = df_calc.groupby(['SKU']).skew()
        dataframe['kurtosis'] = df_calc.groupby(['SKU']).apply(pd.DataFrame.kurt)
        dataframe['coef_var'] = dataframe['std'] / dataframe['mean']
        return dataframe

    def calculate_seasonal_features(self, dataframe):
        """ Seasonal features """
        dataframe['year'] = dataframe['Date'].dt.year
        dataframe['month'] = dataframe['Date'].dt.month
        dataframe['week'] = dataframe['Date'].dt.isocalendar().week
        return dataframe

    def calculate_statistical_features_before_split(self, dataframe):
        """ Statistical features from original data before data splitting """
        df_skus_overview = self.overview.reset_index()
        df_skus_overview_ = df_skus_overview[['SKU', 'mean', 'std', 'coef_var', 'kurtosis', 'skewness']].copy()
        dataframe = dataframe.merge(df_skus_overview_, on='SKU')
        dataframe = dataframe.set_index('SKU')
        return dataframe

    def training_features(self, dataframe, target='Demand', constants=[], dependents=[], to_drop=[], lags=8):
        """ Create lagged dataset for train and test """
        data_lags = create_lags(
            dataframe,
            target_col=target,
            dependent_cols=dependents,
            constant_cols=constants,
            lag_number=lags
        )
        data_lags = data_lags.drop(columns=to_drop)
        return data_lags

    def target_features_split(self, dataset, target=[]):
        """ Get target and input features """
        # target vector
        y = dataset[target]
        # feature matrix
        X = dataset.loc[:, dataset.columns != target]
        return X, y

    def set_scaler(self, train_dataframe):
        """ Scaler object """
        std_scaler = StandardScaler()
        std_scaler.fit(train_dataframe)
        return std_scaler

    def standardize(self, train, test):
        """ Standardize data """
        std_scaler = self.set_scaler(train)
        X_train_std = std_scaler.transform(train)
        X_test_std = std_scaler.transform(test)
        return X_train_std, X_test_std

# OLD

# def outliers_interpolation(df, skus_list, c=2):
#     # Remove outliers from train dataset
#     for sku in skus_list:
#         s = df.loc[sku]

#         s = s[['Date', 'Demand']].set_index('Date')
#         s = validate_series(s)

#         iqr_ad = InterQuartileRangeAD(c=c)
#         anomalies = iqr_ad.fit_detect(s)

#         anomalies_list = anomalies[anomalies['Demand'] == True].reset_index()['Date'].to_list()
#         print(sku, anomalies_list, s.loc[s.index.isin(anomalies_list)])
#         s.loc[s.index.isin(anomalies_list)] = np.NaN
#         s = s.interpolate()

#         df.at[sku, 'Demand'] = s['Demand'].to_list()
#     return df


# def calculate_statistical_features(dataframe):
#     """ Statistical features """
#     df_calc = dataframe['Demand']
#     dataframe['mean'] = df_calc.groupby(['SKU']).mean()
#     dataframe['std'] = df_calc.groupby(['SKU']).std()
#     dataframe['skewness'] = df_calc.groupby(['SKU']).skew()
#     dataframe['kurtosis'] = df_calc.groupby(['SKU']).apply(pd.DataFrame.kurt)
#     dataframe['coef_var'] = dataframe['std'] / dataframe['mean']
#     return dataframe


# def calculate_seasonal_features(dataframe):
#     """ Seasonal features """
#     dataframe['year'] = dataframe['Date'].dt.year
#     dataframe['month'] = dataframe['Date'].dt.month
#     dataframe['week'] = dataframe['Date'].dt.isocalendar().week
#     return dataframe


# def data_split(dataframe, date_to_split):
#     test = dataframe[dataframe['Date'] >= date_to_split]
#     train = dataframe[dataframe['Date'] < date_to_split]
#     return {'train': train, 'test': test}


# def target_features_split(train, test):
#     # target vector
#     y_train = train['Demand']
#     y_test = test['Demand']
#     # feature matrix
#     X_train = train.loc[:, train.columns != 'Demand']
#     X_test = test.loc[:, test.columns != 'Demand']
#     return X_train, y_train, X_test, y_test

