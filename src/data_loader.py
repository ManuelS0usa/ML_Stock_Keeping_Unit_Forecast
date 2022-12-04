import pandas as pd


class Data_Reader():
    """docstring for Data Reader"""

    def __init__(self, filePath="../data/data_2018_2021.xlsx", sheetName="data"):
        self.file_path = filePath
        self.sheet_name = sheetName

    def get_filePath(self):
        return self.file_path

    def set_filePath(self, filePath):
        self.file_path = filePath

    def get_sheetName(self):
        return self.sheet_name

    def set_sheetName(self, sheetName):
        self.sheet_name = sheetName

    def read_file(self):
        """ read file and remove uncessery data """
        df = pd.read_excel(self.file_path, engine="openpyxl", sheet_name=self.sheet_name)
        # reposition column
        column_to_move = df.pop("Zone")
        df.insert(0, "Zone", column_to_move)
        return df

    def get_data(self):
        """ Build dataframes by zone and consider a global one """
        df = self.read_file()
        df_zone_1 = df[df['Zone'] == 1].drop(columns=["Zone"])
        df_zone_2 = df[df['Zone'] == 2].drop(columns=["Zone"])
        df_zone_3 = df[df['Zone'] == 3].drop(columns=["Zone"])
        df_zone_4 = df[df['Zone'] == 4].drop(columns=["Zone"])
        df_zone_5 = df[df['Zone'] == 5].drop(columns=["Zone"])
        # EU and Fr Zone
        df_zone_2_5 = df[(df['Zone'] == 2) | (df['Zone'] == 5)].drop(columns=["Zone"])
        return {
            'global': df,
            'zone_1': df_zone_1,
            'zone_2': df_zone_2,
            'zone_3': df_zone_3,
            'zone_4': df_zone_4,
            'zone_5': df_zone_5,
            'zone_2_5': df_zone_2_5
        }


class Forecast_Data_Reader():
    """docstring for Forecast Data Reader"""

    def __init__(self, filePath="../data/forecast_data.xlsx", sheetName="forecast"):
        self.file_path = filePath
        self.sheet_name = sheetName

    def get_filePath(self):
        return self.file_path

    def set_filePath(self, filePath):
        self.file_path = filePath

    def get_sheetName(self):
        return self.sheet_name

    def set_sheetName(self, sheetName):
        self.sheet_name = sheetName

    def read_file(self):
        """ read file and remove uncessery data """
        df = pd.read_excel(self.file_path, engine="openpyxl", sheet_name=self.sheet_name)
        df = df.set_index('SKU')
        return df
