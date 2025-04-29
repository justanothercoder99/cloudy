import pandas as pd

from dbHandler import DBHandler
from dataStore import DataStore
from cloudStorageHandler import StorageHandler


class Helper:
    def __init__(self) -> None:
        self.dbHandler = DBHandler()
        self.dbHandler.setupEngine()
        self.hoarder = None
        self.storager = None


    # Function to load CSV data into SQL database
    def cleanAndUpload(self, filepath, table_name):
        if self.hoarder is None:
            self.hoarder = DataStore()

        df = pd.read_csv(filepath)
        if table_name == "households":
            self.hoarder.cleanHouseholds(df)
        elif table_name == "products":
            self.hoarder.cleanProducts(df)
        elif table_name == "transactions":
            self.hoarder.cleanTransactions(df)
        
        df.to_csv(filepath)
        self.uploadToCloud(df, table_name)

        if self.hoarder.households is not None and self.hoarder.products is not None and self.hoarder.transactions is not None:
            m1 = self.hoarder.merge(self.hoarder.transactions, self.hoarder.products, on="product_num")
            merged_retail_data = self.hoarder.merge(m1, self.hoarder.households, on="hshd_num")
            merged_retail_data.to_csv("merged_retail_data")
            self.hoarder.merged_retail_data = merged_retail_data
            self.uploadToCloud(merged_retail_data, "merged_retail_data")

    
    def uploadToCloud(self, df, table_name):
        if self.storager is None:
            self.storager = StorageHandler()
        
        self.storager.uploadToCloud(df, table_name)


    def getRetailDataforHH(self, hshd_num):
        results=[]
        query = f'''
SELECT
hshd_num,
basket_num,
purchase_date,
product_num,
department, 
commodity,
loyalty_flag,
age_range,
marital_status,
income_range,
homeowner,
hshd_composition,
hh_size,
children,
spend,
units,
store_region,
week_num,
year,
brand_type,
natural_organic_flag
FROM
merged_retail_data
WHERE
hshd_num = {hshd_num}
ORDER BY
hshd_num,
basket_num,
purchase_date,
product_num,
department,
commodity;
'''

        if self.hoarder is not None and self.hoarder.merged_retail_data is not None:
            df = self.hoarder.merged_retail_data[self.hoarder.merged_retail_data["hshd_num"] == hshd_num]
            results = df.sort_values(by=["hshd_num","basket_num","purchase_date","product_num","department","commodity"])
        else:
            results = self.dbHandler.query(query, method="findall")
        return results