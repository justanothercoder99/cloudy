import pandas as pd
import numpy as np
import os

class DataStore:
    def __init__(self):
        self.init = True
        self.households = None
        self.transactions = None
        self.products = None
        self.merged_retail_data = None
        
    def removeWhitespacesAndIdentifyNull(self, df):
        df.columns = df.columns.str.strip().str.lower()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
                df[col] = df[col].replace('null', np.nan)

    def cleanHouseholds(self, households):
        households.columns = households.columns.str.strip().str.lower()
        self.removeWhitespacesAndIdentifyNull(households)
        households.dropna(subset=['hshd_num'], inplace=True)
        households = households.astype({ "hshd_num": "int32" })
        households.rename(columns={'l': 'loyalty_flag'}, inplace=True)
        households.rename(columns={'marital': 'marital_status'}, inplace=True)
        self.fillUnknowns(households)
        self.households = households

    def cleanProducts(self, products):
        products.columns = products.columns.str.strip().str.lower()
        self.removeWhitespacesAndIdentifyNull(products)
        products.dropna(subset=['product_num'], inplace=True)
        products = products.astype({ "product_num": "int32" })
        products.rename(columns={'brand_ty': 'brand_type'}, inplace=True)
        self.fillUnknowns(products)
        self.products = products

    def cleanTransactions(self, transactions):
        transactions.columns = transactions.columns.str.strip().str.lower()
        self.removeWhitespacesAndIdentifyNull(transactions)
        transactions.rename(columns={'store_r': 'store_region'}, inplace=True)
        transactions.rename(columns={'purchase_': 'purchase_date'}, inplace=True)
        transactions.dropna(subset=['hshd_num', 'product_num', 'basket_num', 'purchase_date'], inplace=True)
        transactions = transactions.astype({ "hshd_num": "int32", "product_num": "int32", "basket_num": "int32" })
        transactions["purchase_date"] = pd.to_datetime(transactions["purchase_date"], format="%d-%b-%y")
        self.fillUnknowns(transactions)
        self.transactions = transactions

    def merge(self, df1, df2, how="left", on=""):
        merged = pd.merge(df1, df2, on=on, how=how)
        return merged

    def fillUnknowns(self, df):
        df.fillna('Unknown', inplace=True)