##
import numpy as np
import pandas as pd
from sklearn import cluster
import seaborn as sns
import scipy.sparse
import pyarrow
import matplotlib.pyplot as plt
import fastparquet
import pickle5 as pckl

plt.rcParams['figure.figsize'] = (15, 10)

##
# baskets = pd.read_parquet('data/baskets.parquet', engine='pyarrow')
# coupons = pd.read_parquet('data/coupons.parquet', engine='pyarrow')
# coupons.to_csv("coupons_csv.csv", header=True, index=False)
# prediction_index = pd.readparquet('data/coupon_index.parquet', engine='pyarrow')

##
# bigData = pd.merge(baskets, coupons, how='left', on=['week', 'shopper', 'product'])
# print(bigData.head())
##
# baskets['order_id'] = baskets.groupby(['week', 'shopper']).ngroup()
with open('data/bigData.pkl', 'rb') as f:
    data = pckl.load(f)
##
print(data.head())

##
data['discount'] = data['discount'].fillna(0)
print(sum(data['discount'].isna()))
##
# data['week'] = data['week'].astype(np.uint8)
# data['product'] = data['product'].astype(np.uint8)
# data['shopper'] = data['shopper'].astype(np.uint8)
# data['price'] = data['price'].astype(np.uint8)
# data['discount'] = data['discount'].astype(np.uint8)
##
zero_discount = data.loc[data['discount'] == 0]
##
product_price = (zero_discount.groupby(['product','price']).size().reset_index().rename(columns={0:'count'})).drop(columns = ['count'])
print(product_price)
##
def co_occurrences_sparse(x, variable_basket="basket", variable_product="product"):
    row = x[variable_basket].values
    col = x[variable_product].values
    dim = (x[variable_basket].max() + 1, x[variable_product].max() + 1)

    basket_product_table = scipy.sparse.csr_matrix(
        (np.ones(len(row), dtype=int), (row, col)), shape=dim
    )
    co_occurrences_sparse = basket_product_table.T.dot(basket_product_table).tocoo()
    co_occurrences_df = pd.DataFrame(
        {
            "product_1": co_occurrences_sparse.row,
            "product_2": co_occurrences_sparse.col,
            "co-occurrence": co_occurrences_sparse.data,
        }
    )
    return co_occurrences_df
##
co_occurrences = co_occurrences_sparse(
    x=data,
    variable_basket="order_id",
    variable_product="product",
)
co_occurrences = co_occurrences.sort_values(
    ["product_1", "product_2"]
).reset_index()
co_occurrences
##
pivot_co_occurences = co_occurrences.pivot(index="product_1", columns="product_2", values="co-occurrence")
pivot_co_occurences.fillna(0, inplace=True)
pivot_co_occurences
##
np.fill_diagonal(pivot_co_occurences.values, 0)
##