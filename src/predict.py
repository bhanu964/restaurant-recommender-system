import pandas as pd
import numpy as np
import joblib



rf = joblib.load("output/model.pkl")
train_cols = joblib.load("output/train_columns.pkl")

test_customers = pd.read_csv("data/test_customers.csv")
test_locations = pd.read_csv("data/test_locations.csv")
vendors = pd.read_csv("data/vendors.csv")
train_orders = pd.read_csv("data/orders.csv")

# normalize
test_customers.columns = test_customers.columns.str.lower()
test_locations.columns = test_locations.columns.str.lower()
vendors.columns = vendors.columns.str.lower()
train_orders.columns = train_orders.columns.str.lower()


all_vendors = vendors['id'].unique()

rows = []
for _, row in test_locations.iterrows():
    cid = row['customer_id']
    loc = row['location_number']

    for v in all_vendors:
        rows.append([cid, loc, v])

test_df = pd.DataFrame(rows, columns=['customer_id','location_number','vendor_id'])

# save original
test_df_original = test_df.copy()


test_df = test_df.merge(test_customers, on='customer_id', how='left')
test_df = test_df.merge(test_locations, on=['customer_id','location_number'], how='left')
test_df = test_df.merge(vendors, left_on='vendor_id', right_on='id', how='left')



test_df['key'] = (
    test_df['customer_id'] + "_" +
    test_df['location_number'].astype(str) + "_" +
    test_df['vendor_id'].astype(str)
)

vendor_popularity = train_orders['vendor_id'].value_counts()
test_df['vendor_popularity'] = test_df['vendor_id'].map(vendor_popularity)

customer_orders = train_orders.groupby('customer_id').size()
test_df['customer_order_count'] = test_df['customer_id'].map(customer_orders)

test_df['distance'] = np.sqrt(
    (test_df['latitude_x'] - test_df['latitude_y'])**2 +
    (test_df['longitude_x'] - test_df['longitude_y'])**2
)


test_df['gender'] = test_df['gender'].astype(str).str.lower().str.strip()

test_df = pd.get_dummies(
    test_df,
    columns=['location_type','vendor_category_en','gender'],
    dummy_na=True
)


test_df_model = test_df.reindex(columns=train_cols, fill_value=0)
test_df_model = test_df_model.apply(pd.to_numeric, errors='coerce').fillna(0)


preds = rf.predict_proba(test_df_model)[:,1]

test_df['target'] = preds


test_df_original['key'] = (
    test_df_original['customer_id'] + "_" +
    test_df_original['location_number'].astype(str) + "_" +
    test_df_original['vendor_id'].astype(str)
)

final_df = test_df_original.merge(
    test_df[['key','target']],
    on='key',
    how='left'
)

final_df['output'] = (
    final_df['customer_id'] + " X " +
    final_df['location_number'].astype(str) + " X " +
    final_df['vendor_id'].astype(str) + " " +
    final_df['target'].astype(str)
)

final_df[['output']].to_csv("output/submission.txt", index=False, header=False)

print("Submission created ✅")