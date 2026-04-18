import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


train_orders = pd.read_csv("data/orders.csv")
train_customers = pd.read_csv("data/train_customers.csv")
train_locations = pd.read_csv("data/train_locations.csv")
vendors = pd.read_csv("data/vendors.csv")

# normalize columns
train_orders.columns = train_orders.columns.str.lower()
train_customers.columns = train_customers.columns.str.lower()
train_locations.columns = train_locations.columns.str.lower()
vendors.columns = vendors.columns.str.lower()



positive = train_orders[['customer_id','location_number','vendor_id']].copy()
positive['target'] = 1



customers = positive['customer_id'].unique()
all_vendors = train_orders['vendor_id'].unique()

import random
neg_samples = []

for cid in customers:
    cust_data = positive[positive['customer_id'] == cid]

    locations = cust_data['location_number'].unique()
    ordered_vendors = cust_data['vendor_id'].unique()

    not_ordered = list(set(all_vendors) - set(ordered_vendors))

    for loc in locations:
        sampled = random.sample(not_ordered, min(3, len(not_ordered)))
        for v in sampled:
            neg_samples.append([cid, loc, v, 0])


negative = pd.DataFrame(
    neg_samples,
    columns=['customer_id','location_number','vendor_id','target']
)

df = pd.concat([positive, negative], ignore_index=True)



df = df.merge(train_customers, on='customer_id', how='left')
df = df.merge(train_locations, on=['customer_id','location_number'], how='left')
df = df.merge(vendors, left_on='vendor_id', right_on='id', how='left')


vendor_popularity = train_orders['vendor_id'].value_counts()
df['vendor_popularity'] = df['vendor_id'].map(vendor_popularity)

customer_orders = train_orders.groupby('customer_id').size()
df['customer_order_count'] = df['customer_id'].map(customer_orders)

df['distance'] = np.sqrt(
    (df['latitude_x'] - df['latitude_y'])**2 +
    (df['longitude_x'] - df['longitude_y'])**2
)


df['gender'] = df['gender'].astype(str).str.lower().str.strip()

df = pd.get_dummies(
    df,
    columns=['location_type','vendor_category_en','gender'],
    dummy_na=True
)

y = df['target']

X = df.drop(columns=['target','customer_id'])
X = X.select_dtypes(exclude=['object'])
X = X.fillna(0)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

preds = rf.predict_proba(X_val)[:,1]

print("ROC:", roc_auc_score(y_val, preds))


joblib.dump(rf, "output/model.pkl")
joblib.dump(X.columns.tolist(), "output/train_columns.pkl")

print("Model saved ✅")