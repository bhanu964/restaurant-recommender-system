import pandas as pd
import random

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading data...")

train_orders = pd.read_csv("data/orders.csv", low_memory=False)
vendors = pd.read_csv("data/vendors.csv")

# -------------------------------
# CREATE POSITIVE SAMPLES
# -------------------------------
print("Creating positive samples...")

positive = train_orders[['customer_id', 'LOCATION_NUMBER', 'vendor_id']].copy()
positive.columns = ['customer_id', 'location_number', 'vendor_id']
positive['target'] = 1

print("Positive samples:", positive.shape)

# -------------------------------
# PREPARE FOR NEGATIVE SAMPLING
# -------------------------------
all_vendors = vendors['id'].unique()
customers = positive['customer_id'].unique()

# Group once (OPTIMIZATION 🔥)
customer_groups = positive.groupby('customer_id')

# -------------------------------
# CREATE NEGATIVE SAMPLES (OPTIMIZED)
# -------------------------------
print("Creating negative samples...")

neg_samples = []

for i, (cid, cust_data) in enumerate(customer_groups):

    if i % 1000 == 0:
        print(f"Processed {i} customers")

    locations = cust_data['location_number'].unique()
    ordered_vendors = cust_data['vendor_id'].unique()

    # vendors not ordered
    not_ordered = list(set(all_vendors) - set(ordered_vendors))

    if len(not_ordered) == 0:
        continue

    for loc in locations:
        # 🔥 sample only 3 vendors (FAST + EFFECTIVE)
        sampled_vendors = random.sample(not_ordered, min(3, len(not_ordered)))

        for v in sampled_vendors:
            neg_samples.append([cid, loc, v, 0])

# -------------------------------
# CREATE NEGATIVE DF
# -------------------------------
negative = pd.DataFrame(
    neg_samples,
    columns=['customer_id', 'location_number', 'vendor_id', 'target']
)

print("Negative samples:", negative.shape)

# -------------------------------
# COMBINE DATA
# -------------------------------
data = pd.concat([positive, negative], ignore_index=True)

print("Final dataset:", data.shape)

# -------------------------------
# SAVE
# -------------------------------
data.to_csv("data/training_data.csv", index=False)

print("✅ Dataset saved as data/training_data.csv")