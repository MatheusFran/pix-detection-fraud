from faker import Faker
from dotenv import load_dotenv
import os
import logging as log
import pandas as pd
import numpy as np



fake = Faker('pt_BR')
np.random.seed(42)

n_customers = 1000
n_transactions = 15000

customer_id = range(1, n_customers + 1)
cpf = [fake.cpf() for _ in range(n_customers)]
age = np.random.randint(18, 80, size=n_customers)
gender = np.random.choice(['M', 'F'], size=n_customers)
pix_key = [fake.email() for _ in range(n_customers)]
account_type = np.random.choice(['corrente', 'poupança'], size=n_customers)
city = [fake.city() for _ in range(n_customers)]

customers = pd.DataFrame({
    'customer_id': customer_id,
    'cpf': cpf,
    'age': age,
    'gender': gender,
    'pix_key': pix_key,
    'account_type': account_type,
    'city': city
})

transaction_id = range(1, n_transactions + 1)
timestamp = [fake.date_time_this_year() for _ in range(n_transactions)]  # Correção: _ em vez de *
sender_id = np.random.choice(customer_id, size=n_transactions)
receiver_id = np.random.choice(customer_id, size=n_transactions)

for i in range(n_transactions):
    if sender_id[i] == receiver_id[i]:
        receiver_id[i] = np.random.choice([x for x in customer_id if x != sender_id[i]])

amount = np.round(np.random.exponential(scale=2000, size=n_transactions), 2)
amount = np.clip(amount, 1, 20000)
device_type = np.random.choice(['mobile', 'web', 'app'], size=n_transactions, p=[0.6, 0.2, 0.2])

fraud = []
for s, a, age_idx in zip(sender_id, amount, [age[i - 1] for i in sender_id]):
    if np.random.rand() < 0.05:  # 5% de chance aleatória
        fraud.append(1)
    elif a > 8000:  # Valor alto
        fraud.append(1)
    elif age_idx < 25 and a > 5000:  # Jovem com valor alto
        fraud.append(1)
    else:
        fraud.append(0)

transactions = pd.DataFrame({
    'transaction_id': transaction_id,
    'timestamp': timestamp,
    'sender_id': sender_id,
    'receiver_id': receiver_id,
    'amount': amount,
    'device_type': device_type,
    'fraud': fraud
})

customers.to_csv('../../data/raw/customers.csv', index=False)
transactions.to_csv('../../data/raw/transactions.csv', index=False)

log.info(f"Created {len(customers)} customers and {len(transactions)} transactions")

print(f"Clientes criados: {len(customers)}")
print(f"Transações criadas: {len(transactions)}")
print(f"Transações fraudulentas: {sum(fraud)} ({sum(fraud)/len(fraud)*100:.1f}%)")

