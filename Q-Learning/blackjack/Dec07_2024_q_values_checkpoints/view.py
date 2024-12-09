import pickle

with open("q_table_ep_21000.pkl", "rb") as f:
    q_table = pickle.load(f)


print(q_table)