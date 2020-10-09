import pickle
import pandas as pd

# data = pd.read_excel('Folds5x2_pp.xlsx')
sheets = pd.ExcelFile('Folds5x2_pp.xlsx').sheet_names
data = None
for sheet in sheets:
    temp = pd.read_excel('Folds5x2_pp.xlsx', sheet_name=sheet)
    data = pd.concat([data, temp], ignore_index='True')

data_dumps = pickle.dumps(data)

with open('data_pickle', 'wb') as f:
    data = f.write(data_dumps)

with open('data_pickle', 'rb') as f:
    data = f.read()
    print(pickle.loads(data))

print(type(data))