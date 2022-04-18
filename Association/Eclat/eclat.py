import pandas as pd
from apyori import apriori

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

results = list(rules)

def inspect(data):
    left=[tuple(l[2][0][0])[0] for l in data]
    right=[tuple(r[2][0][1])[0] for r in data]
    # third=[tuple(t[2][0][2])[0] for t in data]
    support=[s[1] for s in data]
    return list(zip(left,right,support))

dataF=pd.DataFrame(inspect(results),columns=["Product 1","Product 2","SUPPORT"])
print(dataF.nlargest(10,'SUPPORT'))