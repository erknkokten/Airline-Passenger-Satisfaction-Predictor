import pandas as pd
import numpy as np
import time as time
import decision_tree

dic = {"Age": 0, "Flight Distance": 1, "Inflight wifi service": 2,
      "Departure/Arrival time convenient": 3, "Ease of Online booking": 4, "Gate location": 5, "Food and drink": 6,
      "Online boarding": 7,
      "Seat comfort": 8, "Inflight entertainment": 9, "On-board service": 10, "Leg room service": 11,
      "Baggage handling": 12, "Checkin service": 13,
      "Inflight service": 14, "Cleanliness": 15, "Departure Delay in Minutes": 16, "Arrival Delay in Minutes": 17,
      "Gender": 18, "Customer Type": 19,
      "Type of Travel": 20, "Class_Business": 21, "Class_Eco": 22, "Class_Eco Plus": 23, "satisfaction": 24}

df = pd.read_csv('train.csv')

# Got rid of nan's
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].replace({np.nan: df["Arrival Delay in Minutes"].mean()})

df["Gender"] = df["Gender"].replace({'Female': 0, 'Male': 1})
df["Customer Type"] = df["Customer Type"].replace({'Loyal Customer': 1, 'disloyal Customer': 0})
df["Type of Travel"] = df["Type of Travel"].replace({'Business travel': 1, 'Personal Travel': 0})
df["satisfaction"] = df["satisfaction"].replace(
   {'neutral or dissatisfied': 0, 'satisfied': 1})  # For -1 Dissatisfied and 1 Satisfied
# One hot encoding
df = pd.get_dummies(df, columns=["Class"])
df = df[["Age", "Flight Distance", "Inflight wifi service",
        "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding",
        "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
        "Checkin service",
        "Inflight service", "Cleanliness", "Departure Delay in Minutes", "Arrival Delay in Minutes", "Gender",
        "Customer Type",
        "Type of Travel", "Class_Business", "Class_Eco", "Class_Eco Plus", "satisfaction"]]

df.columns = np.arange(0, 25, 1)
df = df.to_numpy()

# Getting the test data
df_test = pd.read_csv('test.csv')

# Got rid of nan's
df_test["Arrival Delay in Minutes"] = df_test["Arrival Delay in Minutes"].replace(
   {np.nan: df_test["Arrival Delay in Minutes"].mean()})

df_test["Gender"] = df_test["Gender"].replace({'Female': 0, 'Male': 1})
df_test["Customer Type"] = df_test["Customer Type"].replace({'Loyal Customer': 1, 'disloyal Customer': 0})
df_test["Type of Travel"] = df_test["Type of Travel"].replace({'Business travel': 1, 'Personal Travel': 0})
df_test["satisfaction"] = df_test["satisfaction"].replace(
   {'neutral or dissatisfied': 0, 'satisfied': 1})  # For -1 Dissatisfied and 1 Satisfied
# One hot encoding
df_test = pd.get_dummies(df_test, columns=["Class"])
df_test = df_test[["Age", "Flight Distance", "Inflight wifi service",
                  "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink",
                  "Online boarding",
                  "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
                  "Checkin service",
                  "Inflight service", "Cleanliness", "Departure Delay in Minutes", "Arrival Delay in Minutes",
                  "Gender",
                  "Customer Type",
                  "Type of Travel", "Class_Business", "Class_Eco", "Class_Eco Plus", "satisfaction"]]

df_test.columns = np.arange(0, 25, 1)

df_test = df_test.to_numpy()


def bootstrap(data):
   size = data.shape[0]
   return data[np.random.choice(size, size, replace=True)]


def choose_feature(feature_count=5):
   a = []
   for k in dic.keys():
       if not (
               k == 'satisfaction' or k == "Flight Distance" or k == "Departure Delay in Minutes" or k == "Arrival Delay in Minutes"):
           a.append(k)
   return np.random.choice(a, feature_count, replace=False)  # Returns an array of selected features


def forest(data, tree_count):
   bagging = np.zeros((tree_count, df.shape[0], df.shape[1]))
   nodes = []
   for i in range(tree_count):
       print("Forest" + str(i))
       bagged = bootstrap(data)
       bagging[i, :, :] = bagged
       nodes.append(decision_tree.construction(bagged, True))

   nodes = np.array(nodes)

   return nodes, bagging


def predict(nodes, testData):
   predictions = np.zeros((testData.shape[0], len(nodes)))
   for i in range(len(nodes)):
       predictions[:, i] = decision_tree.predict_data(testData, nodes[i])

   predictions = np.sum(predictions, axis=1).reshape(predictions.shape[0], 1)
   for i in range(len(predictions)):
       predictions[i] = 1 if predictions[i] >= len(nodes)/2 else 0

   return predictions

start = time.time()
nodes, bagging = forest(df, 250)
end = time.time()
predictions = predict(nodes, df_test)

print("Calculating the Accuracy")
test = df_test[:,24].reshape(df_test.shape[0],1)
dif = (test - predictions)
mse = (dif.transpose().dot(dif)) / test.shape[0]
print("Accuracy is", str((1 - mse) * 100))

comparison = pd.concat([pd.DataFrame(df_test[:, 24]), pd.DataFrame(predictions)], axis='columns', names='Prediction')
