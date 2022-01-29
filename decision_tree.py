import pandas as pd
import numpy as np
import time as time

flag = False   #True for decision tree false for random forest

dic = {"Age": 0, "Flight Distance": 1, "Inflight wifi service": 2,
      "Departure/Arrival time convenient": 3, "Ease of Online booking": 4, "Gate location": 5, "Food and drink": 6,
      "Online boarding": 7,
      "Seat comfort": 8, "Inflight entertainment": 9, "On-board service": 10, "Leg room service": 11,
      "Baggage handling": 12, "Checkin service": 13,
      "Inflight service": 14, "Cleanliness": 15, "Departure Delay in Minutes": 16, "Arrival Delay in Minutes": 17,
      "Gender": 18, "Customer Type": 19,
      "Type of Travel": 20, "Class_Business": 21, "Class_Eco": 22, "Class_Eco Plus": 23, "satisfaction": 24}


def giniIndex(data):
   sizee = data.shape[0]
   data = data[:, dic['satisfaction']].reshape((data.shape[0], 1))
   imp = 1
   prob = data.sum() / sizee
   imp -= prob * prob
   prob = (sizee - data.sum()) / sizee
   imp -= prob * prob

   return imp


class Question:
   def __init__(self, feature, value):
       self.feature = feature
       self.value = value

   def ask(self, data_instance):
       # This functions takes a data instance and asks its questions
       dataval = data_instance[self.feature]
       if type(dataval) == int or type(dataval) == float:
           return dataval >= self.value
       else:
           return dataval == self.value

   def __repr__(self):
       return "Feature: " + str(self.feature) + "\nValue: " + str(self.value)


def partition(data_instances, question):
   true_df = np.zeros(data_instances.shape)
   false_df = np.zeros(data_instances.shape)
   true_idx, false_idx = 0, 0

   for i in range(0, data_instances.shape[0]):
       if question.ask(data_instances[i, :]):
           true_df[true_idx, :] = (data_instances[i, :])
           true_idx = true_idx + 1
       else:
           false_df[false_idx, :] = (data_instances[i, :])
           false_idx = false_idx + 1

   return true_df[0:true_idx], false_df[0:false_idx]


def getBestSplit(data, feature_array=0):
   best_gain = 0
   current = giniIndex(data)
   best_q = None
   if type(feature_array) == int:
       for k in dic:

           if k == 'satisfaction' or k == "Flight Distance" or k == "Departure Delay in Minutes" or k == "Arrival Delay in Minutes":
               continue

           for element in np.unique(data[:, dic[k]]):
               q = Question(dic[k], element)
               acc, inacc = partition(data, q)
               if not (len(acc) == 0 or len(inacc) == 0):
                   p = float(acc.shape[0] / (acc.shape[0] + inacc.shape[0]))
                   gain = current - p * giniIndex(acc) - (1 - p) * giniIndex(inacc)
                   if gain >= best_gain:
                       opt_k = k
                       best_gain, best_q = gain, q
   else:
       for k in dic:
           if k in feature_array:
               for element in np.unique(data[:, dic[k]]):
                   q = Question(dic[k], element)
                   acc, inacc = partition(data, q)
                   if not (len(acc) == 0 or len(inacc) == 0):
                       p = float(acc.shape[0] / (acc.shape[0] + inacc.shape[0]))
                       gain = current - p * giniIndex(acc) - (1 - p) * giniIndex(inacc)
                       if gain >= best_gain:
                           opt_k = k
                           best_gain, best_q = gain, q

   return best_gain, best_q


class Leaf:
   def __init__(self, data):
       count = 0
       q = Question(dic['satisfaction'], 1)
       for i in range(0, data.shape[0]):
           temp = q.ask(data[i, :])
           if temp:
               count = count + 1

       ratio = count / data.shape[0]
       self.true = ratio
       self.false = 1 - ratio

       self.predictions = 1 if ratio > 0.5 else 0  # Hardcoded by 2 since binary classification


class dec_Node:
   def __init__(self, question, acc, inacc):
       self.question = question
       self.acc = acc
       self.inacc = inacc


def construction(data, feature_array=False):
   if not feature_array:
       gain, q = getBestSplit(data)
       #print(gain)
       if gain == 0:
           return Leaf(data)

       acc, inacc = partition(data, q)

       acc_branch = construction(acc)
       inacc_branch = construction(inacc)

   else:
       features = choose_feature()
       gain, q = getBestSplit(data, features)
       #print(gain)
       if gain == 0:
           return Leaf(data)

       acc, inacc = partition(data, q)

       acc_branch = construction(acc, True)
       inacc_branch = construction(inacc, True)

   return dec_Node(q, acc_branch, inacc_branch)


def choose_feature(feature_count=2):
   a = []
   for k in dic.keys():
       if not (
               k == 'satisfaction' or k == "Flight Distance" or k == "Departure Delay in Minutes" or k == "Arrival Delay in Minutes"):
           a.append(k)
   return np.random.choice(a, feature_count, replace=False)  # Returns an array of selected features


def predict_instance(row, node):
   if isinstance(node, Leaf):
       return node.predictions

   if node.question.ask(row):
       return predict_instance(row, node.acc)
   else:
       return predict_instance(row, node.inacc)


def predict_data(data, node):
   predictions = np.zeros(data.shape[0])

   for x in range(data.shape[0]):
       predictions[x] = predict_instance(data[x, :], node)

   return predictions


if flag:
   print("Decision Tree")

   # Getting the train data
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
   df = df.to_numpy()

   start = time.time()
   root = construction(df)
   end = time.time()
   predictions = predict_data(df_test, root)

   print("Accuracy Calculation")
   test = df_test[:,24]
   test = df_test[:, 24].reshape(df_test.shape[0], 1)
   predictions = predictions.reshape(predictions.shape[0], 1)
   dif = (test - predictions)
   mse = (dif.transpose().dot(dif)) / test.shape[0]
   print("Accuracy is", str((1 - mse) * 100))

   comparison = pd.concat([pd.DataFrame(df_test[:, 24]), pd.DataFrame(predictions)], axis='columns', names='Prediction')
