import pandas as pd
import numpy as np
import time as time
from numpy import linalg as LA
df = pd.read_csv('train.csv')
print(df)
#df.columns = ["index","id", "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance", "Inflight wifi service",
#              "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
#            "Seat comfort", "Inflight entertainment", "On-board service",   "Leg room service", "Baggage handling", "Checkin service",
#             "Inflight service", "Cleanliness","Departure Delay in Minutes", "Arrival Delay in Minutes", "satisfaction"]
#

#Got rid of nan's
df["Arrival Delay in Minutes"]=df["Arrival Delay in Minutes"].replace({np.nan: df["Arrival Delay in Minutes"].mean()})

df["Gender"] = df["Gender"].replace({'Female':0,'Male':1})
df["Customer Type"] = df["Customer Type"].replace({'Loyal Customer':1, 'disloyal Customer':0})
df["Type of Travel"] = df["Type of Travel"].replace({'Business travel':1, 'Personal Travel':0})
df["satisfaction"] = df["satisfaction"].replace({'neutral or dissatisfied':-1, 'satisfied':1})  #For -1 Dissatisfied and 1 Satisfied
#One hot encoding
df = pd.get_dummies(df, columns=["Class"])
print(df)

data_ort = df[["Age", "Flight Distance", "Inflight wifi service",
           "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
           "Seat comfort", "Inflight entertainment", "On-board service",    "Leg room service", "Baggage handling", "Checkin service",
           "Inflight service", "Cleanliness","Departure Delay in Minutes", "Arrival Delay in Minutes"]]

data_normal = (data_ort-data_ort.describe().loc["mean"])/data_ort.describe().loc["std"]
data = df[["Gender", "Customer Type", "Type of Travel", "Class_Business", "Class_Eco","Class_Eco Plus"]]
data_normal = pd.concat([data_normal, data], axis='columns')
#data_normal = data_normal[["Class_Business", "Online boarding", "Class_Eco", "Type of Travel", "Inflight entertainment", "Seat comfort", "On-board service", "Leg room service"]]
#data_normal = data_normal.head(40000)
response = df["satisfaction"]
#response = response.head(40000)

#At this point, string categories are turned into discrete values
#Class parameter is one hot encoded

###################### NORMALIZATION TIME #############################

print(data)

##save as .csv
#compression_opts = dict(method='zip',
#                        archive_name='out.csv')
#df.to_csv('out.zip', index=False,
#          compression=compression_opts)

#save as .csv
compression_opts = dict(method='zip',
                     archive_name='normalized.csv')
data_normal.to_csv('normalized.zip', index=False,
       compression=compression_opts)
df_test = pd.read_csv('test.csv')
df_test = df_test.sample(frac=1).reset_index(drop=True)
#Got rid of nan's
df_test["Arrival Delay in Minutes"]=df_test["Arrival Delay in Minutes"].replace({np.nan: df_test["Arrival Delay in Minutes"].mean()})

#Same preprocessing for test data
df_test["Gender"] = df_test["Gender"].replace({'Female':0,'Male':1})
df_test["Customer Type"] = df_test["Customer Type"].replace({'Loyal Customer':1, 'disloyal Customer':0})
df_test["Type of Travel"] = df_test["Type of Travel"].replace({'Business travel':1, 'Personal Travel':0})
df_test["satisfaction"] = df_test["satisfaction"].replace({'neutral or dissatisfied':-1, 'satisfied':1})  # For -1 dissatisfied, 1 satisfied
#One hot encoding
df_test = pd.get_dummies(df_test, columns=["Class"])

data_ort_test = df_test[["Age", "Flight Distance", "Inflight wifi service",
           "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
           "Seat comfort", "Inflight entertainment", "On-board service",    "Leg room service", "Baggage handling", "Checkin service",
           "Inflight service", "Cleanliness","Departure Delay in Minutes", "Arrival Delay in Minutes"]]

data_normal_test = (data_ort_test-data_ort.describe().loc["mean"])/data_ort.describe().loc["std"]
data_test = df_test[["Gender", "Customer Type", "Type of Travel", "Class_Business", "Class_Eco","Class_Eco Plus"]]
data_normal_test = pd.concat([data_normal_test, data_test], axis='columns')
data_test_norm = data_normal_test
#data_test_norm = data_test_norm[["Class_Business", "Online boarding", "Class_Eco", "Type of Travel", "Inflight entertainment", "Seat comfort", "On-board service", "Leg room service"]]
#data_test_norm = data_test_norm.head(4000)
response_test = df_test["satisfaction"]
#response_test = response_test.head(4000)


#Test data is also preprocessed and normalized.




##########PCA#############
def PCA(data_normal, percentage=60, opt_k=0, eigen_vec=np.zeros(data_normal.shape)):
   if opt_k == 0:
       data_normal = data_normal.to_numpy()
       # data_normal_a = data_normal- data_normal.mean(axis=0)
       cov_mat = (data_normal.transpose()).dot(data_normal) / data_normal.shape[0]

       u, w = LA.eigh(cov_mat)  # u= eigenval   v = eigvec
       sort_index = np.argsort(u)[::-1]
       u = u[sort_index]
       w = w[:, sort_index]
       indexx = 0
       pve = np.zeros(24)
       pvee = 0
       opt_k = 0

       temprary1 = np.sum(np.sum((data_normal ** 2))) / data_normal.shape[0]

       eigen = np.zeros(24)
       eigenn = 0

       new_data = np.array(np.zeros(data_normal.shape))
       for k in range(1, 25):
           z = data_normal.dot(w[:, k - 1])
           new_data[:, k - 1] = z
           temprary = z.transpose().dot(z) / data_normal.shape[0]
           pve[indexx] = temprary / temprary1  # PVE LER teker teker tutuluyor
           pvee = pvee + (temprary / temprary1)  # PVE toplamlari
           eigenn = eigenn + u[indexx] / np.sum(u)
           eigen[indexx] = u[indexx] / np.sum(u)
           indexx = indexx + 1
           if percentage <= (pvee * 100):
               opt_k = k
               break

       data_normal_pca = new_data[:, :opt_k]
   else:
       w = eigen_vec
       new_data = np.array(np.zeros(data_normal.shape))
       for k in range(1, opt_k + 1):
           z = data_normal.dot(w[:, k - 1])
           new_data[:, k - 1] = z

       data_normal_pca = new_data[:, :opt_k]

   return data_normal_pca, opt_k, w


##########PCA#############

#AC KAPA
#data_normal, optimum_k, eigenvec = PCA(data_normal = data_normal, percentage= 75)
#data_test_norm, optimum_k, eigenvec = PCA(data_test_norm, 75,  optimum_k, eigenvec)



split = 100000 #Take first 100.000 as train, rest for valid.

train_data = data_normal[:split]
train_response = response[:split]
validation_data = data_normal[split:]
validation_response = response[split:]

#AC KAPA
data_normal = data_normal.to_numpy()
train_data = train_data.to_numpy()
data_test_norm = data_test_norm.to_numpy()
validation_data = validation_data.to_numpy()


validation_response = validation_response.to_numpy().reshape((validation_response.size, 1))

response = response.to_numpy()
response_test = response_test.to_numpy()


#Number of iterations and learning rate

start = time.time()
iters = 100
learn_rate = 0.01

n, p = train_data.shape
prediction_storage = pd.DataFrame(validation_response, dtype='float64', columns=["satisfaction"]).squeeze()
ind1 = 0

w_vals = np.zeros(p)
b_val = 0
lambd = 0

msee=np.zeros(6)
min=1000                #random big number
opt_lambd = 0
indexx = 0
opt_w_vals = np.zeros(p)
opt_b_vals = 0

for lambd in np.arange(0, 0.025, 0.005):
   print("Lambda = " + str(lambd))
   w_vals = np.zeros(p)
   b_val = 0
   for i in range(iters):

       for index in range(0, len(train_data)):
           expression = train_response[index] * (np.dot(train_data[index], w_vals) - b_val)
           if expression >= 1:
               w_vals -= learn_rate * (2 * lambd * w_vals)

           else:
               w_vals -= learn_rate * (2 * w_vals * lambd - (train_response[index] * train_data[index]))
               b_val -= learn_rate * train_response[index]

   predictions = np.dot(validation_data, w_vals) - b_val  # - b_val
   predictions = (pd.DataFrame(predictions)).apply(lambda x: [1 if a > 0 else -1 for a in x])
   prediction_storage = pd.concat([prediction_storage, predictions], axis='columns', names='Iteration')
   dif = (validation_response.reshape((predictions.shape)) - predictions.to_numpy()) / 2
   mseci = ((dif.transpose()).dot(dif)) / len(validation_response)
   msee[indexx] = mseci
   if(mseci < min):
       min = mseci
       opt_lambd = lambd
       opt_w_vals = w_vals
       opt_b_vals = b_val

   indexx = indexx + 1


print("Optimum lambda value was found to be: " + str(opt_lambd) + " with error " + str(min*100))


end = time.time()
duration = end - start
print("Time passed in seconds = ", str(duration))



# Testing optimum Lambda on test set
final_prediction_storage = pd.DataFrame(response_test, dtype='float64', columns=["satisfaction"]).squeeze()
predictions = np.dot(data_test_norm, opt_w_vals) - opt_b_vals  # - b_val
predictions = (pd.DataFrame(predictions)).apply(lambda x:[1 if a > 0 else -1 for a in x])
final_prediction_storage = pd.concat([final_prediction_storage, predictions], axis='columns')
final_prediction_storage.columns = ["Real Response", "Predictions"]


dif = (response_test.reshape((predictions.shape)) - predictions.to_numpy())/2
msee = ((dif.transpose()).dot(dif))/len(response_test)

end = time.time()

duration = end - start

print("MSE = ", str(msee[0][0]), "\nAccuracy = %", str((1-msee[0][0])*100))

print("Output: \n", final_prediction_storage)
