import time as time
import pandas as pd
import numpy as np
from numpy import linalg as LA

df = pd.read_csv('train.csv')
print(df)
# df.columns = ["index","id", "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance", "Inflight wifi service",
#              "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
#            "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service",
#             "Inflight service", "Cleanliness","Departure Delay in Minutes", "Arrival Delay in Minutes", "satisfaction"]
#

# Got rid of nan's
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].replace({np.nan: df["Arrival Delay in Minutes"].mean()})

df["Gender"] = df["Gender"].replace({'Female': 0, 'Male': 1})
df["Customer Type"] = df["Customer Type"].replace({'Loyal Customer': 1, 'disloyal Customer': 0})
df["Type of Travel"] = df["Type of Travel"].replace({'Business travel': 1, 'Personal Travel': 0})
df["satisfaction"] = df["satisfaction"].replace({'neutral or dissatisfied': 0, 'satisfied': 1})
# One hot encoding
df = pd.get_dummies(df, columns=["Class"])

data_ort = df[["Age", "Flight Distance", "Inflight wifi service",
              "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink",
              "Online boarding",
              "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
              "Checkin service",
              "Inflight service", "Cleanliness", "Departure Delay in Minutes", "Arrival Delay in Minutes"]]

data_normal = (data_ort - data_ort.describe().loc["mean"]) / data_ort.describe().loc["std"]
data = df[["Gender", "Customer Type", "Type of Travel", "Class_Business", "Class_Eco", "Class_Eco Plus"]]
data_normal = pd.concat([data_normal, data], axis='columns')
response = df["satisfaction"]

# At this point, string categories are turned into discrete values
# Class parameter is one hot encoded

###################### NORMALIZATION TIME #############################

##save as .csv
# compression_opts = dict(method='zip',
#                        archive_name='out.csv')
# df.to_csv('out.zip', index=False,
#          compression=compression_opts)

# save as .csv
compression_opts = dict(method='zip',
                       archive_name='normalized.csv')
data_normal.to_csv('normalized.zip', index=False,
                  compression=compression_opts)
df_test = pd.read_csv('test.csv')
df_test = df_test.sample(frac=1).reset_index(drop=True)
# Got rid of nan's
df_test["Arrival Delay in Minutes"] = df_test["Arrival Delay in Minutes"].replace(
   {np.nan: df_test["Arrival Delay in Minutes"].mean()})

# Same preprocessing for test data
df_test["Gender"] = df_test["Gender"].replace({'Female': 0, 'Male': 1})
df_test["Customer Type"] = df_test["Customer Type"].replace({'Loyal Customer': 1, 'disloyal Customer': 0})
df_test["Type of Travel"] = df_test["Type of Travel"].replace({'Business travel': 1, 'Personal Travel': 0})
df_test["satisfaction"] = df_test["satisfaction"].replace({'neutral or dissatisfied': 0, 'satisfied': 1})
# One hot encoding
df_test = pd.get_dummies(df_test, columns=["Class"])

data_ort_test = df_test[["Age", "Flight Distance", "Inflight wifi service",
                        "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
                        "Food and drink", "Online boarding",
                        "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
                        "Baggage handling", "Checkin service",
                        "Inflight service", "Cleanliness", "Departure Delay in Minutes", "Arrival Delay in Minutes"]]

data_normal_test = (data_ort_test - data_ort.describe().loc["mean"]) / data_ort.describe().loc["std"]
data_test = df_test[["Gender", "Customer Type", "Type of Travel", "Class_Business", "Class_Eco", "Class_Eco Plus"]]
data_normal_test = pd.concat([data_normal_test, data_test], axis='columns')
data_test_norm = data_normal_test
response_test = df_test["satisfaction"]

# Test data is also preprocessed and normalized.
print("Normalized training data:\n", data_normal, response)
print("Normalized test data:\n", data_test_norm, response_test)


##########PCA#############
def PCA(data_normal, percentage=25, opt_k=0, eigen_vec=np.zeros(data_normal.shape)):
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
           if percentage < (pvee * 100):
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
#data_normal, optimum_k, eigenvec = PCA(data_normal = data_normal,percentage=75)
#data_test_norm, optimum_k, eigenvec = PCA(data_test_norm, 75,  optimum_k, eigenvec)

split = 100000  # Take first 100.000 as train, rest for valid.

train_data = data_normal[:split]
train_response = response[:split]
validation_data = data_normal[split:]
validation_response = response[split:]

#AC KAPA
data_normal = data_normal.to_numpy()
train_data = train_data.to_numpy()
data_test_norm = data_test_norm.to_numpy()
validation_data = validation_data.to_numpy()


response_test = response_test.to_numpy().reshape((response_test.size, 1))
validation_response = validation_response.to_numpy().reshape((validation_response.size, 1))

# K-NN

k_range = (range(1, 15, 5))
mse_vals = np.zeros(len(k_range))
inde = 0
prediction_storage = pd.DataFrame(data=np.zeros((data_test_norm.shape[0], 1), dtype='float64'),
                                 columns=["satisfaction"]).squeeze()
columns = []
start_time = time.time()

for k in k_range:  # VALIDATION -----FINDING OPTIMUM K------
   response_prediction = np.zeros((validation_data.shape[0], 1))
   degis = validation_response.size  # number of datas to be used
   print(k)
   for i in range(0, degis):

       # Lets try 3 folds for test[1], test[2]
       a = (train_data - validation_data[i])
       euclidian_norm = (((a * a).sum(axis=1)) ** (1 / 2)).reshape((a.shape[0], 1))
       euclidian_norm = pd.concat([pd.DataFrame(euclidian_norm, columns=['mean']), train_response], axis='columns')
       sorted = euclidian_norm.sort_values(by='mean', ignore_index=True)
       if sorted.loc[0:k - 1]["satisfaction"].sum() >= k / 2:
           response_prediction[i] = 1

   # Finding MSE for a k value
   dif = (validation_response[0:degis] - response_prediction[0:degis])
   mse = (dif.transpose().dot(dif)) / degis
   mse_vals[inde] = mse
   inde = inde + 1
   prediction_storage = pd.concat(
       [prediction_storage, pd.DataFrame(response_prediction, columns=["satisfaction" + str(k)]).squeeze()],
       axis='columns', names="k= " + str(k))

temp = np.argmin(mse_vals)
opt_k = 1 + (temp * 5)  # Since 1,51,101...
mse_vals = pd.DataFrame(mse_vals, columns=['MSE'])
mse_vals = mse_vals.transpose()
mse_vals.columns = ['k=1', 'k=6', 'k=11']
mse_vals = mse_vals.transpose()
print("MSE values for different k values: \n", mse_vals)
print("Optimal k value is : ", opt_k)
output = pd.concat([prediction_storage[0:degis]['satisfaction' + str(opt_k)],
                   pd.DataFrame(response_prediction, columns=['Real Response'])[0:degis]], axis='columns')
print("Result: \n", output)

end_time = time.time()  # GETTING THE TIMEEEEE

print("Time passed in seconds:" + str((end_time - start_time)))

# Testing optimum K on test set
print("Optimum K value was found to be " + str(opt_k))
response_prediction = np.zeros((data_test_norm.shape[0], 1))
degis = int(len(data_test_norm) / 13)  # number of datas to be used
for i in range(0, degis):

   # Lets try 3 folds for test[1], test[2]
   a = (train_data - data_test_norm[i])
   euclidian_norm = (((a * a).sum(axis=1)) ** (1 / 2)).reshape((a.shape[0], 1))
   euclidian_norm = pd.concat([pd.DataFrame(euclidian_norm, columns=['mean']), train_response], axis='columns')
   sorted = euclidian_norm.sort_values(by='mean', ignore_index=True)
   if sorted.loc[0:opt_k - 1]["satisfaction"].sum() >= opt_k / 2:
       response_prediction[i] = 1

# Finding MSE for a k value
dif = (response_test[0:degis] - response_prediction[0:degis])
mse = (dif.transpose().dot(dif)) / degis

final_prediction_storage = pd.DataFrame(data=response_test[0:degis], dtype='float64',
                                       columns=["satisfaction"]).squeeze()
final_prediction_storage = pd.concat(
   [final_prediction_storage, pd.DataFrame(response_prediction[0:degis], columns=["Predicted"]).squeeze()],
   axis='columns', names="Predicted")

accuracy = (1 - mse) * 100
print("Final ccuracy = %" + str(accuracy))
