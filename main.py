
import pandas as pd
import warnings
import os
#import sys
#import nibabel as nib
import numpy as np
import torch
import torch_geometric as tg
import nilearn.connectome
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import numpy as np
import matplotlib.pyplot as plt
import random

from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
#from nilearn.datasets.neurovault import fetch_neurovault_ids
#from nilearn.input_data import NiftiMasker
from joblib import load, dump
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from nilearn.plotting import plot_connectome
from torch.utils.data import DataLoader
from gcn_windows_dataset_test import TimeWindowsDataset
from condica.main import condica
from condica.utils import _assemble, mask_contrasts, fetch_difumo

warnings.filterwarnings('ignore')

ROOT_PATH =".\\data_set"
def find_func_files():
    files=[]
    names=[]
    for root, d_names, f_names in os.walk(ROOT_PATH):
        for f_name in f_names:
            if f_name.endswith("old.nii.gz"):
                print(f"Found: {f_name}")  # اضافه کردن این خط برای بررسی فایل‌ها
                final_names = f_name
                final_root = os.path.join(root, f_name)
                files.append(final_root)
                names.append(final_names)
    return files, names

def _make_undirected(mat):
    """
    Takes an input adjacency matrix and makes it undirected (symmetric).

    Parameter
    ----------
    mat: array
        Square adjacency matrix.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    sym = (mat + mat.transpose()) / 2
    if len(np.unique(mat)) == 2:  # if graph was unweighted, return unweighted
        return np.ceil(sym)  # otherwise return average
    return sym


def _knn_graph_quantile(mat, self_loops=False, k=8, symmetric=True):
    """
    Takes an input correlation matrix and returns a k-Nearest
    Neighbour weighted undirected adjacency matrix.
    """

    if mat.shape[0] != mat.shape[1]:

        raise ValueError("Adjacency matrix must be square.")
    dim = mat.shape[0]
    print("dim: ",dim)
    if (k <= 0) or (dim <= k):
        raise ValueError("k must be in range [1,n_nodes)")
    is_directed = not (mat == mat.transpose()).all()
    if is_directed:
        raise ValueError(
            "Input adjacency matrix must be undirected (matrix symmetric)!"
        )

    # absolute correlation
    mat = np.abs(mat)
    adj = np.copy(mat)
    # get NN thresholds from quantile
    quantile_h = np.quantile(mat, (dim - k - 1) / dim, axis=0)
    mask_not_neighbours = mat < quantile_h[:, np.newaxis]
    adj[mask_not_neighbours] = 0
    if not self_loops:
        np.fill_diagonal(adj, 0)
    if symmetric:
        adj = _make_undirected(adj)
    return adj


def make_group_graph(connectomes, k=8, self_loops=False, symmetric=True):
    """
    Parameters
    ----------
    connectomes: list of array
        List of connectomes in n_roi x n_roi format, connectomes must all be the same shape.
    k: int, default=8
        Number of neighbours.
    self_loops: bool, default=False
        Wether or not to keep self loops in graph, if set to False resulting adjacency matrix
        has zero along diagonal.
    symmetric: bool, default=True
        Wether or not to return a symmetric adjacency matrix. In cases where a node is in the neighbourhood
        of another node that is not its neighbour, the connection strength between the two will be halved.

    Returns
    -------
    Torch geometric graph object of k-Nearest Neighbours graph for the group average connectome.
    """
    if connectomes[0].shape[0] != connectomes[0].shape[1]:
        raise ValueError("Connectomes must be square.")

    # Group average connectome and nndirected 8 k-NN graph
    avg_conn = np.array(connectomes).mean(axis=0)
    avg_conn = np.round(avg_conn, 6)
    avg_conn_k = _knn_graph_quantile(
        avg_conn, k=k, self_loops=self_loops, symmetric=symmetric
    )

    # Format matrix into graph for torch_geometric
    adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(avg_conn_k))
    return tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])

t,n=find_func_files()
func_files=[]
df1 = pd.read_csv('./data_set/taowu_patients.tsv', delimiter='\t') 
df2 = pd.read_csv('./data_set/data_set_patients.tsv', delimiter='\t') 


for row in df2['name']:
    print(f"row: {row}")
    for f in t:
        if row in f:
            func_files.append(f)
            print(f"path: {f}")

func_files=list(set(func_files))
print(f'func_file {len(func_files)}')
print(f' {type(func_files[0])}')
# Preprocess the .nii.gz files



masker =  NiftiMasker(
    standardize="zscore_sample",
    mask_strategy="epi",
    memory="nilearn_cache",
    memory_level=2,
    smoothing_fwhm=8,
)

features=[]
for func_file in tqdm(func_files):
   #print(func_file)
   masker.fit(func_file)
   X = masker.transform(func_file)
   if len(X) == 137:
    features.append(X)
# X = masker.transform(func_file)
# Extract the features from the preprocessed .nii.gz files
# X = masker.fit_transform(func_file)

print('X:', features)
print('X len:', len(features))
transformer = KernelPCA(n_components=30, kernel='rbf')
X_ts=[]

for feature in features:
    X_transformed = transformer.fit_transform(feature)
    X_ts.append(X_transformed)

X_ts=np.array(X_ts)
print(f"X_ts[0].shape:{X_ts[0].shape}")

# Download the Difumo atlas to reduce the data
mask = fetch_difumo(dimension=512, data_dir="./mask/").maps
components = (
    NiftiMasker(mask_img="./mask/hcp_mask.nii.gz", verbose=1)
    .fit()
    .transform(mask)
)
dump(components, "./mask/difumo_atlases/512/components.pt")
C = load("./mask/difumo_atlases/512/components.pt")
# Reduce the data using the atlas
X_ts = X_ts.dot(C.T)
# Load mixing matrix computed from rest fMRI data
A = np.load("../data/A_rest.npy")
y = df2['status'].values
Y = y["category"].values
_, Y = np.unique(Y, return_inverse=True)
clf = LinearDiscriminantAnalysis()
cv = ShuffleSplit(random_state=0, train_size=0.8, n_splits=20)
scores_noaug = []
scores_withaug = []
for train, test in cv.split(X):
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]
    X_fakes, y_fakes = condica(
        A, X_train, Y_train, len(X[train]), n_quantiles=len(X[train])
    )

X_ts=X_ts.concat(X_fakes)
y=y.concat(y_fakes)



# Estimating connectomes and save for pytorch to load
corr_measure = nilearn.connectome.ConnectivityMeasure(kind="correlation")
print("xxx")
connections=[]
features=X_ts
# for feature in features:
for feature in features:
    feature_reshaped = feature.reshape((feature.shape[1], feature.shape[0]))
    # Fit the ConnectivityMeasure object to the data
    conn = corr_measure.fit_transform([feature_reshaped])[0]
    connections.append(conn)
    print("yyy")
    print("feature_reshaped shape: ",feature_reshaped.shape)
    print("conn shape: ",conn.shape)

    n_regions_extracted = feature_reshaped.shape[-1]
    print("zzz")

    title = 'Correlation between %d regions' % n_regions_extracted

    print('Correlation matrix shape:',conn.shape)

np.array(connections).shape


# First plot the matrix
display = plotting.plot_matrix(conn, vmax=1, vmin=-1,
                    colorbar=True, title=title)
plt.show()



# # make a graph for the subject
graph = make_group_graph([conn], self_loops=False, k=8, symmetric=True)




# Create an empty list
sphere_center = []

# Loop 300 times to fill the list with rows
for _ in range(300):
  # Create an empty sub-list for the current row
  row = []
  # Loop 300 times to fill the sub-list with random integers
  for _ in range(3):
    row.append(random.randint(0, 100))
  # Append the completed row to the main list
  sphere_center.append(row)
plot_connectome(conn, sphere_center,
        display_mode='ortho', colorbar=True,  node_size=50,
        title="Default Mode Network Connectivity")
plt.show()

graph = make_group_graph(connections, self_loops=False, k=8, symmetric=True)

# generate data


# cancatenate the same type of trials
concat_bold = {}
for label in categories:
  cur_label_index = y.index[y == label].tolist()
  curr_bold_seg = X[cur_label_index]
  concat_bold[label] = curr_bold_seg


# split the data by time window size and save to file
window_length = 1
dic_labels = {name: i for i, name in enumerate(categories)}
data_dir='./dataset'
# set output paths
split_path = os.path.join(data_dir, 'dataset_split_win/')
if not os.path.exists(split_path):
  os.makedirs(split_path)
out_file = os.path.join(split_path, '{}_{:04d}.npy')
out_csv = os.path.join(split_path, 'labels.csv')

label_df = pd.DataFrame(columns=['label', 'filename'])

for label, ts_data in concat_bold.items():
  ts_duration = len(ts_data)
  ts_filename = f"{label}_seg"
  valid_label = dic_labels[label]

  # Split the timeseries
  rem = ts_duration % window_length
  # print("window_length: ",window_length)
  n_splits = int(np.floor(ts_duration / window_length))
  # print("Yo ha haha")
  ts_data = ts_data[:(ts_duration - rem), :]
  # print("ts_data",ts_data," n_split: ",n_splits)
  if n_splits>0:
    for j, split_ts in enumerate(np.split(ts_data, n_splits)):

        ts_output_file_name = out_file.format(ts_filename, j)
        # print("Xing Xing")
        split_ts = np.swapaxes(split_ts, 0, 1)
        np.save(ts_output_file_name, split_ts)
        # print("Xing Ping")

        curr_label = {'label': valid_label, 'filename': os.path.basename(ts_output_file_name)}
        label_df = label_df._append(curr_label, ignore_index=True)

label_df.to_csv(out_csv, index=False)

# print("split_path: ",split_path)

# split dataset

random_seed = 42

train_dataset = TimeWindowsDataset(
  data_dir=split_path,
  partition="train",
  random_seed=random_seed,
  pin_memory=True,
  normalize=True,
  shuffle=True)

# valid_dataset = TimeWindowsDataset(
#   data_dir=split_path,
#   partition="valid",
#   random_seed=random_seed,
#   pin_memory=True,
#   normalize=True,
#   shuffle=True)

test_dataset = TimeWindowsDataset(
  data_dir=split_path,
  partition="test",
  random_seed=random_seed,
  pin_memory=False,
  normalize=True,
  shuffle=True)

# print("train dataset: {}".format(train_dataset))
# print("valid dataset: {}".format(valid_dataset))
# print("test dataset: {}".format(test_dataset))



batch_size = 8
torch.manual_seed(random_seed)
train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# test_features, test_labels = next(iter(test_generator))
# print(f"Feature batch shape: {test_features.size()}; mean {torch.mean(test_features)}")
# print(f"Labels batch shape: {test_labels.size()}; mean {torch.mean(torch.Tensor.float(test_labels))}")




class GCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_roi, batch_size=16, n_timepoints=1, n_classes=2):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_roi = n_roi
        self.batch_size = batch_size
        print("n_timepoints: ",n_timepoints)
        self.conv1 = tg.nn.GCNConv(in_channels=n_timepoints, out_channels=batch_size)
        self.conv2 = tg.nn.GCNConv(in_channels=batch_size, out_channels=batch_size)
        self.conv3 = tg.nn.GCNConv(in_channels=batch_size, out_channels=batch_size)
        self.conv4= tg.nn.GCNConv(in_channels=batch_size, out_channels=batch_size)

        # self.conv2 = tg.nn.ChebConv(in_channels=32, out_channels=32, K=2, bias=True)
        # self.conv3 = tg.nn.ChebConv(in_channels=64, out_channels=batch_size, K=2, bias=True)

        # self.fc1 = nn.Linear(self.n_roi * batch_size, 128)
        # self.fc2 = nn.Linear(128, n_classes)
        self.fc1 = nn.Linear(self.n_roi * batch_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.35)


    def forward(self, x):

        x0=x
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x1 = x


        x = self.conv2(x1, self.edge_index, self.edge_weight)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x2 = x

        x = self.conv3(x2, self.edge_index, self.edge_weight)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x3 = x

        x = self.conv4(x3, self.edge_index, self.edge_weight)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x4 = x
        print("x2:",x2.size())
        print("x4:",x4.size())

        x_out = torch.cat([ x1, x2, x3, x4], dim=1)  # Adjust the dimension as needed
        # x=x3+x4
        batch_vector = torch.arange(x.size(0), dtype=int)
        x = torch.flatten(x, 1)
        x = tg.nn.global_mean_pool(x, batch_vector)
        x = x.view(-1, self.n_roi * self.batch_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)


        batch_vector2 = torch.arange(x0.size(0), dtype=int)
        x0 = torch.flatten(x0, 1)
        x0 = tg.nn.global_mean_pool(x0, batch_vector2)
        x0 = x0.view(-1, self.n_roi * self.batch_size)
        x0 = self.fc1(x0)
        x0 = self.dropout(x0)
        x0 = self.fc2(x0)
        x0 = self.dropout(x0)
        x0 = self.fc3(x0)

        print("x:",x.size())

        print("x0:",x0.size())

        penal = (x.size(0) / torch.norm(x - x0))
        penal = 0.09 *penal
        return x, penal

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  acc=0
  loss_=0
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    X=X.double() #raw
    pred, penalty = model(X)

    # print("shape of y",y.size())
    loss = loss_fn(pred, y)+penalty
    # loss = loss_fn(pred, y)
    #Replaces pow(2.0) with abs() for L1 regularization

    l2_lambda = 0.001
    l2_norm = sum(p.pow(2.0).sum()
                  for p in model.parameters())

    loss = loss + l2_lambda * l2_norm

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss, current = loss.item(), batch * dataloader.batch_size
    loss_+=loss
    correct= (torch.argmax(torch.softmax(pred, dim=1), dim=1) == y).type(torch.float).sum().item()
    acc+= (torch.argmax(torch.softmax(pred, dim=1), dim=1) == y).type(torch.float).sum().item()

    # TP = torch.argmax(torch.softmax(pred, dim=1), dim=1) == y
    # FP = torch.argmax(torch.softmax(pred, dim=1), dim=1) != y
    # FN = 1 - TP
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)

    correct /= X.shape[0]

    # if (batch % 10 == 0) or (current == size):
    #   acc_t=acc/size
    #   print(f"#{batch:>5};\ttrain_loss: {loss_:>0.3f};\ttrain_accuracy:{(100*acc_t):>5.1f}%\t\t[{current:>5d}/{size:>5d}]")

  acc/=size
  acc_list.append(acc)
  loss_/=size
  loss_list.append(loss_)



  print(f"train_loss: {loss_:>0.3f};\ttrain_accuracy:{(100*acc):>5.1f}%")
  return loss_,acc

def map_classId_to_className(element):

   return 'HC' if element==0 else 'CUD'


def valid_test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  loss, correct = 0, 0
  # print("size : ",size)

  with torch.inference_mode():
    for X, y in dataloader:
      # print("X_test : ",X)
      # print("y_test : ",y)
      pred = model(X)
      # print("logit : ",pred)
      pred =torch.argmax(torch.softmax(pred, dim=1), dim=1)
      # print("pred : ",pred)
      predx=list(filter(map_classId_to_className,pred.numpy()))
      predx=['HC' if i==0 else 'CUD' for i in predx]

      ypred=list(filter(map_classId_to_className,y.numpy()))
      ypred=['HC' if i==0 else 'CUD' for i in ypred]

      print("Prediction: ",predx," | Real : ",ypred)
      # print(type(pred)," | ",type(y))
      loss += loss_fn(pred.float(), y.float()).item()
      correct += (torch.round(pred) == y).type(torch.float).sum().item()

  loss /= size
  correct /= size
  acc_test_list.append(correct)
  loss_test_list.append(loss)

  return loss, correct

gcn = GCN(graph.edge_index,
     graph.edge_attr,
     n_roi=X.shape[1],
     batch_size=batch_size,
     n_timepoints=window_length,
     n_classes=len(categories))

acc_list=[]
loss_list=[]

acc_test_list=[]
loss_test_list=[]


loss_fn = torch.nn.CrossEntropyLoss()
gcn = gcn.double()

optimizer = torch.optim.Adam(gcn.parameters(),  lr=0.0001, weight_decay=0.001)

epochs = 40
epoch_count=[]
avg_acc=0
avg_loss=0

avg_test_acc=0
avg_test_loss=0

for t in range(epochs):
  epoch_count.append(t)
  print(f"Epoch {t+1}/{epochs}\n-------------------------------")
  loss,acc=train_loop(train_generator, gcn, loss_fn, optimizer)
  avg_acc+=acc
  avg_loss+=loss
  print("\n")

  loss_test, correct_test = valid_test_loop(test_generator, gcn, loss_fn)
  print(f"Test metrics:\n\t test_loss: {loss_test:>f};\t test_accuracy: {(100*correct_test):>0.1f}%")
  avg_test_acc+=correct_test
  avg_test_loss+=loss_test
  print("\n")


print(f"** average of train accuracy : {avg_acc/epochs} **")
print(f"** average of train loss : {avg_loss/epochs} **")

plt.plot(epoch_count, loss_list, 'r')
plt.plot(epoch_count, acc_list, 'b')
plt.legend(['Training Loss','Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('D:/Mexico/Code/images_results_chart_simple/my_plot22/3.png')

plt.show()

for batch, (X, y) in enumerate(test_generator):
    print(y)

loss_test, correct_test = valid_test_loop(test_generator, gcn, loss_fn)
print(f"Test metrics:\n\t test_loss: {loss_test:>f};\t test_accuracy: {(100*correct_test):>0.1f}%")
avg_test_acc+=correct_test
avg_test_loss+=loss_test
print("\n")


print(f"** average of test accuracy : {avg_test_acc/epochs} **")
print(f"** average of test loss : {avg_test_loss/epochs} **")

plt.plot(epoch_count, loss_test_list, 'r')
plt.plot(epoch_count, acc_test_list, 'y')
plt.legend(['Testing Loss','Testing Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()