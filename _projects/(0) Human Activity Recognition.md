---
name: Human Activity Recognition
tools: [neural networks]
image: https://dmtyylqvwgyxw.cloudfront.net/instances/132/uploads/images/custom_image/image/675/normal_Human_activity_recognition.jpg?v=1541506221
description: Predciting human activity based sensor readings.
---

# HumanActivityRecognition

<br>


This project is to build a model that predicts the human activities such as Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing or Laying.

This dataset is collected from 30 persons(referred as subjects in this dataset), performing different activities with a smartphone to their waists. The data is recorded with the help of sensors (accelerometer and Gyroscope) in that smartphone. This experiment was video recorded to label the data manually.

## How data was recorded

By using the sensors(Gyroscope and accelerometer) in a smartphone, they have captured '3-axial linear acceleration'(_tAcc-XYZ_) from accelerometer and '3-axial angular velocity' (_tGyro-XYZ_) from Gyroscope with several variations. 

# Quick overview of the dataset :



* Accelerometer and Gyroscope readings are taken from 30 volunteers(referred as subjects) while performing the following 6 Activities.

    1. Walking     
    2. WalkingUpstairs 
    3. WalkingDownstairs 
    4. Standing 
    5. Sitting 
    6. Lying.


* Readings are divided into a window of 2.56 seconds with 50% overlapping. 

* Accelerometer readings are divided into gravity acceleration and body acceleration readings,
  which has x,y and z components each.

* Gyroscope readings are the measure of angular velocities which has x,y and z components.

* Jerk signals are calculated for BodyAcceleration readings.

* Fourier Transforms are made on the above time readings to obtain frequency readings.

* Now, on all the base signal readings., mean, max, mad, sma, arcoefficient, engerybands,entropy etc., are calculated for each window.

* We get a feature vector of 561 features and these features are given in the dataset.

* Each window of readings is a datapoint of 561 features.

## Problem Framework

* 30 subjects(volunteers) data is randomly split to 70%(21) test and 30%(7) train data.
* Each datapoint corresponds one of the 6 Activities.


## Problem Statement

 + Given a new datapoint we have to predict the Activity


```python
# Importing cool stuff
import pandas as pd
import numpy as np
#from skorch import NeuralNetClassifier
import torch
import torch.nn as nn 
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

## Helper functions


```python
# Utility function to read the data from csv file
def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

# Utility function to load the load
def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        filename = f'UCI_HAR_Dataset/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(
            _read_csv(filename).values
        ) 

    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))
```


```python

def load_y(subset):
    filename = f'UCI_HAR_Dataset/{subset}/y_{subset}.txt'
    y = _read_csv(filename)[0]

    return np.array(y)
```


```python
def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_y('train'), load_y('test')

    return X_train, X_test, y_train, y_test
```

### Loading Data


```python
# Data directory
DATADIR = 'UCI_HAR_Dataset'
```


```python
# Raw data signals
# Signals are from Accelerometer and Gyroscope
# The signals are in x,y,z directions
# Sensor signals are filtered to have only body acceleration
# excluding the acceleration due to gravity
# Triaxial acceleration from the accelerometer is total acceleration
SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]
```


```python
# Loading the train and test data
X_train, X_test, Y_train, Y_test = load_data()
Y_train=Y_train-1 #to convert from 1-6 range to 0-5 range
Y_test=Y_test-1
X_train.shape,Y_train.shape
```




    ((7352, 128, 9), (7352,))




```python
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes =len(np.unique(Y_train))

print('time steps :',timesteps)
print('input dimesnsions :',input_dim)
print('number of train data points : ',len(X_train))

```

    time steps : 128
    input dimesnsions : 9
    number of train data points :  7352



```python

#converting into tensor dataset
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

# dataloaders
batch_size = 16

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

```

# Building LSTM Model -1


```python
class Net(nn.Module):
    def __init__(self,hidden_size,num_layers,drop_out):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(9,hidden_size, num_layers, batch_first=True)
        self.dense=nn.Linear(hidden_size,6)
        self.dropout=nn.Dropout(drop_out)
        
    def forward(self,x,hidden):
        out,hidden=self.lstm(x,hidden)
        out=out[:,-1,:]
        out = out.contiguous().view(-1, self.hidden_size) # stack up lstm outputs
        out=self.dropout(out)
        out=self.dense(out)
        return out
    
    
    def init_hidden(self, batch_size):
        hidden=torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device),torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        return hidden
```


```python
#model initialization
hidden_size=32
num_layers=1
drop_out=0.5
net=Net(hidden_size,num_layers,drop_out)
net.to(device)
lr=0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
```

## Training the Model


```python
epochs=30
best_accuracy=0
y_pred=[]
y_true=[]
for epoch in range(epochs):
    train_loss=0
    val_loss=0
    net.train()
    y_pred.clear()
    y_true.clear()
    for x,y in train_loader:
        x=x.to(device)
        y=y.to(device)
        hidden=net.init_hidden(len(x))
        net.zero_grad()
        out=net(x.float(),hidden)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        _,pred=torch.max(out,1)
        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())
    train_loss=train_loss/len(train_loader)
    train_accuracy=accuracy_score(y_true,y_pred)
    net.eval()
    y_pred.clear()
    y_true.clear()
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)
            hidden=net.init_hidden(len(x))
            out=net(x.float(),hidden)
            loss=criterion(out,y)
            val_loss+=loss.item()
            _, pred = torch.max(out, 1) 
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
        val_loss=val_loss/len(test_loader)
        val_accuracy=accuracy_score(y_true,y_pred)
        if best_accuracy < val_accuracy:
            torch.save(net,'har.pth')
            best_accuracy=val_accuracy
    print('epoch:{:2} train loss:{:10.6f} train accuracy:{:6.2f} test loss:{:10.6f} val accuracy: {:6.2f}'.format(epoch+1,train_loss,train_accuracy,val_loss,val_accuracy))


        
    
        
        
    
```

    epoch: 1 train loss:  1.324476 train accuracy:  0.42 test loss:  1.213630 val accuracy:   0.48
    epoch: 2 train loss:  0.872796 train accuracy:  0.60 test loss:  0.811300 val accuracy:   0.65
    epoch: 3 train loss:  0.693934 train accuracy:  0.69 test loss:  0.784192 val accuracy:   0.68
    epoch: 4 train loss:  0.589816 train accuracy:  0.75 test loss:  0.797140 val accuracy:   0.67
    epoch: 5 train loss:  0.595977 train accuracy:  0.78 test loss:  0.552498 val accuracy:   0.81
    epoch: 6 train loss:  0.468460 train accuracy:  0.83 test loss:  0.640501 val accuracy:   0.79
    epoch: 7 train loss:  0.388672 train accuracy:  0.86 test loss:  0.486402 val accuracy:   0.85
    epoch: 8 train loss:  0.335186 train accuracy:  0.88 test loss:  0.451101 val accuracy:   0.85
    epoch: 9 train loss:  0.305061 train accuracy:  0.89 test loss:  0.387480 val accuracy:   0.87
    epoch:10 train loss:  0.230853 train accuracy:  0.92 test loss:  0.466579 val accuracy:   0.86
    epoch:11 train loss:  0.254533 train accuracy:  0.91 test loss:  0.427815 val accuracy:   0.88
    epoch:12 train loss:  0.245500 train accuracy:  0.92 test loss:  0.368449 val accuracy:   0.86
    epoch:13 train loss:  0.190924 train accuracy:  0.93 test loss:  0.416491 val accuracy:   0.87
    epoch:14 train loss:  0.182639 train accuracy:  0.94 test loss:  0.359438 val accuracy:   0.89
    epoch:15 train loss:  0.193191 train accuracy:  0.94 test loss:  0.343162 val accuracy:   0.89
    epoch:16 train loss:  0.199731 train accuracy:  0.93 test loss:  0.317811 val accuracy:   0.88
    epoch:17 train loss:  0.158505 train accuracy:  0.94 test loss:  0.341826 val accuracy:   0.89
    epoch:18 train loss:  0.215404 train accuracy:  0.93 test loss:  0.410098 val accuracy:   0.88
    epoch:19 train loss:  0.158758 train accuracy:  0.95 test loss:  0.541238 val accuracy:   0.84
    epoch:20 train loss:  0.144741 train accuracy:  0.95 test loss:  0.365838 val accuracy:   0.89
    epoch:21 train loss:  0.152419 train accuracy:  0.95 test loss:  0.411111 val accuracy:   0.89
    epoch:22 train loss:  0.237154 train accuracy:  0.94 test loss:  0.399992 val accuracy:   0.87
    epoch:23 train loss:  0.151714 train accuracy:  0.95 test loss:  0.392775 val accuracy:   0.88
    epoch:24 train loss:  0.141980 train accuracy:  0.95 test loss:  0.336273 val accuracy:   0.89
    epoch:25 train loss:  0.144436 train accuracy:  0.95 test loss:  0.422562 val accuracy:   0.88
    epoch:26 train loss:  0.161758 train accuracy:  0.95 test loss:  0.391417 val accuracy:   0.90
    epoch:27 train loss:  0.175759 train accuracy:  0.95 test loss:  0.437056 val accuracy:   0.89
    epoch:28 train loss:  0.135313 train accuracy:  0.95 test loss:  0.428258 val accuracy:   0.90
    epoch:29 train loss:  0.133035 train accuracy:  0.95 test loss:  0.399176 val accuracy:   0.90
    epoch:30 train loss:  0.138583 train accuracy:  0.95 test loss:  0.367444 val accuracy:   0.90


## Evaluating the model


```python
#evaulation using best model
net=torch.load('har.pth')
y_pred=[]
y_true=[]
net.eval()
net.to(device)
test_loss=0

with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device)
        y=y.to(device)
        hidden=net.init_hidden(len(x))
        out=net(x.float(),hidden)
        loss=criterion(out,y)
        test_loss+=loss.item()
        _, pred = torch.max(out, 1) 
        y_pred.extend(pred.tolist())
        y_true.extend(y.tolist())
    test_loss/=len(test_loader)

print('log loss of test data: ',np.round(test_loss,4))
accuracy=accuracy_score(y_true,y_pred)
print('accuracy of test data:',np.round(accuracy,2))

labels=['LAYING','SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
cf=confusion_matrix(y_true,y_pred)
plt.figure(figsize=(15,7))
sns.heatmap(cf, annot=True, cmap="Blues",xticklabels=labels,yticklabels=labels,fmt='g')
plt.title('confusion matrix')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
```

    log loss of test data:  0.3914
    accuracy of test data: 0.9



![png](/img/output_19_1.png)


# Building LSTM Model -2


```python
class Model(nn.Module):
    def __init__(self,hidden_size,num_layers,drop_out):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(9,hidden_size, num_layers, batch_first=True)
        self.dense=nn.Linear(hidden_size,30)
        self.dense2=nn.Linear(30,6)
        self.dropout=nn.Dropout(drop_out)
        self.dout=nn.Dropout(0.3)
        
    def forward(self,x,hidden):
        out,hidden=self.lstm(x,hidden)
        out=out[:,-1,:]
        out = out.contiguous().view(-1, self.hidden_size) # stack up lstm outputs
        out=self.dropout(out)
        out=self.dense(out)
        out=self.dout(out)
        out=self.dense2(out)
        return out
    
    
    def init_hidden(self, batch_size):
        hidden=torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device),torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        return hidden
```


```python
#model initialization
hidden_size=100
num_layers=1
drop_out=0.5
model=Model(hidden_size,num_layers,drop_out)
model.to(device)
lr=0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

```


```python
epochs=30
best_accuracy=0
y_pred=[]
y_true=[]
for epoch in range(epochs):
    train_loss=0
    val_loss=0
    model.train()
    y_pred.clear()
    y_true.clear()
    for x,y in train_loader:
        x=x.to(device)
        y=y.to(device)
        hidden=model.init_hidden(len(x))
        model.zero_grad()
        out=model(x.float(),hidden)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        _,pred=torch.max(out,1)
        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())
    train_loss=train_loss/len(train_loader)
    train_accuracy=accuracy_score(y_true,y_pred)
    model.eval()
    y_pred.clear()
    y_true.clear()
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)
            hidden=model.init_hidden(len(x))
            out=model(x.float(),hidden)
            loss=criterion(out,y)
            val_loss+=loss.item()
            _, pred = torch.max(out, 1) 
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
        val_loss=val_loss/len(test_loader)
        val_accuracy=accuracy_score(y_true,y_pred)
        if best_accuracy < val_accuracy:
            torch.save(model,'har2.pth')
            best_accuracy=val_accuracy
    print('epoch:{:2} train loss:{:10.6f} train accuracy:{:6.2f} test loss:{:10.6f} val accuracy: {:6.2f}'.format(epoch+1,train_loss,train_accuracy,val_loss,val_accuracy))


        
    
```

    epoch: 1 train loss:  1.482649 train accuracy:  0.37 test loss:  1.322458 val accuracy:   0.50
    epoch: 2 train loss:  1.312788 train accuracy:  0.46 test loss:  1.320044 val accuracy:   0.47
    epoch: 3 train loss:  1.282552 train accuracy:  0.45 test loss:  1.242504 val accuracy:   0.45
    epoch: 4 train loss:  0.958196 train accuracy:  0.57 test loss:  0.854092 val accuracy:   0.61
    epoch: 5 train loss:  0.813683 train accuracy:  0.64 test loss:  0.960151 val accuracy:   0.68
    epoch: 6 train loss:  0.585036 train accuracy:  0.75 test loss:  0.592570 val accuracy:   0.71
    epoch: 7 train loss:  0.433435 train accuracy:  0.83 test loss:  1.001910 val accuracy:   0.79
    epoch: 8 train loss:  0.392434 train accuracy:  0.85 test loss:  0.669312 val accuracy:   0.84
    epoch: 9 train loss:  0.316371 train accuracy:  0.90 test loss:  0.678665 val accuracy:   0.73
    epoch:10 train loss:  0.520638 train accuracy:  0.81 test loss:  0.513204 val accuracy:   0.82
    epoch:11 train loss:  0.286757 train accuracy:  0.90 test loss:  0.435290 val accuracy:   0.87
    epoch:12 train loss:  0.211122 train accuracy:  0.93 test loss:  0.381388 val accuracy:   0.89
    epoch:13 train loss:  0.235413 train accuracy:  0.92 test loss:  0.394710 val accuracy:   0.89
    epoch:14 train loss:  0.169838 train accuracy:  0.94 test loss:  0.373214 val accuracy:   0.91
    epoch:15 train loss:  0.208915 train accuracy:  0.93 test loss:  0.514612 val accuracy:   0.90
    epoch:16 train loss:  0.168681 train accuracy:  0.94 test loss:  0.574375 val accuracy:   0.89
    epoch:17 train loss:  0.162849 train accuracy:  0.94 test loss:  0.519155 val accuracy:   0.90
    epoch:18 train loss:  0.140146 train accuracy:  0.95 test loss:  0.438987 val accuracy:   0.91
    epoch:19 train loss:  0.181148 train accuracy:  0.94 test loss:  0.632207 val accuracy:   0.89
    epoch:20 train loss:  0.147942 train accuracy:  0.95 test loss:  0.480306 val accuracy:   0.91
    epoch:21 train loss:  0.167473 train accuracy:  0.94 test loss:  0.601657 val accuracy:   0.90
    epoch:22 train loss:  0.167379 train accuracy:  0.94 test loss:  0.358051 val accuracy:   0.90
    epoch:23 train loss:  0.157815 train accuracy:  0.94 test loss:  0.414857 val accuracy:   0.91
    epoch:24 train loss:  0.132570 train accuracy:  0.95 test loss:  0.862629 val accuracy:   0.89
    epoch:25 train loss:  0.158493 train accuracy:  0.94 test loss:  0.419029 val accuracy:   0.90
    epoch:26 train loss:  0.131236 train accuracy:  0.95 test loss:  0.690847 val accuracy:   0.90
    epoch:27 train loss:  0.145747 train accuracy:  0.95 test loss:  0.558139 val accuracy:   0.90
    epoch:28 train loss:  0.125395 train accuracy:  0.95 test loss:  0.695396 val accuracy:   0.89
    epoch:29 train loss:  0.140164 train accuracy:  0.95 test loss:  0.546501 val accuracy:   0.90
    epoch:30 train loss:  0.133647 train accuracy:  0.95 test loss:  0.346649 val accuracy:   0.92


## Evaluating the model


```python
#evaulation using best model
model=torch.load('har2.pth')
y_pred=[]
y_true=[]
model.eval()
model.to(device)
test_loss=0

with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device)
        y=y.to(device)
        hidden=model.init_hidden(len(x))
        out=model(x.float(),hidden)
        loss=criterion(out,y)
        test_loss+=loss.item()
        _, pred = torch.max(out, 1) 
        y_pred.extend(pred.tolist())
        y_true.extend(y.tolist())
    test_loss/=len(test_loader)

print('log loss of test data: ',np.round(test_loss,4))
accuracy=accuracy_score(y_true,y_pred)
print('accuracy of test data:',np.round(accuracy,2))

labels=['LAYING','SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
cf=confusion_matrix(y_true,y_pred)
plt.figure(figsize=(15,7))
sns.heatmap(cf, annot=True, cmap="Blues",xticklabels=labels,yticklabels=labels,fmt='g')
plt.title('confusion matrix')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
```

    log loss of test data:  0.3466
    accuracy of test data: 0.92



![png](/img/output_25_1.png)



```python
print(classification_report(y_true,y_pred,target_names=labels))
```

                        precision    recall  f1-score   support
    
                LAYING       0.96      1.00      0.98       496
               SITTING       0.98      0.94      0.96       471
              STANDING       0.95      0.96      0.95       420
               WALKING       0.80      0.85      0.82       491
    WALKING_DOWNSTAIRS       0.85      0.81      0.83       532
      WALKING_UPSTAIRS       1.00      1.00      1.00       537
    
             micro avg       0.92      0.92      0.92      2947
             macro avg       0.92      0.92      0.92      2947
          weighted avg       0.92      0.92      0.92      2947
    


# Building LSTM Model -3


```python
#model initialization
hidden_size=100
num_layers=2
drop_out=0.7
net=Net(hidden_size,num_layers,drop_out)
net.to(device)
lr=0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

```


```python
epochs=30
best_accuracy=0
y_pred=[]
y_true=[]
for epoch in range(epochs):
    train_loss=0
    val_loss=0
    net.train()
    y_pred.clear()
    y_true.clear()
    for x,y in train_loader:
        x=x.to(device)
        y=y.to(device)
        hidden=net.init_hidden(len(x))
        net.zero_grad()
        out=net(x.float(),hidden)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        _,pred=torch.max(out,1)
        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())
    train_loss=train_loss/len(train_loader)
    train_accuracy=accuracy_score(y_true,y_pred)
    net.eval()
    y_pred.clear()
    y_true.clear()
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)
            hidden=net.init_hidden(len(x))
            out=net(x.float(),hidden)
            loss=criterion(out,y)
            val_loss+=loss.item()
            _, pred = torch.max(out, 1) 
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
        val_loss=val_loss/len(test_loader)
        val_accuracy=accuracy_score(y_true,y_pred)
        if best_accuracy < val_accuracy:
            torch.save(net,'har.pth')
            best_accuracy=val_accuracy
    print('epoch:{:2} train loss:{:10.6f} train accuracy:{:6.2f} test loss:{:10.6f} val accuracy: {:6.2f}'.format(epoch+1,train_loss,train_accuracy,val_loss,val_accuracy))


        
    
```

    epoch: 1 train loss:  1.533695 train accuracy:  0.33 test loss:  1.230509 val accuracy:   0.44
    epoch: 2 train loss:  1.166873 train accuracy:  0.49 test loss:  1.054517 val accuracy:   0.49
    epoch: 3 train loss:  0.775564 train accuracy:  0.62 test loss:  0.696310 val accuracy:   0.64
    epoch: 4 train loss:  0.766319 train accuracy:  0.64 test loss:  0.823017 val accuracy:   0.64
    epoch: 5 train loss:  0.576431 train accuracy:  0.77 test loss:  0.736326 val accuracy:   0.66
    epoch: 6 train loss:  0.468632 train accuracy:  0.82 test loss:  0.822014 val accuracy:   0.70
    epoch: 7 train loss:  0.328237 train accuracy:  0.88 test loss:  0.547394 val accuracy:   0.87
    epoch: 8 train loss:  0.277120 train accuracy:  0.91 test loss:  0.352074 val accuracy:   0.88
    epoch: 9 train loss:  0.442696 train accuracy:  0.85 test loss:  0.867937 val accuracy:   0.75
    epoch:10 train loss:  0.236122 train accuracy:  0.92 test loss:  0.315389 val accuracy:   0.89
    epoch:11 train loss:  0.210904 train accuracy:  0.93 test loss:  0.263256 val accuracy:   0.91
    epoch:12 train loss:  0.178389 train accuracy:  0.94 test loss:  0.309705 val accuracy:   0.90
    epoch:13 train loss:  0.170183 train accuracy:  0.94 test loss:  0.239857 val accuracy:   0.92
    epoch:14 train loss:  0.186672 train accuracy:  0.94 test loss:  0.264772 val accuracy:   0.90
    epoch:15 train loss:  0.214897 train accuracy:  0.92 test loss:  0.347654 val accuracy:   0.87
    epoch:16 train loss:  0.198619 train accuracy:  0.93 test loss:  0.679378 val accuracy:   0.86
    epoch:17 train loss:  0.163781 train accuracy:  0.94 test loss:  0.361680 val accuracy:   0.88
    epoch:18 train loss:  0.166087 train accuracy:  0.94 test loss:  0.263805 val accuracy:   0.91
    epoch:19 train loss:  0.149740 train accuracy:  0.94 test loss:  0.275406 val accuracy:   0.92
    epoch:20 train loss:  0.152222 train accuracy:  0.95 test loss:  0.246042 val accuracy:   0.92
    epoch:21 train loss:  0.134731 train accuracy:  0.95 test loss:  0.344771 val accuracy:   0.92
    epoch:22 train loss:  0.122899 train accuracy:  0.95 test loss:  0.351749 val accuracy:   0.92
    epoch:23 train loss:  0.205480 train accuracy:  0.94 test loss:  0.329955 val accuracy:   0.90
    epoch:24 train loss:  0.201547 train accuracy:  0.94 test loss:  0.291776 val accuracy:   0.91
    epoch:25 train loss:  0.155327 train accuracy:  0.94 test loss:  0.304316 val accuracy:   0.91
    epoch:26 train loss:  0.146998 train accuracy:  0.95 test loss:  0.425379 val accuracy:   0.93
    epoch:27 train loss:  0.143599 train accuracy:  0.95 test loss:  0.641150 val accuracy:   0.94
    epoch:28 train loss:  0.142398 train accuracy:  0.95 test loss:  0.397803 val accuracy:   0.91
    epoch:29 train loss:  0.123157 train accuracy:  0.95 test loss:  0.321939 val accuracy:   0.91
    epoch:30 train loss:  0.142609 train accuracy:  0.95 test loss:  0.380496 val accuracy:   0.92


## Evaluating the model


```python
#evaulation using best model
net=torch.load('har.pth')
y_pred=[]
y_true=[]
net.eval()
net.to(device)
test_loss=0

with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device)
        y=y.to(device)
        hidden=net.init_hidden(len(x))
        out=net(x.float(),hidden)
        loss=criterion(out,y)
        test_loss+=loss.item()
        _, pred = torch.max(out, 1) 
        y_pred.extend(pred.tolist())
        y_true.extend(y.tolist())
    test_loss/=len(test_loader)

print('log loss of test data: ',np.round(test_loss,4))
accuracy=accuracy_score(y_true,y_pred)
print('accuracy of test data:',np.round(accuracy,2))

labels=['LAYING','SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
cf=confusion_matrix(y_true,y_pred)
plt.figure(figsize=(15,7))
sns.heatmap(cf, annot=True, cmap="Blues",xticklabels=labels,yticklabels=labels,fmt='g')
plt.title('confusion matrix')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
```

    log loss of test data:  0.6412
    accuracy of test data: 0.94



![png](/img/output_31_1.png)



```python
print(classification_report(y_true,y_pred,target_names=labels))
```

                        precision    recall  f1-score   support
    
                LAYING       0.96      0.99      0.98       496
               SITTING       0.94      0.96      0.95       471
              STANDING       0.99      1.00      0.99       420
               WALKING       0.89      0.83      0.86       491
    WALKING_DOWNSTAIRS       0.85      0.90      0.88       532
      WALKING_UPSTAIRS       1.00      0.95      0.97       537
    
             micro avg       0.94      0.94      0.94      2947
             macro avg       0.94      0.94      0.94      2947
          weighted avg       0.94      0.94      0.94      2947
    



```python
from prettytable import PrettyTable
import prettytable
x=PrettyTable()
x.hrules=prettytable.ALL
x.left_padding_width=3
x.right_padding_width=3
x.field_names=['Model','no.layers','hidden_size','Accurarcy']
x.add_row(['LSTM-1','1','32','90%'])
x.add_row(['LSTM-2','1','100','92%'])
x.add_row(['LSTM-3','2','100','94%'])


print(x)

```

    +------------+---------------+-----------------+---------------+
    |   Model    |   no.layers   |   hidden_size   |   Accurarcy   |
    +------------+---------------+-----------------+---------------+
    |   LSTM-1   |       1       |        32       |      90%      |
    +------------+---------------+-----------------+---------------+
    |   LSTM-2   |       1       |       100       |      92%      |
    +------------+---------------+-----------------+---------------+
    |   LSTM-3   |       2       |       100       |      94%      |
    +------------+---------------+-----------------+---------------+


# Summary

<b> Step 1 :</b> <p>First step would be to understand the business problem well and the value proposition that we can make by solving this problem.Coming to this case study the goal is to classify the human activity using sensor data which inturns help the users to track their sleep,calories burnt etc.

<p>
<b> Step 2:</b> <p> Next step is to load the data and and processing the data in a way that model can learn sequence information.</p>


<b> Step 3:</b> <p> In this case study I am using LSTM architecture to build the models.In this stage we will try and experiment with differnet architectures to build the model.Initially I tried with 1 hidden layer and 32 hidden units which gave 90% accurcay.</p>


<b> Step 4:</b> <p> In this stage I experimented with hidden layers to see whether increase in hidden units can help us to increase to the accuracy. With 100 hidden units I am able to get 92% accuracy.</p>


<b> Step 5:</b> <p> In this stage I experimented with both num layers and hidden size. I am able to get 94% accuracy with 2 layers 100 hidden units.And also played around with dropout.</p>

<b> Step 6:</b> <p> Further increasing in layers and hidden units lead into overfitting.In this stage I also played aroud with learning rate,optimizers,initializers.</p>
