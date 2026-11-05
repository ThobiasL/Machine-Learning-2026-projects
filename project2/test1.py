import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader


dataset = pd.read_csv(r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights_weather_dataset.csv")

# Cleaning the dataset
object_list = [["ECTRL ID", "ADEP", "ADES", "AC Registration", "ACTUAL OFF BLOCK TIME" , "ACTUAL ARRIVAL TIME", "Actual Distance Flown (nm)"],
               ["FILED OFF BLOCK TIME", "FILED ARRIVAL TIME"],
               ["AC Type", "AC Operator", "ICAO Flight Type", "STATFOR Market Segment"]]

# Dropping the unneeded columns
dataset = dataset.drop(columns=object_list[0], axis=1)

# Converting the datetime object columns into different int columns
for col in object_list[1]:
    # Converting the datetime object columns into datetime format
    dataset[col] = pd.to_datetime(dataset[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    # Creating new columns for month, day of week, hour, and minute
    dataset[col + "_month"] = dataset[col].dt.month
    dataset[col + "_dayofweek"] = dataset[col].dt.day_of_week
    dataset[col + "_hour"] = dataset[col].dt.hour
    dataset[col + "_minute"] = dataset[col].dt.minute

# Dropping the original datetime object columns
dataset = dataset.drop(columns=object_list[1], axis=1)

# Converting the string object columns into int format
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
for col in object_list[2]:
    dataset[col] = encoder.fit_transform(dataset[[col]])

dataset.info()
# Dropping the rows with NaN values
dataset = dataset.dropna()
#dataset = dataset.dropna(subset=["Requested FL", "dep_cloud_base_m"])
dataset.info()

# Splitting the dataset into train 60%, validation 20%, and test 20%
trainvalidation, test = train_test_split(dataset, test_size=0.2)
train, validation = train_test_split(trainvalidation, test_size=0.25)

# Removing the target columns from the features and creating separate target datasets
target = ["DEP_DELAY_MIN", "ARR_DELAY_MIN"]
target_index = [dataset.columns.get_loc(col) for col in target]
train_features = train.drop(train.columns[target_index], axis=1)
train_target = train[target]
validation_features = validation.drop(validation.columns[target_index], axis=1)
validation_target = validation[target]
test_features = test.drop(test.columns[target_index], axis=1)
test_target = test[target]

# Setting the device to GPU if available, otherwise CPU
if torch.cuda.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"
    print("GPU not available, using CPU instead.")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = torch.tensor.to(torch.accelerator.current_accelerator())

# Converting the features and targets into tensors
def to_tensor(data):
    return tensor(data.to_numpy(), dtype=torch.float32)

train_features = to_tensor(train_features)
train_target = to_tensor(train_target)
validation_features = to_tensor(validation_features)
validation_target = to_tensor(validation_target)
test_features = to_tensor(test_features)
test_target = to_tensor(test_target)

print(train_features.shape, train_target.shape, validation_features.shape, validation_target.shape)

class NeuralNetwork(nn.Module):
    def __init__(self, dataset, target, object_list):
        super().__init__()
        self.n_features = dataset.shape[1] - len(target) - len(object_list[0])  # Number of features excluding target columns
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork(target_index.shape[1]).to(device)
print(model)

# Checking that real data fits through the network
model.eval()
with torch.no_grad():
    print(model(target_index[:5].to(device)))