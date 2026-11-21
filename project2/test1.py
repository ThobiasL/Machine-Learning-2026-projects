import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn

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

# removing the target columns from the features and creating separate target  variable
target = ["DEP_DELAY_MIN", "ARR_DELAY_MIN"]
x = dataset.drop(columns=target, axis=1)
y = dataset[target]

# Splitting the dataset into train 80% and test 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Setting up K for the folds in the cross-validation
cross_validation = KFold(n_splits=9, shuffle=True)

# Setting the device to GPU if available (CUDA or Intel XPU), otherwise CPU
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"
print("Device:", device)

EPOCHS = 10
BATCH_SIZE = 4096


# Converting a numpy array into a tensor
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)


class NeuralNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# Training a brand-new network on one set of training data
def train_model(x_train, y_train):
    model = NeuralNetwork(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            if len(batch) < 2:      # BatchNorm1d can't train on a single row
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch].to(device)), y_train[batch].to(device))
            loss.backward()
            optimizer.step()
    return model


# Predicting in large chunks so the GPU doesn't run out of memory
def predict(model, x_evaluate):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x_evaluate[i:i + 65536].to(device)).cpu()
                          for i in range(0, len(x_evaluate), 65536)]).numpy()


# Scaling, training and scoring, with the scalers fitted on the training part only
def fit_and_score(x_train, y_train, x_evaluate, y_evaluate):
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    model = train_model(to_tensor(x_scaler.transform(x_train)), to_tensor(y_scaler.transform(y_train)))
    pred = y_scaler.inverse_transform(predict(model, to_tensor(x_scaler.transform(x_evaluate))))
    scores = {"MAE": mean_absolute_error(y_evaluate, pred),
              "MSE": mean_squared_error(y_evaluate, pred),
              "R2": r2_score(y_evaluate, pred)}
    return model, scores


# Performing cross-validation on the training data
results = []
for fold, (train_idx, val_idx) in enumerate(cross_validation.split(x_train), start=1):
    _, scores = fit_and_score(x_train.iloc[train_idx], y_train.iloc[train_idx],
                              x_train.iloc[val_idx], y_train.iloc[val_idx])
    results.append(scores)

# Printing the results of the cross-validation
validation_result = pd.DataFrame(results).mean()
print("Cross-validation results:")
print(f"Mean Absolute Error: {validation_result['MAE']:.2f}")
print(f"Mean Squared Error: {validation_result['MSE']:.2f}")
print(f"R^2 Score: {validation_result['R2']:.2f}")
print("-" * 30)

'''
# Training on all the training data and evaluating on the test data
model, test_scores = fit_and_score(x_train, y_train, x_test, y_test)
print("Test results:")
print(f"Mean Absolute Error: {test_scores['MAE']:.2f}")
print(f"Mean Squared Error: {test_scores['MSE']:.2f}")
print(f"R^2 Score: {test_scores['R2']:.2f}")
'''
