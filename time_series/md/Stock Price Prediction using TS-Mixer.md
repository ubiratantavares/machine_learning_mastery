[Linkj](https://medium.com/@harshsh15229/stock-price-prediction-using-ts-mixer-1dfc0a11c1b7)

I recently came across the powerful Time Mixer model, known for delivering impressive results on complex [datasets](https://www.kaggle.com/datasets/vijayvvenkitesh/microsoft-stock-time-series-analysis). Curious to put it to the test, I decided to apply it to a dataset I found on Kaggle, which contains historical stock prices for Microsoft. In this project, we’ll explore how Time Mixer can be leveraged to predict the next day’s actual closing price for Microsoft stock, demonstrating its potential for financial time series forecasting

So, before going forward let’s head to the Content:

1. What is Time Mixer?
2. EDA(Exploratory Data Analysis of the data).
3. Applying the model on the dataset.
4. Seeing the accuracy and the result.

# **What is Time Mixer?**

Time Mixer represents a significant step forward in time series analysis, especially when dealing with complex variations in the target variable over time. Traditional models like ARIMA can struggle to capture these intricate patterns, leading to subpar predictions. This is where Time Mixer shines.

Time Mixer belongs to a new class of models known as mixer models. These models are designed to identify dependencies and hidden patterns in sequences, much like how transformers handle text or images. Unlike Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks that process sequences step-by-step — making them slower for longer sequences — Time Mixer operates in parallel, allowing it to handle sequences more efficiently. Plus, it’s simpler and more streamlined compared to transformers, making it an exciting choice for time series analysis where traditional methods fall short

**Time-mixer Design**:

- Time-mixer specifically targets **time series data**, which involves sequences of data points indexed in time order. It is designed to model temporal dependencies efficiently, capturing both short-term and long-term patterns.
- The model typically includes **mixing layers** that process input sequences in a way that allows the model to learn complex temporal patterns without requiring explicit attention mechanisms.

![](https://miro.medium.com/v2/resize:fit:700/1*H-euj1Z1YLsIht27k20FsQ.png)

Architecture of Time Mixer

# **Exploratory Data Analysis(EDA)**

data = pd.read_csv(“/content/Microsoft_Stock.csv”)

Let’s start with installing libraries.

pip install pandas  
pip install numpy  
pip install tensorflow  
pip install sklearn  
pip install matplotlib

Let’s dive into the data.

microsoft_data = pd.read_csv("/content/Microsoft_Stock.csv")  
microsoft_data.head()

![](https://miro.medium.com/v2/resize:fit:450/1*DPE-C4G16cUfbfiz7WuTBw.png)

So, this how are data frame looks it has 6 columns, which are Date, Open,High, Low, Close & Volume.Let’s check the datatype of all the columns along with the shape of the DataFrame.

microsoft_data.info()

![](https://miro.medium.com/v2/resize:fit:327/1*UPqhwXNNVZQhFzvhuz6mQQ.png)

As , we can see that the Date column has datatype object and the number of rows in our dataset is 1511,Let’s first convert the datatype of Date column into datetime.

microsoft_data['Date']=microsoft_data.Date.astype('datetime64[ns]')

Let’s plot a bar graph to see the variation over the year.

stock_volume = microsoft_data.groupby(microsoft_data.Date.dt.year).Volume.max().reset_index()  
stock_volume  
plt.figure(figsize=[16,5])  
plt.bar(stock_volume.Date,stock_volume.Volume)  
plt.xlabel('Date')  
plt.ylabel('Volume')  
plt.title("Volume with Year")

![](https://miro.medium.com/v2/resize:fit:700/1*qr_oD1nm9HnN3KdEC76kCg.png)

This how the graph looks we can see the dip in volume in the year 2019,it is where they have the lowest value of Volume, **we have taken each max volume in this graph.**

Let’s create more columns for the better understanding of the dataset.

microsoft_data = microsoft_data.set_index('Date')  
microsoft_data['year'] = microsoft_data.index.year  
microsoft_data['month'] = microsoft_data.index.month  
microsoft_data['day'] = microsoft_data.index.day  
microsoft_data['hour'] = microsoft_data.index.hour  
microsoft_data['dayofweek'] = microsoft_data.index.dayofweek  
microsoft_data['dayofyear'] = microsoft_data.index.dayofyear  
microsoft_data['weekofyear'] = microsoft_data.index.isocalendar().week  
microsot_data['quarter'] = microsot_data.index.quarter

This is how the value of our new added columns in our dataset looks like.

![](https://miro.medium.com/v2/resize:fit:515/1*LwrcwGSsruQwltjdeA5ENw.png)

Now, Let’s Look at the subplot of the volume over the years.

import seaborn as sns  
color_pal = sns.color_palette('tab10')  
plt.subplots(figsize=(15, 5))  
sns.lineplot(microsoft_data=microsoft_data, x='month', y='Volume', hue='year',palette=color_pal, ci=False)  
plt.show()

![](https://miro.medium.com/v2/resize:fit:700/1*p70lIj3m0TWplemFPe649A.png)

Let’s visualize how the closing price has trended over the years using a graph.

data['Close'].plot(figsize=[15,7])  
plt.xlabel("Date")  
plt.ylabel("Close")  
plt.plot()

![](https://miro.medium.com/v2/resize:fit:700/1*-qhpk9RjuAMKy8ApTKTR9g.png)

Well the price has significantly grown over the years, however it has seen a drop in the year 2020.

# TS-Mixer Model

Now, Let’s build our model.

we will start with importing the required libraries.

#importing necessary libraries  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.preprocessing import MinMaxScaler  
from torch.utils.data import DataLoader, TensorDataset  
import pandas as pd  
import numpy as np

I have created a separate file for both EDA and TS-mixer. I would suggest to have a separate file for both of them as it would help you to navigate more easily and the things are more organized in that way.

Lets start with uploading csv in a Data Frame.

data = pd.read_csv("/content/Microsoft_Stock.csv")  
# Convert the 'Date' column to datetime  
data['Date'] = pd.to_datetime(data['Date'])

Now we will normalize our columns using Min Max Scaler.

#Normalizing the data  
scaler = MinMaxScaler()  
data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

Converting the DataFrame into numpy array.

values = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

Lets provide a sequence length and create our dataset. **Sequence length** refers to the number of time steps used as input for making a prediction in time series models. In our case, it’s the number of previous days (or observations) considered when predicting the next day’s price.

def create_sequences(data, seq_length):  
    xs, ys = [], []  
    for i in range(len(data) - seq_length):  
        x = data[i:i+seq_length]  
        y = data[i+seq_length, 3]  # Close price as target  
        xs.append(x)  
        ys.append(y)  
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)  
  
# Set the sequence length to 180  
seq_length = 180  
X, y = create_sequences(values, seq_length)

Splitting the data into training and testing dataset.

  
train_size = int(X.shape[0] * 0.8)  
X_train, y_train = X[:train_size], y[:train_size]  
X_test, y_test = X[train_size:], y[train_size:]

Creating DataLoader

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)  
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

Now, it’s time to setup our model.

class TSMixer(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, output_size):  
        super(TSMixer, self).__init__()  
        self.mixer = nn.Sequential(  
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1),  
            nn.ReLU(),  
            *[nn.Sequential(  
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1),  
                nn.ReLU()) for _ in range(num_layers)],  
            nn.Conv1d(in_channels=hidden_size, out_channels=output_size, kernel_size=1)  
        )  
      
    def forward(self, x):  
        x = x.transpose(1, 2)    
        x = self.mixer(x)  
        x = x.transpose(1, 2)    
        return x[:, -1, :]  

Initializing the model, loss function and optimizer.

input_size = X_train.shape[2]  # 5 features: Open, High, Low, Close, Volume  
hidden_size = 64  
num_layers = 2  
output_size = 1  
  
model = TSMixer(input_size, hidden_size, num_layers, output_size)  
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

Now, it’s time to train our model.

epochs = 50  
for epoch in range(epochs):  
    model.train()  
    for batch_X, batch_y in train_loader:  
        optimizer.zero_grad()  
        outputs = model(batch_X)  
        loss = criterion(outputs, batch_y.unsqueeze(1))  
        loss.backward()  
        optimizer.step()  
  
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')  
  
# 4. Prediction  
model.eval()  
predictions = []  
with torch.no_grad():  
    for batch_X, _ in test_loader:  
        pred = model(batch_X)  
        predictions.append(pred)  
          
# Convert predictions to a numpy array and invert normalization  
# Create a placeholder for inverse scaling  
predictions = torch.cat(predictions).numpy()  
# Initialize an array of zeros with the same number of features as the original data  
predicted_prices_full = np.zeros((predictions.shape[0], values.shape[1]))  
  
# Place the predictions in the 'Close' column (3rd index)  
predicted_prices_full[:, 3] = predictions[:, 0]  
  
# Inverse transform  
predicted_prices_full = scaler.inverse_transform(predicted_prices_full)  
  
# Extract the 'Close' prices from the inverse-transformed data  
predicted_prices = predicted_prices_full[:, 3]  
  
# Convert predictions to a DataFrame and save to a CSV file  
predicted_prices_df = pd.DataFrame(predicted_prices, columns=['Predicted_Close'])  
predicted_prices_df.to_csv('predicted_prices.csv', index=False)  
  
print("Predictions saved to predicted_prices.csv")

Here i have taken epoch value 50, make sure to change it as per your data.

![](https://miro.medium.com/v2/resize:fit:341/1*obWNKmti4zbxHf8JIkyyng.png)

These are the epoch values along with the MSE

This how the output would look like, you can also alter the value of learning rate as per your data need.

It’s time to check the model accuracy.

y_test_np = y_test.numpy().reshape(-1, 1)  
  
# Create an array with the same shape as the original features, filled with zeros  
y_test_full = np.zeros((y_test_np.shape[0], values.shape[1]))  
  
# Place y_test_np in the Close column (assuming it's the 4th column as before)  
y_test_full[:, 3] = y_test_np[:, 0]  
  
# Apply inverse transform only on the relevant column  
actual_prices_full = scaler.inverse_transform(y_test_full)  
  
# Extract the actual Close prices  
actual_prices = actual_prices_full[:, 3]  
  
# Calculate the MSE between actual and predicted Close prices  
mse = mean_squared_error(actual_prices, predicted_prices)  
print(f"Mean Squared Error: {mse}")  
  
# Matching predictions with dates  
predicted_dates = data['Date'].iloc[train_size + seq_length:].reset_index(drop=True)  
  
# Combine dates, actual prices, and predicted prices into a DataFrame  
predicted_prices_df = pd.DataFrame({  
    'Date': predicted_dates,  
    'Actual_Close': actual_prices,  
    'Predicted_Close': predicted_prices  
})  
  
# Save the predictions with dates  
predicted_prices_df.to_csv('predicted_prices_with_dates.csv', index=False)  
  
print("Predictions with dates saved to predicted_prices_with_dates.csv")  
  
# Visualization  
plt.figure(figsize=(12, 6))  
plt.plot(predicted_prices_df['Date'], predicted_prices_df['Actual_Close'], label='Actual Close Price')  
plt.plot(predicted_prices_df['Date'], predicted_prices_df['Predicted_Close'], label='Predicted Close Price')  
plt.xlabel('Date')  
plt.ylabel('Close Price')  
plt.title('Actual vs Predicted Close Prices')  
plt.legend()  
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()

The Graph of Actual and Predicted Close Prices Looks like:

![](https://miro.medium.com/v2/resize:fit:700/1*ZYs8aV1JcajeYSS8mx4QgA.png)

The result is quite phenomenal with MSE(Mean Square Error) score as 21.56, this value can further be reduced by altering the values of input layer, though you should be careful while increasing the layer as it might led to overfitting.

So, this how our model works, you can further analyze its efficiency by comparing its results with the other models like ARIMA, Exponential Smoothing etc.

[[plataformas/Machine Learning Mastery/Tutoriais/Time Series/Time Series]]
