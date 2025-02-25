
[Link](https://medium.com/@marcelboersma/neuralforecasting-for-finance-predict-stock-values-d880db4ca4a1)

In the fast-evolving world of finance, computational and artificial intelligence models reshape the way we do finance. These models manage financial risks and decide in the blink of an eye whether to buy or sell stocks. Developing new models in this area requires relevant financial data and state-of-the-art models. In this tutorial, we deep-dive into the intersection of this. How do we quickly and easily prep financial data and calibrate state-of-the-art models? The complete tutorial code is available in [Colab](https://colab.research.google.com/drive/1AIxSlN8PE2EZAfG5RJukMFy-yO5xKzZb?usp=sharing).

This tutorial shows how to obtain financial data from Yahoo Finance and prepare it for use by the NeuralForecasting library. It also shows how to train models like NHITS, AutoFormer, etc., with the NeuralForecasting library.

So, buckle up and get ready to engineer your future financially!

# Financial data

Yahoo Finance provides a wealth of financial data, from stock prices to option prices to news about companies — it is all available on Yahoo Finance. Financial engineers would love to use such data to research new models. yFinance is an excellent Python package that simplifies our lives regarding financial data. Sounds like a done deal, right? Not so fast! Whereas yFinance is excellent at retrieving raw data, we also show how to prepare it for input in a neural network. But before deep-diving into that, let’s start with the basics of yFinance.

I like tech companies, so let’s retrieve the information on Apple with the ticker AAPL. First, we have a look at Yahoo Finance itself. Here, it shows the following:

![](https://miro.medium.com/v2/resize:fit:700/1*Iz19OBdp0HfCnYXzA8xEyQ.png)

All relevant information, including the latest stock price updates, financials, options, and more, is available. The yFinance Python package makes this easy to retrieve. Let’s start with some financials: the balance sheet.

import yfinance as yf  
aapl = yf.Ticker('AAPL')  
aapl.balance_sheet

![](https://miro.medium.com/v2/resize:fit:1000/1*3hZwjltnN2VT-C2Tvv9W1w.png)

Here, we retrieve information on Apple's balance sheet from the past couple of years. Let’s move on to historic stock data, which we retrieve as follows:

import yfinance as yf  
data = yf.download("AAPL", start="2015-01-01", end="2023-06-30")  
data

![](https://miro.medium.com/v2/resize:fit:1000/1*q5nRF0XVt76Kq-bNIDBWHQ.png)

Why am I writing a tutorial about this? Let me show you the problem. Imagine you want to create a train/validation/test dataset. Easy, right? Because we just split it into parts. However, a more realistic scenario is a moving window for the test dataset and an increasing knowledge base for the training dataset. I could not have explained it better than the people from [Nixtla](https://nixtlaverse.nixtla.io/) did, so I won't. Here is a figure from their website that explains the concept of chained windows:

![](https://miro.medium.com/v2/resize:fit:700/1*QheYUo1-U1G4T-8NVXeU7A.gif)

Source: [https://nixtlaverse.nixtla.io/neuralforecast/examples/getting_started_complete.html](https://nixtlaverse.nixtla.io/neuralforecast/examples/getting_started_complete.html)

On a more detailed level, we need to consider forecasting windows and the step size with which we want them to move.

![](https://miro.medium.com/v2/resize:fit:700/1*rgSiEFOcz3naoCQJPlPdhA.png)

Source: [https://nixtlaverse.nixtla.io/neuralforecast/examples/getting_started_complete.html](https://nixtlaverse.nixtla.io/neuralforecast/examples/getting_started_complete.html)

This is why just using the yFinance library isn’t enough to prepare for these forecasting models. A lot of bookkeeping needs to be done to feed the data correctly to the neural network.

# Time series data

Nixtla developed a Python package named [NeuralForecast](https://github.com/Nixtla/neuralforecast) that provides excellent support for handling time-series data and state-of-the-art time-series forecasting models. Let’s return to the data. How do we prepare the data so that the library will take care of the heavy lifting of data bookkeeping? Nixtla proposed a simple data format, which is defined as follows:

- **unique_ids**: a unique identifier for each time series (the series are stored in long format)
- **ds**: the date
- **y**: target variable to predict

Let’s get ready for our stock data. Here is some code to prepare our dataset. We download the relevant stock tick data of some tech companies like Apple, Microsoft, and Google. Then, we select the variable of interest. In our case, the Adj Close value.

To get it in a shape ready for the NeuralForecasting library, we need to melt our data into a long format. In simple terms, we concatenate the time series of each stock into one long table with a column that indicates which stock it is (Ticker column). A simple step remains: renaming the columns into the specified format.

import yfinance as yf  
import pandas as pd  
  
  
# Fetch data for multiple tickers  
tickers = ["AAPL", "GOOG", "MSFT"]  
data = yf.download(tickers, start="2015-01-01", end="2023-06-30")  
  
# Reshape the data  
df = data['Adj Close']  # No need to unstack here  
  
# Convert the Series to a DataFrame if it's not already (optional but recommended)  
if isinstance(df, pd.Series):  
    df = df.to_frame()  
  
# Melt the dataframe to long format  
hist = df.melt(ignore_index=False, var_name='Ticker', value_name='Adj Close')  
hist.reset_index(inplace=True)  
  
hist.rename(columns={'Date': 'ds', 'Ticker': 'unique_id', 'Adj Close':'y'}, inplace=True)  
hist.head()

Let’s check the result, here we see the date (ds), and the unique_id (Ticker), and y (Adj Close) value.

|    | ds                  | unique_id   |       y |  
|---:|:--------------------|:------------|--------:|  
|  0 | 2015-01-02 00:00:00 | AAPL        | 24.4022 |  
|  1 | 2015-01-05 00:00:00 | AAPL        | 23.7147 |  
|  2 | 2015-01-06 00:00:00 | AAPL        | 23.717  |  
|  3 | 2015-01-07 00:00:00 | AAPL        | 24.0495 |  
|  4 | 2015-01-08 00:00:00 | AAPL        | 24.9736 |

The dataset is now ready for use, and we can leverage the library to the full extent. So, let’s explore what that exactly means.

# A free lunch with NeuralForecast

There is no such thing as a free lunch, but it is almost free. The models build in NeuralForecast work with the standardized dataset. As a result, it is straightforward to implement state-of-the-art models. Here we go with the NHITS model:

horizon = 12  
  
# Try different hyperparmeters to improve accuracy.  
models = [NHITS(h=horizon,                      # Forecast horizon  
                input_size=2 * horizon,         # Length of input sequence  
                max_steps=1000,                 # Number of steps to train  
                n_freq_downsample=[2, 1, 1],    # Downsampling factors for each stack output  
                mlp_units = 3 * [[1024, 1024]]) # Number of units in each block.  
          ]  
nf = NeuralForecast(models=models, freq='M')  
nf.fit(df=hist, val_size=horizon)

We specify the variable models. In this case, it is the NHITS model. Note that we can specify a list of models if we want. I won’t discuss the details of the model, as we could devote an entire blog post to this. Then, we define the NeuralForecast object, feed it the list of models, and set the frequency object. For now, we set the frequency to monthly intervals. The last line starts the training on our dataset. Now that the model is trained. Let’s make some predictions. We make some in-sample predictions with

Y_hat_insample = nf.predict_insample(step_size=horizon)

and we plot these predictions using Matplotlib

import matplotlib.pyplot as plt  
  
plt.figure(figsize=(10, 5))  
  
# Iterate through each unique stock using the index  
for unique_id in Y_hat_insample.index.unique():  
    stock_data = Y_hat_insample.loc[unique_id]  # Index-based selection  
  
    # Plot true values and forecast for the current stock  
    plt.plot(stock_data['ds'], stock_data['y'], label=f'True ({unique_id})')  
    plt.plot(stock_data['ds'], stock_data['NHITS'], label=f'Forecast ({unique_id})', linestyle='--')  # Dashed line for forecast  
  
    # Mark the train-test split for this stock (if applicable)  
    # Assuming the split point is the same for all stocks  
    if len(stock_data) > 12:    
        plt.axvline(stock_data['ds'].iloc[-12], color='black', linestyle='dotted', alpha=0.7)  # Dotted line for split  
  
# General plot formatting  
plt.xlabel('Timestamp [t]')  
plt.ylabel('Stock value')  
plt.title('True vs. Forecast Values per Stock')  
plt.grid(alpha=0.4)  # Less obtrusive grid  
  
# Adjust legend to fit better  
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')   
  
plt.tight_layout()   
plt.show()

resulting in pretty accurate predictions by the NHITS model:

![](https://miro.medium.com/v2/resize:fit:700/1*7Cx7Vz3BoYBljdFpxj9HJQ.png)

With a couple of lines of code, we can run an advanced model on multiple time series. That's pretty impressive, I would say.

# Advanced analysis

Let’s pick up the pace and build code to analyze multiple models. We aim to discover the optimal hyperparameters for these models, train them, and visualize the results.

First, we need to install Ray and StatsForecast. We use Ray to find the optimal parameters for a model and StatsForecast to plot the results of our trained models.

pip install statsforecast ray

In this example, we compare the NHITS model with the Autoformer model. First, we define the parameters we want to sweep and the ranges Ray should search in

config_nhits = {  
    "input_size": tune.choice([48, 48*2, 48*3]),                
    "start_padding_enabled": True,  
    "n_blocks": 5*[1],                                                
    "mlp_units": 5 * [[64, 64]],                                    
    "n_pool_kernel_size": tune.choice([5*[1], 5*[2], 5*[4],  
                                      [8, 4, 2, 1, 1]]),              
    "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],  
                                      [1, 1, 1, 1, 1]]),              
    "learning_rate": tune.loguniform(1e-4, 1e-2),                     
    "scaler_type": tune.choice([None]),                               
    "max_steps": tune.choice([1000]),                                 
    "batch_size": tune.choice([1, 4, 10]),                            
    "windows_batch_size": tune.choice([128, 256, 512]),               
    "random_seed": tune.randint(1, 20),                               
}  
  
config_autoformer = {  
    "input_size": tune.choice([48, 48*2, 48*3]),                
    "encoder_layers": tune.choice([2,4]),                       
    "learning_rate": tune.loguniform(1e-4, 1e-2),               
    "scaler_type": tune.choice(['robust']),                     
    "max_steps": tune.choice([500, 1000]),                      
    "batch_size": tune.choice([1, 4]),                          
    "random_seed": tune.randint(1, 20),                         
}

We don’t discuss the parameters, and if you want to know more, we recommend reading the models' documentation. Next, we define the NeuralForecast object. Note that we use the **Auto**Autoformer and **Auto**NHITS, which are different from what we used before, enabling Ray to do the heavy lifting for us.

nf = NeuralForecast(  
    models=[  
        AutoAutoformer(h=48, config=config_autoformer, loss=MQLoss(), num_samples=2),  
        AutoNHITS(h=48, config=config_nhits, loss=MQLoss(), num_samples=5),  
    ],  
    freq='H'  
)

Then, the whole procedure starts with the following line of code (this can take a while to finish)

nf.fit(df=hist)

With the trained model, we want to do some inference. Thus, we make the predictions and clean up the resulting data frame.

fcst_df = nf.predict()  
fcst_df.columns = fcst_df.columns.str.replace('-median', '')

StatsForecast knows how to deal with the data frame we just created. With a couple of lines of code, we can plot our models’ output:

from statsforecast import StatsForecast  
StatsForecast.plot(hist, fcst_df, engine='matplotlib', max_insample_length=48 * 3, level=[80, 90])

![](https://miro.medium.com/v2/resize:fit:2168/1*4TbKq4ZXq_v8oE7SMlRsqw.png)

In this plot, we observe the historical data used for training and the inference in the future. It plots the expected value for both models and the 80–90 percent levels as shaded areas. We run the following code when we want to study a single model: NHIST.

StatsForecast.plot(hist, fcst_df, models=["AutoNHITS"], engine='matplotlib', max_insample_length=48 * 3, level=[80, 90])

which shows the plots for the NHIST model

![](https://miro.medium.com/v2/resize:fit:2168/1*5Zur3u1haH17fyXjiIbLSg.png)

with this advanced setup, we (1) performed a parameter sweep with Ray, (2) trained the model with the selected hyperparameters, and (3) made predictions with the trained model and plotted the stock values with confidence bounds. All this with just a couple of lines of code, thanks to the people from NIXTLA!

# Conclusion

It is simple, elegant, and works! We showed that running the state-of-the-art is not tricky, and with a couple of lines of code, you can run the most advanced time-series models. We know the adventure starts when you have it up and running with your data, but that doesn’t require a considerable investment. You can try it by opening this [Colab notebook](https://colab.research.google.com/drive/1AIxSlN8PE2EZAfG5RJukMFy-yO5xKzZb?usp=sharing) and playing with your time-series data.

# Disclaimer

The information contained herein is of a general nature and is not intended to address the specific circumstances of any particular individual or entity. The views expressed here are my personal views and do not necessarily reflect the position of my employer.

# References

- [https://aroussi.com/post/download-options-data](https://aroussi.com/post/download-options-data)
- Olivares, K. G., Challú, C., Garza, F., Canseco, M. M., & Dubrawski, A. (2022). NeuralForecast: User friendly state-of-the-art neural forecasting models [Conference presentation]. PyCon Salt Lake City, Utah, US.[https://github.com/Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast)
- Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., & Dubrawski, A. (2023, June). Nhits: Neural hierarchical interpolation for time series forecasting. In _Proceedings of the AAAI conference on artificial intelligence_ (Vol. 37, №6, pp. 6989–6997).
- Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. _Advances in neural information processing systems_, _34_, 22419–22430.




