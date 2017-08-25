---
layout: default
title: Short Term Forecasting using Recurrent Neural Networks
description: Short term wind forecasting using recurrent neural networks (LTSM) and Keras 
categories: [machine-learning, forecasting, LSTM]
---

# Short Term Forecasting using Recurrent Neural Networks

## Overview

Short term forecasting in the renewable energy sector is becoming increasingly important. With renewables penetrating the market faster than expected, and the inherent uncertainty involved with weather forecasts, it is putting a lot of strain on the existing energy providers and distributors to both manage and balance the power grid. Doing so efficiently requires accurate knowledge of both both supply and demand. As both of these are future events, forecasting is required to predict these factors. Energy demand is a more stable signal than supply of renewable energy which is based on local weather systems relative to the energy generation. Of course demand can also spike unexpectedly with events such as extreme weather spells. Bad predictions can cause energy providers to end up with a shortfall in supply which means they will be required to generate the shortfall by burning more expensive and less climate friendly fuels. It can also result in large oversupply where the company is burning fuel needlessly with is both bad economically and for the climate. Therefore, better predictions in the [1, 48] hour time horizon are absolutely central to efficiently balancing supply and demand in the energy grid. 

In this post, we will show you how to implement a short term weather forecast using a type of deep learning known as recurrent neural networks ([RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)). The particular type of RNN we use is called a Long Short Term Memory ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)) network. We will use [Keras](https://keras.io/) (version 2+) with the [TensorFlow](https://www.tensorflow.org/) backend as the framework for building this network. 

In this post we're not going to argue the merits of deep learning via LSTMs vs more classical methods such as [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) or vice versa. Instead we're simply showing how to approach and implement a forecasting problem using LSTMs. In general simpler machine learning models should be tried first.


### ARIMA

Auto-Regressive Integrated Moving Average ([ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)) models are the most general and commonly used class of models for forecasting of a stationary signal (or on which can be made stationary). They support random walk, seasonal trend, non-seasonal exponentional smoothing and autoregressive models. 

Lags of the stationarized series in the forecasting equation are called "autoregressive" terms, lags of the forecast errors are called "moving average" terms, and a time series which needs to be differenced to be made stationary is said to be an "integrated" version of a stationary series. Random-walk and random-trend models, autoregressive models, and exponential smoothing models are all special cases of ARIMA models. See [here](https://people.duke.edu/~rnau/411arim.htm) for more details. While there is a relatively systematic approach to determining the ARIMA model parameters. There are many many publications and applications of ARIMA to renewables forecasting with various extensions to solve different issues. 

To see an application of ARIMA for forecasting and the method for determining the model parameters, see the following [notebook](https://datascience.ibm.com/exchange/public/entry/view/815137c868b916821dec777bdc23013c). 


### Why use a Recurrent Neural Network

Recurrent neural networks enable the learning and encoding of temporal features of a signal. This is idea for forecasting signals which are in some way predictive based on past events. LSTMs are a type of recurrent networks that overcome some of the historic issues related to training recurrent networks, such as the [vanishing gradients problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). We won't delve into the details of LSTMs here and will instead point you to this [article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a thorough overview. 



```python
%matplotlib inline

import seaborn as sns
sns.set_color_codes()
```

### Observational Data

We get the observational data from NOAA. NOAA collects the weather station observations from stations in countries all around the world. To get it for a given station, you just need to know whats known as the USAF and WBAN IDs. We are using Barcelona El Prat Airport in this notebook but you can change to whatever. 

We providea script that allows you to download the data locally

    data/donwload-observations.sh
    
We then read and parse this data for the exact weather observation station we wish to work with (there are, or have been multiple in El Prat). Don't use data from many stations as its unlikely to be consistent (same height, location etc). 

Sometimes the stations have readings every 30 minutes and sometimes every hour. We have seen that not all of the meterological readings are available every 30 minutes in such circumstances and so in this example we only use the hourly data. 



```python
import gzip
from io import BytesIO
from ish_parser import ish_parser

def read_observations(years, usaf='081810', wban='99999'):
    parser = ish_parser()
    
    for year in years:
        path = "data/observations/{usaf}-{wban}-{year}.gz".format(year=year, usaf=usaf, wban=wban)
        with gzip.open(path) as gz:
            parser.loads(bytes.decode(gz.read()))
            
    reports = parser.get_reports()
    
    station_latitudes = [41.283, 41.293] 
    observations = pd.DataFrame.from_records(((r.datetime, 
                                               r.air_temperature.get_numeric(),
                                               (r.precipitation[0]['depth'].get_numeric() if r.precipitation else 0),
                                               r.humidity.get_numeric(),
                                               r.sea_level_pressure.get_numeric(),
                                               r.wind_speed.get_numeric(),
                                               r.wind_direction.get_numeric()) 
                                              for r in reports if r.latitude in station_latitudes and r.datetime.minute == 0),
                             columns=['timestamp', 'AT', 'precipitation', 'humidity', 'pressure', 'wind_speed', 'wind_direction'], 
                             index='timestamp')
    
    observations = observations[['AT', 'precipitation', 
                             'humidity', 'pressure', 
                             'wind_speed']]
    
    return observations

```

### Numerical Weather Models

The numerical weather model (NWM) we're going to use to help us take into account information other than the local station observations is the NEMS4 model. This is provided free of charge to download by [MeteoBlue](https://www.meteoblue.com) for a given date range for a given station. We read and parse this raw download format here.  

Using a weather model is really important as you cannot hope to account for weather systems moving across a local region without it. An important point to note is that the weather model and observations are not meant to match exactly as the model may be predicting for a different height than the observations. However the boundary and ramp events are what we're after. 

TODO: Get this info off Nico


```python
import json
import pandas as pd
import numpy as np

nems4_lookahead=12

def read_nems4(years, prediction_hours=12):
    predictions=pd.DataFrame()
    for year in years:
        with open('data/NEMS4/{}.json'.format(year)) as json_data:
            d = json.load(json_data)
            if not predictions.empty:
                predictions = predictions.append( pd.DataFrame(d['history_1h']))
            else:
                predictions = pd.DataFrame(d['history_1h'])

    predictions = predictions.set_index('time')
    predictions.index.name = 'timestamp'
    
    # shift dataset back 12 hours as its a the value is the prediction for the given timestmap 12 hours previously
    predictions.index = pd.to_datetime(predictions.index) - pd.Timedelta(hours=nems4_lookahead)
    predictions.index.tz = 'UTC'

    predictions = predictions[['temperature', 'precipitation', 
                   'relativehumidity', 'sealevelpressure', 
                   'windspeed']]
    
    predictions = predictions.rename(columns={
        'windspeed': 'nems4_wind_speed', 
        'temperature': 'nems4_AT',
        'precipitation': 'nems4_precipitation',
        'relativehumidity': 'nems4_humidity',
        'sealevelpressure': 'nems4_pressure'})
    
    return predictions

```

### Join the NWM and Observations
Join the datasets by the timestamps


```python
years = range(2010, 2015)
dataset = pd.merge(read_observations(years), read_nems4(years), left_index=True, right_index=True, how='inner')

original = dataset.copy(deep=True)
dataset.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AT</th>
      <th>precipitation</th>
      <th>humidity</th>
      <th>pressure</th>
      <th>wind_speed</th>
      <th>nems4_AT</th>
      <th>nems4_precipitation</th>
      <th>nems4_humidity</th>
      <th>nems4_pressure</th>
      <th>nems4_wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>43177.000000</td>
      <td>43130.000000</td>
      <td>43162.000000</td>
      <td>35117.000000</td>
      <td>43213.000000</td>
      <td>43217.000000</td>
      <td>43217.000000</td>
      <td>43217.000000</td>
      <td>43217.000000</td>
      <td>43217.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16.911279</td>
      <td>0.084672</td>
      <td>68.819865</td>
      <td>1016.236418</td>
      <td>4.076880</td>
      <td>16.149210</td>
      <td>0.037606</td>
      <td>71.680380</td>
      <td>1015.828378</td>
      <td>3.464177</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.829713</td>
      <td>1.083148</td>
      <td>14.874102</td>
      <td>6.826032</td>
      <td>2.096274</td>
      <td>6.763601</td>
      <td>0.272675</td>
      <td>15.757395</td>
      <td>7.093197</td>
      <td>2.032260</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.500000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>980.200000</td>
      <td>0.000000</td>
      <td>-5.840000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>982.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.800000</td>
      <td>0.000000</td>
      <td>60.000000</td>
      <td>1012.500000</td>
      <td>2.600000</td>
      <td>11.130000</td>
      <td>0.000000</td>
      <td>61.000000</td>
      <td>1012.000000</td>
      <td>1.930000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16.800000</td>
      <td>0.000000</td>
      <td>70.000000</td>
      <td>1016.700000</td>
      <td>3.600000</td>
      <td>16.410000</td>
      <td>0.000000</td>
      <td>73.000000</td>
      <td>1016.000000</td>
      <td>3.090000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>79.000000</td>
      <td>1020.300000</td>
      <td>5.100000</td>
      <td>21.400000</td>
      <td>0.000000</td>
      <td>84.000000</td>
      <td>1020.000000</td>
      <td>4.640000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>35.300000</td>
      <td>56.000000</td>
      <td>100.000000</td>
      <td>1038.700000</td>
      <td>17.000000</td>
      <td>39.210000</td>
      <td>16.000000</td>
      <td>100.000000</td>
      <td>1037.000000</td>
      <td>22.320000</td>
    </tr>
  </tbody>
</table>
</div>



### Transformations - Preprocessing

There are a number of preprocessing steps we must take on this data. We only perform the bare minimum in this notebook as an example of how to work with the data. Much more time and analysis should be spend working with missing datapoints in particular and investigating or trying to detect where the instruments may not be working well (e.g. it is reading zero for a given period). Looking at the dataframe statistics in the cell above provide one with a quick eyeball estimate of outliers which should be filtered - for example, if the wind was 100m/sec at any point it is obviously wrong. 

- Dropping Duplicates

As the name implies, we need to ensure we have only unique timepoints. 

- Imputing Missing Values

As mentioned, this step needs some more love. We apply very simple forward filling in this notebook. 

- Standardize / Stationary Signal

A nice resource for understanding stationarity is [here](https://people.duke.edu/~rnau/411diff.htm). A standardised series is easier to predict. Weather data in particular is non stationary with cyclical trends. Hence, we use first order differencing here to transform the signal from non stationary to stationary. We could spend an entire notebook anlaysing this and showing why etc. Instead we refer you back to the [resource](https://people.duke.edu/~rnau/411diff.htm). 

- Normalize

First order differencing results in an approximately zero mean. Despite some sources, it is better to have the data centered around zero for a neural network than have it scaled between [0,1] for example (especially when using tanh activation function). However, an important reason for apply some scaling is to ensure that certain input features don't dominate the learning (if one feature after standardising had range [-1000,1000] and another [-1,1], then the contribution of the first to the distance will dominate that of the second). See this [article](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html) for an interesting discussion on the topic. 

Be careful not to use the MinMaxScaler after standardising the signal via first order differences. If you do, you may notice that the mean is no longer zero. This is exactly what we don't want. Therefore in this notebook we use the sklearn StandardScaler which ensures centering and scaling are applied on each input column (feature). 

- Add Horizon (Prediction) columns

As we are working with first order difference values for the wind speed, our predictions are going to be the differences also at the various time horizons. There are two ways we can achieve this. 

The first way is to predict the difference for each time point up to our max horizon. That is if we have a 3 hour horizon we would predict 3 values for hour [1,2,3] respectively. Then to get the prediction at hour 3 we would get the current real value and add the differences for each hour. A major issue with this approach is that the errors compound. At hour 3 we have included the errors for the predictions at hour 1 and 2 also. Extend that to a larger horizon and it doesn't work well. Additionally, you must predict the value at each time step up to the horizon and if the horizon is large it increases the complexity of the network (number of parameters) needlessly.

The second approach is to simply predict the difference at the horizons. So if we wanted to predict for horizons [1,3] then we only need to predict these two values and the prediction at hour 3 can be obtained by adding the 3 hour difference to the current value. This prevents compounding of errors. However, we are not sure if this approach causes stationarity problems in our predictions. 


```python
from sklearn import preprocessing

def drop_duplicates(df):
    print("Number of duplicates: {}".format(len(df.index.get_duplicates())))
    return df[~df.index.duplicated(keep='first')]
    
def impute_missing(df):
    # todo test with moving average / mean or something smarter than forward fill
    print("Number of rows with nan: {}".format(np.count_nonzero(df.isnull())))
    df.fillna(method='ffill', inplace=True)
    return df
    
def first_order_difference(data, columns):
    for column in columns:
        data[column+'_d'] = data[column].diff(periods=1)
    
    return data.dropna()

def derive_prediction_columns(data, column, horizons):
    pd.options.mode.chained_assignment = None
    
    for look_ahead in horizons:
        data['prediction_' + str(look_ahead)] = data[column].diff(periods=look_ahead).shift(-look_ahead)
    
    return data.dropna()

def scale_features(scaler, features):
    scaler.fit(features)
    
    scaled = scaler.transform(features)
    scaled = pd.DataFrame(scaled, columns=features.columns)
    
    return scaled

def inverse_prediction_scale(scaler, predictions, original_columns, column):
    loc = original_columns.get_loc(column)
    
    inverted = np.zeros((len(predictions), len(original_columns)))
    inverted[:,loc] = np.reshape(predictions, (predictions.shape[0],))
    
    inverted = scaler.inverse_transform(inverted)[:,loc]
    
    return inverted

def invert_all_prediction_scaled(scaler, predictions, original_columns, horizons):
    inverted = np.zeros(predictions.shape)
    
    for col_idx, horizon in enumerate(horizons):
        inverted[:,col_idx] = inverse_prediction_scale(
            scaler, predictions[:,col_idx], 
            original_columns,
            "prediction_" + str(horizon))
        
    return inverted

def inverse_prediction_difference(predictions, original):
    return predictions + original

def invert_all_prediction_differences(predictions, original):
    inverted = predictions
    
    for col_idx, horizon in enumerate(horizons):
        inverted[:, col_idx] = inverse_prediction_difference(predictions[:,col_idx], original)
        
    return inverted
```


```python
dataset = drop_duplicates(dataset)
dataset = impute_missing(dataset)

#select features we're going to use
features = dataset[['wind_speed', 
                    'nems4_wind_speed', 
                    'AT', 
                    'nems4_AT', 
                    'humidity', 
                    'nems4_humidity',
                    'pressure',
                    'nems4_pressure']]

# the time horizons we're going to predict (in hours)
horizons = [1, 12]

features = first_order_difference(features, features.columns)
features = derive_prediction_columns(features, 'wind_speed', horizons)

scaler = preprocessing.StandardScaler()
scaled = scale_features(scaler, features)

scaled.describe()
```

    Number of duplicates: 0
    Number of rows with nan: 0





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wind_speed</th>
      <th>nems4_wind_speed</th>
      <th>AT</th>
      <th>nems4_AT</th>
      <th>humidity</th>
      <th>nems4_humidity</th>
      <th>pressure</th>
      <th>nems4_pressure</th>
      <th>wind_speed_d</th>
      <th>nems4_wind_speed_d</th>
      <th>AT_d</th>
      <th>nems4_AT_d</th>
      <th>humidity_d</th>
      <th>nems4_humidity_d</th>
      <th>pressure_d</th>
      <th>nems4_pressure_d</th>
      <th>prediction_1</th>
      <th>prediction_12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
      <td>4.310900e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.041692e-16</td>
      <td>3.533841e-16</td>
      <td>-2.320732e-16</td>
      <td>-2.188872e-16</td>
      <td>1.766921e-16</td>
      <td>-2.505335e-17</td>
      <td>3.858216e-15</td>
      <td>2.761143e-15</td>
      <td>-1.633825e-17</td>
      <td>1.218673e-17</td>
      <td>4.779916e-18</td>
      <td>1.454578e-17</td>
      <td>2.301365e-17</td>
      <td>-1.589013e-17</td>
      <td>-1.528749e-17</td>
      <td>-3.840415e-17</td>
      <td>-2.617107e-17</td>
      <td>1.823373e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
      <td>1.000012e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.944119e+00</td>
      <td>-1.704865e+00</td>
      <td>-2.993382e+00</td>
      <td>-3.254907e+00</td>
      <td>-4.086010e+00</td>
      <td>-3.784563e+00</td>
      <td>-5.122080e+00</td>
      <td>-4.768033e+00</td>
      <td>-6.395646e+00</td>
      <td>-1.447074e+01</td>
      <td>-8.009811e+00</td>
      <td>-8.656896e+00</td>
      <td>-6.802778e+00</td>
      <td>-9.080636e+00</td>
      <td>-2.591511e+01</td>
      <td>-6.966350e+00</td>
      <td>-6.395534e+00</td>
      <td>-4.641788e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.044104e-01</td>
      <td>-7.551439e-01</td>
      <td>-7.518787e-01</td>
      <td>-7.404916e-01</td>
      <td>-5.922197e-01</td>
      <td>-6.764342e-01</td>
      <td>-5.423194e-01</td>
      <td>-5.382363e-01</td>
      <td>-7.267316e-01</td>
      <td>-5.333957e-01</td>
      <td>-5.789955e-01</td>
      <td>-5.364996e-01</td>
      <td>-5.101937e-01</td>
      <td>-5.342834e-01</td>
      <td>-3.715364e-01</td>
      <td>-1.034077e-03</td>
      <td>-7.266890e-01</td>
      <td>-6.676166e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.275993e-01</td>
      <td>-1.843273e-01</td>
      <td>-4.710922e-03</td>
      <td>3.897706e-02</td>
      <td>7.966304e-02</td>
      <td>8.474023e-02</td>
      <td>8.739766e-02</td>
      <td>2.573652e-02</td>
      <td>5.226357e-05</td>
      <td>-1.180667e-02</td>
      <td>-9.647500e-02</td>
      <td>-1.519434e-01</td>
      <td>1.578045e-05</td>
      <td>-1.362968e-04</td>
      <td>-1.339644e-03</td>
      <td>-1.034077e-03</td>
      <td>8.598094e-05</td>
      <td>3.114281e-04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.876172e-01</td>
      <td>5.784018e-01</td>
      <td>8.303589e-01</td>
      <td>7.755527e-01</td>
      <td>6.843575e-01</td>
      <td>7.824835e-01</td>
      <td>6.169324e-01</td>
      <td>5.897093e-01</td>
      <td>4.361226e-01</td>
      <td>4.855224e-01</td>
      <td>3.860455e-01</td>
      <td>3.757035e-01</td>
      <td>5.102253e-01</td>
      <td>5.340108e-01</td>
      <td>3.688571e-01</td>
      <td>-1.034077e-03</td>
      <td>4.361510e-01</td>
      <td>6.682395e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.161669e+00</td>
      <td>9.278434e+00</td>
      <td>2.690953e+00</td>
      <td>3.408293e+00</td>
      <td>2.095311e+00</td>
      <td>1.797383e+00</td>
      <td>3.250295e+00</td>
      <td>2.986594e+00</td>
      <td>6.759142e+00</td>
      <td>1.261550e+01</td>
      <td>7.623853e+00</td>
      <td>8.317237e+00</td>
      <td>7.313019e+00</td>
      <td>1.157305e+01</td>
      <td>2.757832e+01</td>
      <td>6.964282e+00</td>
      <td>6.759093e+00</td>
      <td>4.275051e+00</td>
    </tr>
  </tbody>
</table>
</div>



### Creating the Test & Train Datasets for Keras LSTM

In Keras there are two types of LSTM configurations. One is called stateful and the other non stateful. This terminology can be confusing as after all, the very reason for using LSTMs is their temporal memory (i.e. state). This is different to what the term stateful in the LSTM configuration in Keras means.

A stateful Keras LSTM network is one where the internal LSTM units are not reset at all during a training epoc (in fact even between epocs one must manually reset them). This means that the LSTMs build and keep state for the entire training set which means the data must be played through in order. While this is desirable to learn longer term temporal features and dependencies it can be problematic if we have a certain temporal window we want to focus on. 

A non stateful LSTM in Keras terminology resembles more of a classic sliding window approach to training, except where the successive timesteps in the window are framed one after another as a two dimensional array instead of flattened out into a long one dimensional array. This allows the LSTM to learn the state and temporal dependencies between the frames in the time window and Keras will automatically reset the state between each training window. This means that there is no ordering or continuity requirements on the training data outside of the windows.

We cannot reasonably expect the model to pick up very long term dependencies for the weather here given even the most accurate global weather models are chance at horizons of 10 days. Therefore we define our window size (sequence_length) to be 48 hours. You can change to 72 etc to see the difference. The nice things with LSTMs is that you do not increase the number of parameters in the LSTM network by changing this sequence_lenght.

The shape of the X datasets will be a 3D tensor

    (n_samples, sequence_length, n_features)


We are predicting multiple time horizons into the future. Hence we will have multiple network outputs, one for each horizon, which is a 2D tensor

    (n_samples, n_horizons)
    



```python
def prepare_test_train(data, features, predictions, sequence_length, split_percent=0.9):
    
    num_features = len(features)
    num_predictions = len(predictions)
    
     # make sure prediction cols are at end
    columns = features + predictions
    
    data = data[columns].values
    
    print("Using {} features to predict {} horizons".format(num_features, num_predictions))
    
    result = []
    for index in range(len(data) - sequence_length+1):
        result.append(data[index:index + sequence_length])

    result = np.array(result)
    # shape (n_samples, sequence_length, num_features + num_predictions)
    print("Shape of data: {}".format(np.shape(result)))
    
    row = round(split_percent * result.shape[0])
    train = result[:row, :]
    #np.random.shuffle(train) # not using stateful lstm
    
    X_train = train[:, :, :-num_predictions]
    y_train = train[:, -1, -num_predictions:]
    X_test = result[row:, :, :-num_predictions]
    y_test = result[row:, -1, -num_predictions:]
    
    print("Shape of X train: {}".format(np.shape(X_train)))
    print("Shape of y train: {}".format(np.shape(y_train)))
    print("Shape of X test: {}".format(np.shape(X_test)))
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))
    
    y_train = np.reshape(y_train, (y_train.shape[0], num_predictions))
    y_test = np.reshape(y_test, (y_test.shape[0], num_predictions))
    
    return X_train, y_train, X_test, y_test, row
```


```python
sequence_length = 48

prediction_cols = ['prediction_' + str(h) for h in horizons]
feature_cols = ['wind_speed_d', 'nems4_wind_speed_d', 
                'AT_d', 'nems4_AT_d', 
                'humidity_d', 'nems4_humidity_d', 
                'pressure_d', 'nems4_pressure_d']

X_train, y_train, X_test, y_test, row_split = prepare_test_train(
    scaled,
    feature_cols,
    prediction_cols,
    sequence_length,
    split_percent = 0.8)
```

    Using 8 features to predict 2 horizons
    Shape of data: (43062, 48, 10)
    Shape of X train: (34450, 48, 8)
    Shape of y train: (34450, 2)
    Shape of X test: (8612, 48, 8)


### Validate Test & Train Dataset Preparation

Ensure we can undo each transformation to get back the original signal. Sanity checks are always good

This is surprisingly tricky when performing first order differencing and the data preprocesssing / structuring required by the LSTM. Ensuring you are adding the correct "original" value is really important. 



```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

#(-1 is because we only take the last y row in each sequence)
sequence_offset = sequence_length  - 1

# validate train
inverse_scale = invert_all_prediction_scaled(scaler, y_train, scaled.columns, horizons)

assert(mean_squared_error(
    features[prediction_cols][sequence_offset:row_split+sequence_offset], 
    inverse_scale) < 1e-10)


undiff_prediction = invert_all_prediction_differences(
    inverse_scale, 
    features['wind_speed'][sequence_offset:row_split+sequence_offset])

for i, horizon in enumerate(horizons):
    assert(mean_squared_error(
        features['wind_speed'][sequence_offset+horizon:row_split+sequence_offset+horizon], 
        undiff_prediction[:,i]) < 1e-10)

    
# validate test
inverse_scale = invert_all_prediction_scaled(scaler, y_test, scaled.columns, horizons)

assert(mean_squared_error(
    features[prediction_cols][sequence_offset+row_split:], 
    inverse_scale) < 1e-10)

undiff_prediction = invert_all_prediction_differences(
    inverse_scale, 
    features['wind_speed'][sequence_offset+row_split:])

for i, horizon in enumerate(horizons):
    assert(mean_squared_error(
        features['wind_speed'][sequence_offset+row_split+horizon:], 
        undiff_prediction[:-horizon,i]) < 1e-10)
```

### Build the LSTM Model

Build the non stateful LSTM network. 

We apply regularisation via dropout between each layer in the network. This should help overfitting. The RMSProp optimizer is recommended when working with LSTMs. The only tunable property of this is the learning rate. Keras has some callbacks that allow for tuning of this as the training progresses (e.g. see [ReduceLROnPlateau](https://keras.io/callbacks/#reducelronplateau)). 

The first and last layers deserve a comment. The input_shape argument in the first layer specifies the number of input features which is X_train.shape[2]. The last layer is the output layer (hence linear activation) and the size is equal to the number of time horizons we're predicting - y_train.shape[1]. 


```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop

def build_model(layers):
    model = Sequential()
    
    model.add(LSTM(
            layers[1],
            input_shape=(None, layers[0]),
            return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(layers[2], return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(layers[3], return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(layers[4], activation="linear"))
    
    model.compile(loss="mse", optimizer=RMSprop)
    
    print(model.summary())
          
    return model
```

### Train and Evaluate the Model

Fit the model with the training data. We use a validation split of the training data (10%) to allow for things such as early stopping to work (it validates the validation loss against the training loss). 

The size of each network layer is passed in via an argument. The first and last values in that layers array represent the input and output size respectively. The other 3 values represent the size of the 3 layers of the network. 

The batch_size here refers to the number of samples to be taken between gradient updates. Powers of two are recommended and typically values are 256, 512 etc. Batch learning greatly improves the speed of training and the stability of the convergence. 


#### Using AWS or Google Cloud GPUs

A GPU is absolutely essential to train a netwok in a reasonble amount of time. Fortunately most of the cloud providers not provide access to a variety of options for deep learning with optimized instances and GPUs. We won't go into the details of how to do for each or the benefits of one versus the other in this post. We trained the model on a p2.2xlarge instance on AWS using the AWS Spot market which cost approx $0.3 / hour. Training **???** epocs took approximately **???** hours. 


#### Hyperparameter Tuning

Hyperparameter tuning is an essential part of any machine learning process. Do not fall into the trap of trying to manually hand tune the parameters. There are many formal approaches which optimize this (see my blog [post](http://www.willfleury.com/machine-learning/bayesian-optimization/2017/05/15/hyperparameter-optimisation.html) on the suject). If you are running on Google Cloud Machine Learning it actually has the ability to perform the hyperparameter optimization built into the API. Exactly what type of optimization it performs under the hood is unclear however. 




```python
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

def run_network(X_train, y_train, X_test, layers, epochs, batch_size=512):
    model = build_model(layers)
    history = None
    
    try:
        history = model.fit(
            X_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=0.1,
            callbacks=[
                TensorBoard(log_dir='/tmp/tensorboard', write_graph=True),
                EarlyStopping(monitor='val_loss', patience=2, mode='auto')
            ])
    except KeyboardInterrupt:
        print("Training interrupted")
    
    predicted = model.predict(X_test)
    
    return model, predicted, history

```


```python
model, predicted, history = run_network(
    X_train, 
    y_train, 
    X_test,
    layers=[X_train.shape[2], 60, 60, 60, y_train.shape[1]],
    epochs=4)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, None, 60)          16560     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, None, 60)          0         
    _________________________________________________________________
    lstm_2 (LSTM)                (None, None, 60)          29040     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, None, 60)          0         
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 60)                29040     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 60)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 122       
    =================================================================
    Total params: 74,762
    Trainable params: 74,762
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 31005 samples, validate on 3445 samples
    Epoch 1/4
    31005/31005 [==============================] - 86s - loss: 0.7620 - val_loss: 0.6697
    Epoch 2/4
    31005/31005 [==============================] - 86s - loss: 0.6419 - val_loss: 0.5950
    Epoch 3/4
    31005/31005 [==============================] - 87s - loss: 0.6080 - val_loss: 0.5966
    Epoch 4/4
    31005/31005 [==============================] - 84s - loss: 0.5961 - val_loss: 0.5811


### Validation

Get error scores for the predicted test data. We provide two measures here, [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) to match what the training was and [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) which can be more relevant for understanding a forecast accuracy. 

We also provide the error scores for each horizon individually. This is very important. We have noticed some articles and worse, libraries, which only provide the **averaged** error score across all the predicted horizons. Naturally, a 1 hour timestep prediction is going to be more accurate than a 24 hour. If we are predicting many horizons then the shorter horizons will be more accurate and reduce the average error score. Therefore its important to evaluate each horizon individually to understand your real predictive power at each step. 

We must also transform the predictions back into actual wind speed values. That means we must unscale and undifference the predictions at the various horizons. We then provide the error scores at these real world scales to again understand the predictions better in a real world context. 



```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("MSE {:.3}, MAE {:.3}".format(
    mean_squared_error(y_test, predicted),
    mean_absolute_error(y_test, predicted)))

for i, horizon in enumerate(horizons):
    print("MSE {:.3}, MAE {:.3} for horizon {}".format(
        mean_squared_error(y_test[:,i], predicted[:,i]),
        mean_absolute_error(y_test[:,i], predicted[:,i]),
        horizon))

```

    MSE 0.603, MAE 0.574
    MSE 0.818, MAE 0.669 for horizon 1
    MSE 0.387, MAE 0.479 for horizon 12



```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

inverse_scale = invert_all_prediction_scaled(scaler, predicted, scaled.columns, horizons)

predicted_signal = invert_all_prediction_differences(
    inverse_scale, 
    features['wind_speed'][sequence_offset+row_split:])

print("Original Signal MAE is {:.3}".format(
        mean_absolute_error(
            features[features.filter(regex='prediction').columns][sequence_offset+row_split:], 
            predicted_signal)))

for i, horizon in enumerate(horizons):
    print("Original Signal MAE for horizon {} is {:.3}".format(horizon,
        mean_absolute_error(
            features['prediction_' + str(horizon)][sequence_offset+row_split:], 
            predicted_signal[:,i])))

```

    Original Signal MAE is 4.3
    Original Signal MAE for horizon 1 is 4.2
    Original Signal MAE for horizon 12 is 4.4


### Visualising 

Finally, we should visualise the predicted wind speeds. We will draw a plot for each time horizon independently. 


```python
import matplotlib.pyplot as plt

# plot comparison with observation, numerical weather model and our prediction
plot_samples=250
plots = len(horizons)

real_signal = features['wind_speed'][sequence_offset+row_split:].values
nems4_predicted_signal = features['nems4_wind_speed'][sequence_offset+row_split:].shift(nems4_lookahead)

fig = plt.figure(figsize=(14, 5 * plots))
fig.suptitle("Model Prediction at each Horizon")

for i, horizon in enumerate(horizons):
    plt.subplot(plots,1,i+1)
    
    plt.plot(real_signal[:plot_samples], label='actual')
    plt.plot(nems4_predicted_signal.values[:plot_samples], label='nems4')
    plt.plot(predicted_signal[:plot_samples, i], label='predicted')
    plt.title("Prediction for {} Hour Horizon".format(horizon))
    plt.xlabel("Hour")
    plt.legend()

```

{% include image.html img="/assets/images/machine-learning/forecasting_image_0.png" %}

### Critique 

One very important property we are missing for our predictions, is the confidence our model has in a given prediction - aka credible interval. We can actually extend this model and add a Mixture Density Network ([MDN](http://edwardlib.org/tutorials/mixture-density-network)) as the final layer in the network. MDNs are very useful when combined with neural networks, where the outputs of the neural network are the parameters of the mixture model, rather than direct prediction of the data label. So for each input, you would have a set of mean parameters, a set of standard deviation parameters, and a set of probabilities that the output point would fall into those gaussian distributions ([taken from](http://blog.otoro.net/2015/06/14/mixture-density-networks/)). In a follow up post we will extend our model with an MDN.

Another critique is that we don't do any feature engineering at all. While one of the stated benefits of deep learning in general is its inherent ability to extract latent features, it would be beneficial at least to test out some standard forecasting features such as trend strenghts etc. They may or may not improve the result. Another interesting idea which Uber recently published about their use of LSTMs for forecasting was to use a separate LSTM autoencoder network to create additional features as input to the LSTM model for prediction. See their article [here](https://eng.uber.com/neural-networks/). This is very easy to achieve with Keras. 





