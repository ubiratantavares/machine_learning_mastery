
[Link](https://machinelearningmastery.com/basic-statistical-analysis-with-numpy/)

## Introduction

Statistical analysis is important in data science. It helps us understand data better. NumPy is a key Python library for numerical operations. It simplifies and speeds up this process. In this article, we will explore several functions for basic statistical analysis offered by NumPy.

NumPy is a Python library for numerical computing. It helps with working on arrays and mathematical functions. It makes calculations faster and easier. NumPy is essential for data analysis and scientific work in Python.

To get started, you first need to import NumPy to do statistical analysis.

|   |   |
|---|---|
|1|import numpy asnp|

By convention, we use `np` as an alias for NumPy. This makes it easier to call its functions.

Let’s now have a look at several key statistical functions for basic statistical analysis in NumPy.

## Mean

The mean is a measure of central tendency. It is the total of all values divided by how many values there are. We use the **mean()** function to calculate the mean.

Syntax: `np.mean(data)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br># Calculate the mean<br><br>mean=np.mean(data)<br><br># Print the result<br><br>print(f"Mean: {mean}")<br><br># Mean: 3.0|

## Average

The average is often used interchangeably with the mean. It is the total of all values divided by how many values there are. We use **average()** function to calculate the average. This function is useful because it allows for the inclusion of weights to compute a weighted average.

Syntax: `np.average(data)`, `np.average(data, weights=weights)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br>weights=np.array([1,2,3,4,5])<br><br># Calculate the average<br><br>average=np.average(data)<br><br># Calculate the weighted average<br><br>weighted_average=np.average(data,weights=weights)<br><br># Print the results<br><br>print(f"Average: {average}")<br><br>print(f"Weighted Average: {weighted_average}")<br><br># Average: 3.0<br><br># Weighted Average: 3.6666666666666665|

## Median

The median is the middle value in an ordered dataset. The median is the middle value when the dataset has an odd number of values. The median is the average of the two middle values when the dataset has an even number of values. We use the **median()** function to calculate the median.

Syntax: `np.median(data)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br># Calculate the median<br><br>median=np.median(data)<br><br># Print the result<br><br>print(f"Median: {median}")<br><br># Median: 3.0|

## Variance

Variance measures how spread out the numbers are from the mean. It shows how much the values in a dataset differ from the average. A higher variance means more spread. We use the **var()** function to calculate the variance.

Syntax: `np.var(data)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br># Calculate the variance<br><br>variance=np.var(data)<br><br># Print the result<br><br>print(f"Variance: {variance}")<br><br># Variance: 2.0|

## Standard Deviation

Standard deviation shows how much the numbers vary from the mean. It is the square root of variance. A higher standard deviation means more spread. It’s easier to understand because it uses the same units as the data. We use the **std()** function to calculate the standard deviation.

Syntax: `np.std(data)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br># Calculate the standard deviation<br><br>std_dev=np.std(data)<br><br># Print the result<br><br>print(f"Standard Deviation: {std_dev}")<br><br># Standard Deviation: 1.4142135623730951|

## Minimum and Maximum

The minimum and maximum functions help identify the smallest and largest values in a dataset, respectively. We use the **min()** and **max()** functions to calculate these values.

Syntax: `np.min(data)`, `np.max(data)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br># Calculate the minimum and maximum<br><br>minimum=np.min(data)<br><br>maximum=np.max(data)<br><br># Print the results<br><br>print(f"Minimum: {minimum}")<br><br>print(f"Maximum: {maximum}")<br><br># Minimum: 1<br><br># Maximum: 5|

## Percentiles

Percentiles show where a value stands in a dataset. For example, the 25th percentile is the value below which 25% of the data falls. Percentiles help us understand the distribution of the data. We use the **percentile()** function to calculate percentiles.

Syntax: `np.percentile(data, percentile_value)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br># Calculate the 25th and 75th percentiles<br><br>percentiles=np.percentile(data,[25,75])<br><br># Print the results<br><br>print(f"25th Percentile: {percentiles[0]}")<br><br>print(f"75th Percentile: {percentiles[1]}")<br><br># 25th Percentile: 2.0<br><br># 75th Percentile: 4.0|

## Correlation Coefficient

The correlation coefficient shows how two variables relate linearly. It ranges from -1 to 1. A value of 1 means a positive relationship. A value of -1 means a negative relationship. A value of 0 means no linear relationship. We use the **corrcoef()** function to calculate the correlation coefficient.

Syntax: `correlation_matrix = np.corrcoef(data1, data2)`, `correlation_coefficient = correlation_matrix[0, 1]`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12|# Sample data<br><br>data1=np.array([1,2,3,4,5])<br><br>data2=np.array([5,4,3,2,1])<br><br># Calculate the correlation coefficient matrix<br><br>correlation_matrix=np.corrcoef(data1,data2)<br><br># Extract the correlation coefficient between data1 and data2<br><br>correlation_coefficient=correlation_matrix[0,1]<br><br>print(f"Correlation Coefficient: {correlation_coefficient}")<br><br># Correlation Coefficient: -1.0|

## Range (Peak-to-Peak)

Range (Peak-to-Peak) measures the spread of data. It is the difference between the highest and lowest values. This helps us see how spread out the data is. We use the **ptp()** function from to calculate the range.

Syntax: `range = np.ptp(data)`

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10|# Sample data<br><br>data=np.array([1,2,3,4,5])<br><br># Calculate the range<br><br>range=np.ptp(data)<br><br># Print the result<br><br>print(f"Range: {range}")<br><br># Range: 4|

## Conclusion

NumPy helps with basic statistical analysis. For more complex statistics, other libraries like SciPy can be used. Knowing these basics helps improve data analysis.

