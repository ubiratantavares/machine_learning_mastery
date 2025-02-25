# 10 Python One-Liners That Will Boost Your Data Science Workflow

Python is the most popular data science programming language, as it’s versatile and has a lot of support from the community. With so much usage, there are many ways to improve our data science workflow that you might not know.

In this article, we will explore ten different Python one-liners that would boost your data science work.

What are they? Let’s have a look.

## 1. Efficient Missing Data Handling

Missing data is a constant occurrence in datasets. It could happen because of a variety of reasons, from data mismanagement to natural conditions and beyond. Nevertheless, we need to decide how to handle the missing data.

Some would make it into the missing data category or drop them all. However, there are times we opt to fill in the missing data.

If we want to fill in the missing data, we can use the Pandas fillnamethod. It’s easy to use as we only need to pass the value we want to fill as the replacement for the missing value, but we can make it more efficient.

Let’s see the code below.

```Python
df.fillna({col: df[col].median() for col in df.select_dtypes(include='number').columns} |
          {col: df[col].mode()[0] for col in df.select_dtypes(include='object').columns}, inplace=True)
```

By combining the fillna with the condition, we can fill the numerical missing data with the median and the categorical missing data with the mode.

With one line, you can quickly fill in all the different missing data in other columns.

## 2. Highly Correlated Features Removal

Multicollinearity occurs when our dataset contains many independent variables that are highly correlated with each other instead of with the target. This negatively impacts the model performance, so we want to keep less correlated features.

We can combine the Pandas correlation feature with the conditional selection to quickly select the less correlated features. For example, here is how we can choose the features that have the maximum Pearson correlation with the others below 0.95.

```Python
df = df.loc[:, df.corr().abs().max() < 0.95]
```

Trying out the correlation features and the threshold to see if the prediction model is good or not.

## 3. Conditional Column Apply

Creating a new column with multiple conditions can sometimes be complicated, and the line to perform them can be long. However, we can use the apply method from the Pandas to use specific conditions when developing the new feature while still using multiple column values.

For example, here are examples of creating a new column where the values are based on the condition of the other column values.

```Python
df['new_col'] = df.apply(lambda x: x['A'] * x['B'] if x['C'] > 0 else x['A'] + x['B'], axis=1)
```

You can try out another condition that follows your requirements.

## 4. Finding Common and Different Element

Python provides many built-in data types, including Set. The Set data type is unique data that represents an unordered list of data but only with unique elements. It’s often used for many data operations, which include finding the common elements.

For example, we have the following set:

```Python
set1 = {"apple", "banana", "cherry", "date", "fig"}
set2 = {"cherry", "date", "elderberry", "fig", "grape"}
```

Then, we want to find the common element between both sets. We can use the following method.

```Python
set1.intersection(set2)
```

Output:

{'cherry', 'date', 'fig'}

It’s a simple but useful way to find the common element. In reverse, we can also find the elements that are different within both sets.

```Python
set1.difference(set2)
```

Output:

{'apple', 'banana'}

Try using them in your data workflow when you are required to find the common and different elements.

## 5. Boolean Masks for Filtering

When working with the NumPy array and its derivate object, we sometimes want to filter the data according to our requirements. In this case, we can create a boolean mask to filter the data based on the boolean condition we set.

Let’s say we have the following list of data.

```Python
import numpy as np
data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
```

Then, we can use the boolean mask to filter the data we want. For example, we want only even numbers.

```Python
data[(data % 2 == 0)]
```

Output:

array([10, 20, 30, 40, 50])

This is also the basis of the Pandas filtering; however, a Boolean mask can be more versatile as it works in the NumPy array as well.

## 6. List Count Occurrence

When working with a list or any other data with multiple values, there are times when we want to know the frequency for each value. In this case, we can use the counter function to count them automatically.

For example, consider having the following list.

```Python
data = [10, 10, 20, 20, 30, 35, 40, 40, 40, 50]
```

Then, we can use the counter function to calculate the frequency.

```Python
from collections import Counter
Counter(data)
```

Output:

Counter({10: 2, 20: 2, 30: 1, 35: 1, 40: 3, 50: 1})

The result is a dictionary for the count occurrence. Use them when you need quick frequency calculation.

## 7. Numerical Extraction from Text

Regular expressions (Regex) are defined character lists that match a pattern in text. They are usually used when we want to perform specific text manipulation, and that’s precisely what we can do with this one-liner.

In the example below, we can use a combination of Regex and map to extract numbers from the text.

```Python
import re
list(map(int, re.findall(r'\d+', "Sample123Text456")))
```

Output:

[123, 456]

The example above only works for integer data, but learning more about regular expressions can give you the power and flexibility to adapt this one-liner for multiple use cases.

## 8. Flatten Nested List

When we prepare our data for analysis, we can encounter list data that contains a list within the list, which we can call nested. If we find something like that, we might want to flatten it for further data analysis or visualization.

For example, let’s say we have the following nested list.

```Python
nested_list = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
]
```

We can then flatten the list with the following code.

```Python
sum(nested_list, [])
```

Output:

[1, 2, 3, 4, 5, 6, 7, 8, 9]

With this one-dimensional data list, you can analyze further and in a more straightforward manner if needed.

## 9. List to Dictionary

Have you ever got into a situation where you have several lists and want to combine the information in the dictionary form? For example, the use case may be related to mapping purposes or feature encoding.

In this case, we can convert the list we have into a dictionary using the zip function.

For example, we have the following list.

```Python
fruit = ['apple', 'banana', 'cherry']
values = [100, 200, 300]
```

With the combination of zip and dict, we can combine both of the lists above into one.

```Python
dict(zip(fruit, values))
```

Output:

{'apple': 100, 'banana': 200, 'cherry': 300}

This is a quick way to combine both pieces of data into one structure, which can then be used for further data preprocessing.

### 10. Dictionary Merging

When we have a dictionary that contains the information we require for data preprocessing, we should combine them. For example, we have performed the list to dictionary action like above and ended up with the following dictionaries:

```Python
fruit_mapping = {'apple': 100, 'banana': 200, 'cherry': 300}
furniture_mapping = {'table': 100, 'chair': 200, 'sofa': 300}
```
Then, we want to combine them as that information could be important as a whole. To do that, we can use the following one-liner.

```Python
{**fruit_mapping, **furniture_mapping }

Output&gt;&gt;
{'apple': 100,
 'banana': 200,
 'cherry': 300,
 'table': 100,
 'chair': 200,
 'sofa': 300}
```

As you can see, both dictionaries have become one dictionary. This is very useful in many cases that require you to aggregate data.

## Conclusion

In this article, we have explored ten different Python one-liners that would improve your data science workflow. These one-liners have focused on:

* Efficient Missing Data Handling

* Highly Correlated Features Removal

* Conditional Column Apply

* Finding Common and Different Element

* Boolean Masks for Filtering

* List Count Occurrence

* Numerical Extraction from Text

* Flatten Nested List

* List to Dictionary

* Dictionary Merging
