
[Link](https://www.kdnuggets.com/speeding-up-your-python-code-with-numpy)

There are several reasons why NumPy could accelerate the Python code execution, including:

- NumPy using C Code instead of Python during looping
- The better CPU caching process
- Efficient algorithms in mathematical operations
- Able to use parallel operations
- Memory-efficient in large datasets and complex computations

For many reasons, NumPy is effective in improving Python code execution. This tutorial will show examples of how NumPy speeds up the code process. Let's jump into it.

## NumPy in Accelerate Python Code Execution

   
The first example compares Python list and NumPy array numerical operations, which acquire the object with the intended value result.

For example, we want a list of numbers from two lists we add together so we perform the vectorized operation. We can try the experiment with the following code:

```
import numpy as np
import time

sample = 1000000

list_1 = range(sample)
list_2 = range(sample)
start_time = time.time()
result = [(x + y) for x, y in zip(list_1, list_2)]
print("Time taken using Python lists:", time.time() - start_time)

array_1 = np.arange(sample)
array_2 = np.arange(sample)
start_time = time.time()
result = array_1 + array_2
print("Time taken using NumPy arrays:", time.time() - start_time)
```

```
Output>>
Time taken using Python lists: 0.18960118293762207
Time taken using NumPy arrays: 0.02495265007019043
```

As you can see in the above output, the execution of NumPy arrays is faster than that of the Python list in acquiring the same result.

Throughout the example, you would see that the NumPy execution is faster. Let’s see if we want to perform aggregation statistical analysis.

```
array = np.arange(1000000)

start_time = time.time()
sum_rst = np.sum(array)
mean_rst = np.mean(array)
print("Time taken for aggregation functions:", time.time() - start_time)
```

```
Output>> 
Time taken for aggregation functions: 0.0029935836791992188
```

NumPy can process the aggregation function pretty fast. If we compare it with the Python execution, we can see the execution time differences.

```
list_1 = list(range(1000000))

start_time = time.time()
sum_rst = sum(list_1)
mean_rst = sum(list_1) / len(list_1)
print("Time taken for aggregation functions (Python):", time.time() - start_time)
```

```
Output>>
Time taken for aggregation functions (Python): 0.09979510307312012
```

With the same result, Python's in-built function would take much more time than NumPy. If we had a much bigger dataset, Python would take way longer to finish the NumPy.

Another example is when we try to perform in-place operations, we can see that the NumPy would be much faster than the Python example.

```
array = np.arange(1000000)
start_time = time.time()
array += 1
print("Time taken for in-place operation:", time.time() - start_time)
```

```
list_1 = list(range(1000000))
start_time = time.time()
for i in range(len(list_1)):
    list_1[i] += 1
print("Time taken for in-place list operation:", time.time() - start_time)
```

```
Output>>
Time taken for in-place operation: 0.0010089874267578125
Time taken for in-place list operation: 0.1937870979309082
```

The point of the example is that if you have an option to perform with NumPy, then it’s much better as the process would be much faster.

We can try a more complex implementation, using matrix multiplication to see how fast NumPy is compared to Python.

```
def python_matrix_multiply(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def numpy_matrix_multiply(A, B):
    return np.dot(A, B)

n = 200
A = [[np.random.rand() for _ in range(n)] for _ in range(n)]
B = [[np.random.rand() for _ in range(n)] for _ in range(n)]

A_np = np.array(A)
B_np = np.array(B)

start_time = time.time()
python_result = python_matrix_multiply(A, B)
print("Time taken for Python matrix multiplication:", time.time() - start_time)

start_time = time.time()
numpy_result = numpy_matrix_multiply(A_np, B_np)
print("Time taken for NumPy matrix multiplication:", time.time() - start_time)
```

```
Output>>
Time taken for Python matrix multiplication: 1.8010151386260986
Time taken for NumPy matrix multiplication: 0.008051872253417969
```

As you can see, NumPy is even faster in more complex activities, such as Matrix Multiplication, which uses standard Python code.

We can try out many more examples, but NumPy should be faster than Python's built-in function execution times.  
 

## Conclusion

   
NumPy is a powerful package for mathematical and numerical processes. Compared to the standard Python in-built function, NumPy execution time would be faster than the Python counterpart. That is why, try to use NumPy if it’s applicable to speed up our Python code.

[[DataScience]]

