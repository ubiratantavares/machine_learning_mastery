import matplotlib.pyplot as plt
import torch

x = torch.tensor(3.0, requires_grad=True)
print(x)

y = 3 * x ** 2
print(y)
y.backward()
print(x.grad)
print("\n")

print(x.data)
print(x.grad)
print(x.grad_fn)
print(x.is_leaf)
print(x.requires_grad)
print("\n")

print(y.data)
print(y.grad_fn)
print(y.is_leaf)
print(y.requires_grad)
print("\n")

x = torch.tensor(3.0, requires_grad = True)
y = 6 * x ** 2 + 2 * x + 4
print(y)
y.backward()
print(x.grad)
print("\n")


u = torch.tensor(3., requires_grad = True)
v = torch.tensor(4., requires_grad = True)

f = u**3 + v**2 + 4*u*v

print(u)
print(v)
print(f)

f.backward()

print(u.grad)
print(v.grad)
print("\n")

# compute the derivative of the function with multiple values
x = torch.linspace(-20, 20, 20, requires_grad = True)
Y = x ** 2
y = torch.sum(Y)
y.backward()

# ploting the function and derivative
function_line, = plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'Function')
function_line.set_color("red")
derivative_line, = plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'Derivative')
derivative_line.set_color("green")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function and Derivative')    
plt.legend()
plt.show()
