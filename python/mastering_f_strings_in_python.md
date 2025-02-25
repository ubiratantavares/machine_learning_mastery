# Mastering f-strings in Python
Discover how to leverage Python's f-strings (formatted string literals) to write cleaner, more efficient, and more readable code.
By Abid Ali Awan, KDnuggets Assistant Editor on November 7, 2024 in Python
FacebookTwitterLinkedInRedditEmailCompartilhar


f-strings in Python
Image by Author

 
F-strings, or formatted string literals, were introduced in Python 3.6 as a convenient and concise way to format strings. They provide a way to embed expressions inside string literals, using curly braces {}. F-strings are not only more readable and concise but also faster than the older string formatting methods. 

While many developers understand the basics, this guide explores advanced f-string techniques that can enhance code quality and efficiency. We will cover embedding expressions, formatting numbers, aligning text, using dictionaries, and multi-line f-strings.

The new way to Cloud
Migrate to Big Query

 

1. Basic Usage
 

To create an f-string, you simply prefix the string with the letter f or F. Inside the string, you can include variables and expressions within curly braces {}.

We have added two valuables; one is a string, and one is an int. 

name = "Abid"
age = 33
print(f"Hello, my name is {name} and I am {age} years old.")
 

Output:

Hello, my name is Abid and I am 33 years old.
 


2. Embedding Expressions
 

F-strings can evaluate expressions directly within the curly braces. You can even run any calculation within the curly brackets, and it will work. 

a = 6
b = 14
print(f"The sum of {a} and {b} is {a + b}.")
 

Output:

The sum of 6 and 14 is 20.
 

You can also call Python functions within the braces.

def get_greeting(name):
    return f"Hello, {name}!"

print(f"{get_greeting('Abid')}")
 

Output:

Hello, Abid!
 


3. Formatting Numbers
 

F-strings support number formatting options, which can be very useful for controlling the display of numerical values. Instead of writing multiple lines of code to format numbers manually, you can use f-strings to simplify the process. 

For example, we will display the float variable called `cost_ratio` with three decimal places. 

cost_ratio = 6.5789457766
print(f"Cost ratio rounded to 3 decimal places: {cost_ratio:.3f}")
 

Output:

Cost ratio rounded to 3 decimal places: 6.579
 

Add a thousand separators to a long series of numbers. 

house_cost = 8930000

print(f"Formatted number: {house_cost:,}")
 

Output:

Formatted number: 8,930,000
 

To format a number as a percentage, use the % symbol.

percentage = 0.25

print(f"Percentage: {percentage:.2%}")
 

Output:

Percentage: 25.00%
 


4. Aligning Text
 

F-strings allow you to align text to the left, right, or center using <, >, and ^ respectively. You can also specify a width for the alignment as shown below. 

id = "Id"
name = "Name"
add = "Address"
formatted_name = f"|{id:<10}|{name:>10}|{add:^10}|"
print(formatted_name)
 

Output:

|Id        |      Name| Address  |
 


5. F-strings with Dictionaries
 

F-strings can work seamlessly with dictionaries. We will access the dictionary keys within the curly braces.

person = {"name": "Abid", "age": 33}

print(f"Name: {person['name']}, Age: {person['age']}")
 

Output:

Name: Abid, Age: 33
 


6. Multiline F-strings
 

F-strings can also be used with triple quotes to create multiline strings. This is useful for formatting long text blocks.

name = "Abid"
age = 33
multiline_string = f"""
Name: {name}
Age: {age}
"""
print(multiline_string)
 

Output:

Name: Abid
Age: 33
 


Final Thoughts
 

To truly appreciate the power and simplicity of f-strings, you need to experience it firsthand. As you start embedding variables, formatting numbers, and aligning text, you will see how f-strings transform your code into something both elegant and efficient. This concise syntax not only streamlines your programming tasks but also elevates your coding style to a professional level.
