# function decorator that calls the function twice
def repeat_decorator(fn):
    def decorated_fn():
        fn()
        fn()
    # returns a function
    return decorated_fn

# using the decorator on hello_world function
@repeat_decorator
def hello_world():
    print ("Hello world!")

# call the function
hello_world()
