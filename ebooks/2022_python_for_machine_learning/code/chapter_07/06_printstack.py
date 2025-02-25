import traceback
import random

def print_tb_with_local():
    """Print stack trace with local variables. This does not need to be in
    exception. Print is using the system's print() function to stderr.
    """
    import traceback, sys
    tb = sys.exc_info()[2]
    stack = []
    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next
    traceback.print_exc()
    print("Locals by frame, most recent call first", file=sys.stderr)
    for frame in stack:
        # print the function name and line number of the script
        print("Frame {0} in {1} at line {2}".format(
            frame.f_code.co_name,
            frame.f_code.co_filename,
            frame.f_lineno), file=sys.stderr)
        # print each variable defined inside each function
        for key, value in frame.f_locals.items():
            print(f"\t{key} = ", file=sys.stderr)
            try:
                if '__repr__' in dir(value):
                    print(value.__repr__(), file=sys.stderr)
                elif '__str__' in dir(value):
                    print(value.__str__(), file=sys.stderr)
                else:
                    print(value, file=sys.stderr)
            except:
                print("", file=sys.stderr)

def compute():
    n = random.randint(0, 10)
    m = random.randint(0, 10)
    return n/m

def compute_many(n_times):
    try:
        for _ in range(n_times):
            x = compute()
        print(f"Completed {n_times} times")
    except:
        print("Something wrong")
        print_tb_with_local()

compute_many(100)

