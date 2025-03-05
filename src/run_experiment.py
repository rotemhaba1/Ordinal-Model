import concurrent.futures


def your_function(param):
    print(f"Running with {param}")



with concurrent.futures.ThreadPoolExecutor() as executor:
    future_X = executor.submit(your_function, 'X')
    future_Y = executor.submit(your_function, 'Y')

    concurrent.futures.wait([future_X, future_Y])

    print(f"X finished: {future_X.done()}")
    print(f"Y finished: {future_Y.done()}")
