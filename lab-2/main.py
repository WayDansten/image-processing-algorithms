import numpy as np

def calculate_simple_monte_carlo_integral(a, b, function):
    sample = np.random.Generator().uniform(a, b)
    y = function(sample)
    interval = b - a
    return y * interval

def calculate_stratified_monte_carlo_integral(a, b, function, step):
    partition = np.arange(a, b, step)
    samples = np.random.Generator().uniform(partition[:-1], partition[1:])
    y = function(samples)
    return np.mean(y) * (b - a)

def calculate_importance_sample_monte_carlo_integral(a, b, function, weight_function):
    pass

def calculate_multi_importance_sample_monte_carlo_integral():
    pass

def calculate_russian_roulette_monte_carlo_integral():
    pass


square = lambda x: x ** 2
square_integral = lambda x: x ** 3 / 3

a, b = 2, 5
results = {}

results['analytical'] = square_integral(b) - square_integral(a)
results['simple_monte_carlo'] = calculate_simple_monte_carlo_integral(a, b, square)
results['stratified_monte_carlo'] = {
    1: calculate_stratified_monte_carlo_integral(a, b, square, 1),
    0.5: calculate_stratified_monte_carlo_integral(a, b, square, 0.5)
}
