import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import mean_squared_error

# Define the function models
def linear_model(x, a, b): return a * x + b
def polynomial_model(x, a, b, c): return a * x**2 + b * x + c
def exponential_model(x, a, b): return a * np.exp(b * x)
def rational_model(x, a, b, c): return (a * x + b) / (x + c)

# Function to fit models and find the best one
def find_best_fit(x, y):
    # Fit the models
    import warnings
    warnings.simplefilter("ignore", category=OptimizeWarning)
    popt_linear, _ = curve_fit(linear_model, x, y)
    popt_poly, _ = curve_fit(polynomial_model, x, y)
    popt_exp, _ = curve_fit(exponential_model, x, y, maxfev=10000)
    popt_rat, _ = curve_fit(rational_model, x, y)

    # Predict values and calculate MSE
    y_pred_linear = linear_model(x, *popt_linear)
    y_pred_poly = polynomial_model(x, *popt_poly)
    y_pred_exp = exponential_model(x, *popt_exp)
    y_pred_rat = rational_model(x, *popt_rat)

    mse_linear = mean_squared_error(y, y_pred_linear)
    mse_poly = mean_squared_error(y, y_pred_poly)
    mse_exp = mean_squared_error(y, y_pred_exp)
    mse_rat = mean_squared_error(y, y_pred_rat)

    # Find the best model
    mse_scores = {
        'linear': mse_linear,
        'polynomial': mse_poly,
        'exponential': mse_exp,
        'rational': mse_rat
    }

    best_model = min(mse_scores, key=mse_scores.get)

    if best_model == 'linear':
        a, b = popt_linear
        return f"def transform_source_to_target1(x): return {a} * x + ({b})"
        # return lambda x: a * x + b

    elif best_model == 'polynomial':
        a, b, c = popt_poly
        return f"def transform_source_to_target1(x): return {a} * x**2 + ({b} * x) + ({c})"
        # return lambda x: a * x**2 + b * x + c

    elif best_model == 'exponential':
        a, b = popt_exp
        return f"def transform_source_to_target1(x):\n  import numpy as np\n  return {a} * np.exp({b} * x)"
        # return lambda x: a * np.exp(b * x)

    elif best_model == 'rational':
        a, b, c = popt_rat
        return f"def transform_source_to_target1(x): return ({a} * x + ({b})) / (x + {c})"
        # return lambda x: (a * x + b) / (x + c)



def parse_number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def get_numeric_function(examples):
    x = np.array([exp[0] for exp in examples])
    y = np.array([parse_number(exp[1]) for exp in examples])
    func = find_best_fit(x, y)
    return func


# # Example usage
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([1.2, 1.9, 3.7, 3.2, 5.1])
#
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 4, 6, 8, 10])
#
# best_fit_function, equation_str = find_best_fit(x, y)
#
# # Print the function value at x = 6
# print(best_fit_function(6))
#
# # Print the equation
# print("Best fit equation:", equation_str)
