"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

def mul(x:float, y:float) -> float:
    """
    Multiplies two numbers.

    Parameters:
    - x, y: The numbers to multiply.

    Returns:
    - The product of x and y.
    """
    return x*y

def id(x:float) -> float:
    """
    Returns the unchanged number

    Parameters:
    - x: The number.

    Returns:
    - input number
    """
    return x

def add(x:float, y:float) -> float:
    """
    Adds two numbers.

    Parameters:
    - x, y: The numbers to add.

    Returns:
    - The sum of x and y.
    """
    return x+y

def neg(x:float) -> float:
    """
    Negates a number

    Parameters:
    - x: The number to negate.

    Returns:
    - Negative of x
    """
    return -x

def lt(x:float, y:float) -> bool:
    """
    Checks if one number is less than another

    Parameters:
    - x, y: The numbers to compare.

    Returns:
    - True if y bigger than x, False otherwise.
    """

    if(x<y):
        return True
    return False

def eq(x:float, y:float) -> bool:
    """
    Check if two numbers are equal.

    Parameters:
    - x, y: The numbers to compare.

    Returns:
    - True if numbers are equal, False otherwise.
    """

    if(x==y):
        return True
    return False

def max(x:float, y:float) -> float:
    """
    Returns the larger of two numbers

    Parameters:
    - x, y: The numbers to compare.

    Returns:
    - The larger value.
    """

    if(x<y):
        return y
    return x

def is_close(x:float, y:float) -> float:
    """
    Check if two numbers are close in value.

    Parameters:
    - a, b: The numbers to compare.
    - rel_tol: Relative tolerance (default 1e-9).
    - abs_tol: Absolute tolerance (default 0.0).

    Returns:
    - True if numbers are close, False otherwise.
    """
    if(abs(x-y)< 1e-2):
        return True
    return False

def sigmoid(x:float)-> float:
    """
    Computes the sigmoid function.

    Parameters:
    - x: A numerical value.

    Returns:
    - The sigmoid value of x.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)



def relu(x:float)-> float:
    """
    Computes the ReLU (Rectified Linear Unit) function.

    Parameters:
    - x: A numerical value.

    Returns:
    - max(0, x)
    """
    if x == 0:
        return 0.0
    return max(0.0, x)

def log(x:float) -> float:
    """
    Computes the natural logarithm (base e).

    Parameters:
    - x: A positive numerical value.

    Returns:
    - The natural logarithm of x.

    Raises:
    - ValueError if x is non-positive.
    """
    if x <= 0:
        raise ValueError("log(x) is undefined for x <= 0.")
    return math.log(x)


def exp(x:float)-> float:
    """
    Computes the exponential function e^x.

    Parameters:
    - x: A numerical value.

    Returns:
    - e^x
    """
    return math.exp(x)

def inv(x:float)->float:
    """
    Computes the reciprocal (inverse) of x.

    Parameters:
    - x: A numerical value.

    Returns:
    - 1/x

    Raises:
    - ValueError if x is zero.
    """
    if x == 0:
        raise ValueError("inv(x) is undefined for x = 0.")
    return 1.0 / x

def log_back(x:float, d_out:float)-> float:
    """
    Computes the derivative of log(x) times a second argument d_out.

    Parameters:
    - x: A positive numerical value.
    - d_out: The gradient from the next layer.

    Returns:
    - d_out / x

    Raises:
    - ValueError if x is non-positive.
    """
    if x <= 0:
        raise ValueError("log_back(x, d_out) is undefined for x <= 0.")
    return d_out / x

def inv_back(x:float, d_out:float)-> float:
    """
    Computes the derivative of the reciprocal function (1/x) times a second argument d_out.

    Parameters:
    - x: A numerical value.
    - d_out: The gradient from the next layer.

    Returns:
    - -d_out / (x^2)

    Raises:
    - ValueError if x is zero.
    """
    if x == 0:
        raise ValueError("inv_back(x, d_out) is undefined for x = 0.")
    return -d_out / (x ** 2)

def relu_back(x:float, d_out:float)-> float:
    """
    Computes the derivative of ReLU function times a second argument d_out.

    Parameters:
    - x: A numerical value.
    - d_out: The gradient from the next layer.

    Returns:
    - d_out if x > 0, otherwise 0.
    """
    return d_out if x > 0 else 0

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order function that applies a given function to each element of an iterable.

    Parameters:
    - fn: A function that takes a float and returns a float.

    Returns:
    - A function that takes an iterable of floats and returns an iterable of floats.
    """
    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return apply

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order function that combines elements from two iterables using a given function.

    Parameters:
    - fn: A function that takes two floats and returns a float.

    Returns:
    - A function that takes two iterables of floats and returns an iterable of floats.
    """
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x,y) for x, y in zip(ls1, ls2)]
    
    return apply

def reduce(fn: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """
    Higher-order function that reduces an iterable to a single value using a given function.

    Parameters:
    - fn: A function that takes two floats and returns a float.
    - start: The initial value to start the reduction.

    Returns:
    - A function that takes an iterable of floats and returns a single float.
    """
    def apply(ls: Iterable[float]) -> float:
        ls = iter(ls) 
        result = next(ls, None)
        
        if result is None:
            return 0

        for x in ls:
            result = fn(result, x) 
        return result

    return apply


negList = map(neg)

addLists = zipWith(add)

sum = reduce(add)

prod = reduce(mul)
