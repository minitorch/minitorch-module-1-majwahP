from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals = list(vals)

    vals[arg] += epsilon/2
    f_plus = f(*vals)

    vals[arg] -= epsilon
    f_minus = f(*vals)

    return (f_plus-f_minus)/epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    ordered_nodes = []  
    visited = set()

    def dfs(node: Variable):
        """Depth-First Search to traverse the graph."""
        if node.unique_id in visited:
            return
        
        visited.add(node.unique_id)

        for inp in node.history.inputs:
            if not inp.is_constant(): 
                dfs(inp)

        ordered_nodes.append(node)  

    dfs(variable)

    return ordered_nodes

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    ordered_nodes: Iterable[Variable] = topological_sort(variable) 
    derivative_dict = {node.unique_id: 0 for node in ordered_nodes} # Takes all uniqe id:s in ordered nodes as keys and assign value 0 for initilization
    derivative_dict[variable.unique_id] = deriv #initilize left most node

    for node in ordered_nodes[::-1]:
        #print(f"Variable: {node}, Unique ID: {node.unique_id}")
        if node.is_leaf():  
            node.accumulate_derivative(derivative_dict[node.unique_id])
            #print("LEAF")
        else:
            variables_partial_deriv = node.chain_rule(derivative_dict[node.unique_id])
            #print("Chain rule...")
            for var, grad in variables_partial_deriv: #var and grad comes from chain rule
                if var.unique_id in derivative_dict:
                    derivative_dict[var.unique_id] += grad
                else:
                    derivative_dict[var.unique_id] = grad
                #print("dict")
                #print(derivative_dict)


           


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
