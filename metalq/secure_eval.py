"""
Secure Expression Evaluator for Metal-Q

Provides sandboxed expression evaluation using AST parsing with strict
whitelisting of allowed operations. Prevents code injection and arbitrary
code execution.

Security features:
- AST-based parsing (no eval/exec)
- Whitelist of allowed operations
- Resource limits (complexity, depth, size)
- Type validation
- Comprehensive error handling
"""

import ast
import math
import operator
import logging
from typing import Any, Dict, List, Optional, Set, Union, Callable
from functools import wraps
import time
import threading
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


class ExpressionError(Exception):
    """Raised when an expression cannot be evaluated."""
    pass


class NodeType(Enum):
    """Allowed AST node types."""
    # Literals and identifiers
    CONSTANT = "Constant"
    NAME = "Name"

    # Binary operations
    ADD = "Add"
    SUB = "Sub"
    MULT = "Mult"
    DIV = "Div"
    FLOOR_DIV = "FloorDiv"
    MOD = "Mod"
    POW = "Pow"

    # Unary operations
    UNARY_PLUS = "UAdd"
    UNARY_MINUS = "USub"

    # Comparison operations (for future extension)
    EQ = "Eq"
    NOT_EQ = "NotEq"
    LT = "Lt"
    LTE = "LtE"
    GT = "Gt"
    GTE = "GtE"

    # Boolean operations
    AND = "And"
    OR = "Or"
    NOT = "Not"

    # Math functions
    CALL = "Call"

    # Container types (restricted)
    LIST = "List"
    TUPLE = "Tuple"
    SUBSCRIPT = "Subscript"


@dataclass
class EvaluationLimits:
    """Resource limits for expression evaluation."""
    max_depth: int = 10  # Maximum AST depth
    max_nodes: int = 100  # Maximum number of nodes
    max_time: float = 1.0  # Maximum evaluation time (seconds)
    max_iterations: int = 1000  # Maximum loop iterations (for comprehensions)
    max_string_length: int = 1000  # Maximum string length
    max_number: float = 1e308  # Maximum number value (less than float max)


class SecureEvaluator:
    """
    Secure expression evaluator using AST parsing.

    This evaluator only allows safe mathematical operations and
    prevents any form of code injection or arbitrary execution.
    """

    # Whitelisted operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Whitelisted comparison operators
    COMPARISONS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    # Whitelisted boolean operators
    BOOL_OPS = {
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
        ast.Not: operator.not_,
    }

    # Whitelisted math functions (safe subset)
    FUNCTIONS = {
        # Basic math
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,

        # Math module functions (safe ones only)
        'sqrt': math.sqrt,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'degrees': math.degrees,
        'radians': math.radians,
        'floor': math.floor,
        'ceil': math.ceil,
        'factorial': math.factorial,

        # Constants
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
    }

    def __init__(self,
                 variables: Optional[Dict[str, Any]] = None,
                 limits: Optional[EvaluationLimits] = None,
                 allow_functions: bool = True):
        """
        Initialize secure evaluator.

        Args:
            variables: Dictionary of allowed variable names and values
            limits: Resource limits for evaluation
            allow_functions: Whether to allow function calls
        """
        self.variables = variables or {}
        self.limits = limits or EvaluationLimits()
        self.allow_functions = allow_functions

        # Tracking for resource limits
        self.node_count = 0
        self.start_time = None
        self.call_depth = 0

    def evaluate(self, expression: str) -> Any:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Expression string to evaluate

        Returns:
            Result of expression evaluation

        Raises:
            SecurityError: If expression contains unsafe operations
            ExpressionError: If expression cannot be evaluated
        """
        # Reset tracking
        self.node_count = 0
        self.call_depth = 0
        self.start_time = time.time()

        # Parse expression into AST
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ExpressionError(f"Invalid syntax: {e}")

        # Validate AST structure
        self._validate_ast(tree)

        # Evaluate with timeout
        return self._evaluate_with_timeout(tree.body)

    def _evaluate_with_timeout(self, node: ast.AST) -> Any:
        """Evaluate node with timeout protection."""
        result = [None]
        exception = [None]

        def evaluate_thread():
            try:
                result[0] = self._evaluate_node(node)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=evaluate_thread, daemon=True)
        thread.start()
        thread.join(self.limits.max_time)

        if thread.is_alive():
            raise SecurityError(
                f"Expression evaluation timed out after {self.limits.max_time}s"
            )

        if exception[0]:
            raise exception[0]

        return result[0]

    def _validate_ast(self, tree: ast.AST):
        """
        Validate AST for security issues.

        Raises:
            SecurityError: If dangerous constructs are found
        """
        for node in ast.walk(tree):
            # Count nodes
            self.node_count += 1
            if self.node_count > self.limits.max_nodes:
                raise SecurityError(
                    f"Expression too complex: exceeds {self.limits.max_nodes} nodes"
                )

            # Check for dangerous node types
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityError("Import statements not allowed")

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                raise SecurityError("Function/class definitions not allowed")

            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                raise SecurityError("Loops not allowed")

            if isinstance(node, ast.comprehension):
                raise SecurityError("Comprehensions not allowed")

            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise SecurityError("Global/nonlocal not allowed")

            if isinstance(node, ast.Attribute):
                # Attribute access is dangerous - could access __class__, __import__, etc.
                raise SecurityError("Attribute access not allowed")

            if isinstance(node, ast.Lambda):
                raise SecurityError("Lambda expressions not allowed")

            if isinstance(node, ast.Yield):
                raise SecurityError("Yield expressions not allowed")

            # Check function calls
            if isinstance(node, ast.Call):
                if not self.allow_functions:
                    raise SecurityError("Function calls not allowed")

                # Check for lambda in call
                if isinstance(node.func, ast.Lambda):
                    raise SecurityError("Lambda expressions not allowed")

                # Only allow whitelisted functions
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.FUNCTIONS:
                        raise SecurityError(f"Function '{func_name}' not allowed")
                else:
                    raise SecurityError("Complex function calls not allowed")

    def _evaluate_node(self, node: ast.AST) -> Any:
        """
        Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate

        Returns:
            Result of node evaluation
        """
        # Check time limit
        if time.time() - self.start_time > self.limits.max_time:
            raise SecurityError("Expression evaluation timed out")

        # Track recursion depth
        self.call_depth += 1
        if self.call_depth > self.limits.max_depth:
            raise SecurityError(
                f"Expression too deep: exceeds depth {self.limits.max_depth}"
            )

        try:
            result = self._eval_node(node)
        finally:
            self.call_depth -= 1

        # Validate result
        if isinstance(result, (int, float)):
            if abs(result) > self.limits.max_number:
                raise SecurityError(f"Number too large: {result}")

        return result

    def _eval_node(self, node: ast.AST) -> Any:
        """Internal node evaluation."""
        # Literals
        if isinstance(node, ast.Constant):
            # Block string constants for mathematical expressions
            if isinstance(node.value, str):
                raise SecurityError("String literals not allowed in mathematical expressions")
            return node.value

        # Python 3.7 compatibility
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Str):
            # Block string literals
            raise SecurityError("String literals not allowed in mathematical expressions")
        if isinstance(node, ast.NameConstant):
            return node.value

        # Variables
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.variables:
                return self.variables[name]
            elif name in self.FUNCTIONS and not self.allow_functions:
                raise SecurityError(f"Function '{name}' not allowed")
            elif name in self.FUNCTIONS:
                return self.FUNCTIONS[name]
            else:
                raise ExpressionError(f"Undefined variable: {name}")

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            op = type(node.op)

            # Block string formatting operations
            if isinstance(left, str) or isinstance(right, str):
                if op in [ast.Mod, ast.Mult]:  # % and * on strings
                    raise SecurityError("String operations not allowed")

            if op in self.OPERATORS:
                try:
                    return self.OPERATORS[op](left, right)
                except ZeroDivisionError:
                    raise ExpressionError("Division by zero")
                except Exception as e:
                    raise ExpressionError(f"Operation failed: {e}")
            else:
                raise SecurityError(f"Operation {op.__name__} not allowed")

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand)
            op = type(node.op)

            if op in self.OPERATORS:
                return self.OPERATORS[op](operand)
            else:
                raise SecurityError(f"Operation {op.__name__} not allowed")

        # Comparisons
        if isinstance(node, ast.Compare):
            left = self._evaluate_node(node.left)

            for op, comparator in zip(node.ops, node.comparators):
                right = self._evaluate_node(comparator)
                op_type = type(op)

                if op_type in self.COMPARISONS:
                    result = self.COMPARISONS[op_type](left, right)
                    if not result:
                        return False
                    left = right
                else:
                    raise SecurityError(f"Comparison {op_type.__name__} not allowed")

            return True

        # Boolean operations
        if isinstance(node, ast.BoolOp):
            op = type(node.op)

            if op == ast.And:
                for value in node.values:
                    result = self._evaluate_node(value)
                    if not result:
                        return False
                return True

            elif op == ast.Or:
                for value in node.values:
                    result = self._evaluate_node(value)
                    if result:
                        return True
                return False

            else:
                raise SecurityError(f"Boolean operation {op.__name__} not allowed")

        # Function calls
        if isinstance(node, ast.Call):
            if not self.allow_functions:
                raise SecurityError("Function calls not allowed")

            # Get function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id

                if func_name not in self.FUNCTIONS:
                    raise SecurityError(f"Function '{func_name}' not allowed")

                func = self.FUNCTIONS[func_name]

                # Evaluate arguments
                args = [self._evaluate_node(arg) for arg in node.args]

                # No keyword arguments allowed for simplicity
                if node.keywords:
                    raise SecurityError("Keyword arguments not allowed")

                # Call function with error handling
                try:
                    return func(*args)
                except Exception as e:
                    raise ExpressionError(f"Function {func_name} failed: {e}")
            else:
                raise SecurityError("Complex function calls not allowed")

        # Lists (restricted)
        if isinstance(node, ast.List):
            if len(node.elts) > 100:
                raise SecurityError("List too large")
            return [self._evaluate_node(elt) for elt in node.elts]

        # Tuples (restricted)
        if isinstance(node, ast.Tuple):
            if len(node.elts) > 100:
                raise SecurityError("Tuple too large")
            return tuple(self._evaluate_node(elt) for elt in node.elts)

        # If we get here, the node type is not supported
        raise SecurityError(f"Node type {type(node).__name__} not allowed")


def safe_eval(expression: str,
              variables: Optional[Dict[str, Any]] = None,
              limits: Optional[EvaluationLimits] = None) -> Any:
    """
    Convenience function for safe expression evaluation.

    Args:
        expression: Mathematical expression to evaluate
        variables: Dictionary of variable values
        limits: Resource limits

    Returns:
        Result of expression evaluation

    Raises:
        SecurityError: If expression is unsafe
        ExpressionError: If expression cannot be evaluated
    """
    evaluator = SecureEvaluator(variables, limits)
    return evaluator.evaluate(expression)


class SafeParameterExpression:
    """
    Safe replacement for ParameterExpression using secure evaluation.
    """

    def __init__(self, expression: str):
        """
        Initialize safe parameter expression.

        Args:
            expression: Mathematical expression string
        """
        self.expression = expression
        self._validate_expression()

    def _validate_expression(self):
        """Validate expression at creation time."""
        try:
            # Parse and validate without evaluating
            tree = ast.parse(self.expression, mode='eval')
            evaluator = SecureEvaluator()
            evaluator._validate_ast(tree)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    def evaluate(self, params: Dict[str, float]) -> float:
        """
        Evaluate expression with given parameters.

        Args:
            params: Dictionary mapping parameter names to values

        Returns:
            Evaluated result
        """
        evaluator = SecureEvaluator(variables=params)
        return evaluator.evaluate(self.expression)

    def __repr__(self):
        return f"SafeParameterExpression('{self.expression}')"


# Decorator for adding security checks to existing functions
def secure_expression(func: Callable) -> Callable:
    """
    Decorator to add security checks to expression evaluation functions.
    """
    @wraps(func)
    def wrapper(expression: str, *args, **kwargs):
        # Validate expression first
        evaluator = SecureEvaluator()
        try:
            tree = ast.parse(expression, mode='eval')
            evaluator._validate_ast(tree)
        except Exception as e:
            raise SecurityError(f"Unsafe expression: {e}")

        # Call original function
        return func(expression, *args, **kwargs)

    return wrapper


# Example of safe mathematical expression builder
class ExpressionBuilder:
    """
    Builder for creating safe mathematical expressions programmatically.
    """

    def __init__(self):
        self._expression = ""

    def add(self, value: Union[str, float]) -> 'ExpressionBuilder':
        """Add a value."""
        if self._expression:
            self._expression = f"({self._expression}) + {value}"
        else:
            self._expression = str(value)
        return self

    def subtract(self, value: Union[str, float]) -> 'ExpressionBuilder':
        """Subtract a value."""
        if self._expression:
            self._expression = f"({self._expression}) - {value}"
        else:
            self._expression = f"-{value}"
        return self

    def multiply(self, value: Union[str, float]) -> 'ExpressionBuilder':
        """Multiply by a value."""
        if self._expression:
            self._expression = f"({self._expression}) * {value}"
        else:
            self._expression = str(value)
        return self

    def divide(self, value: Union[str, float]) -> 'ExpressionBuilder':
        """Divide by a value."""
        if self._expression:
            self._expression = f"({self._expression}) / {value}"
        else:
            self._expression = f"1 / {value}"
        return self

    def power(self, value: Union[str, float]) -> 'ExpressionBuilder':
        """Raise to a power."""
        if self._expression:
            self._expression = f"({self._expression}) ** {value}"
        else:
            self._expression = f"1 ** {value}"
        return self

    def function(self, func_name: str, *args) -> 'ExpressionBuilder':
        """Apply a function."""
        if func_name not in SecureEvaluator.FUNCTIONS:
            raise ValueError(f"Function '{func_name}' not allowed")

        args_str = ", ".join(str(arg) for arg in args)
        if self._expression:
            self._expression = f"{func_name}({self._expression}, {args_str})"
        else:
            self._expression = f"{func_name}({args_str})"
        return self

    def build(self) -> str:
        """Build the expression string."""
        return self._expression

    def evaluate(self, variables: Optional[Dict[str, float]] = None) -> float:
        """Build and evaluate the expression."""
        return safe_eval(self._expression, variables)