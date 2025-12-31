"""
metalq/parameter.py - Symbolic Parameter for Parameterized Circuits

パラメータ化された量子回路で使用するシンボリックパラメータ。
後から具体的な値をバインドして実行できる。

Example:
    theta = Parameter('θ')
    qc = Circuit(1)
    qc.ry(theta, 0)
    
    # 値をバインドして実行
    result = mq.run(qc, params={theta: 0.5})
"""
from __future__ import annotations
from typing import Optional, Union, Set
import uuid
import math


class Parameter:
    """
    Symbolic parameter for parameterized quantum circuits.
    
    Attributes:
        name: Human-readable name
        
    Example:
        theta = Parameter('θ')
        phi = Parameter('φ')
        
        qc = Circuit(2)
        qc.ry(theta, 0)
        qc.rz(phi, 1)
    """
    
    __slots__ = ('_name', '_uuid')
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a parameter.
        
        Args:
            name: Human-readable name. If None, auto-generated.
        """
        self._name = name if name is not None else f"p_{uuid.uuid4().hex[:8]}"
        self._uuid = uuid.uuid4()
    
    @property
    def name(self) -> str:
        """Get parameter name."""
        return self._name
    
    def __repr__(self) -> str:
        return f"Parameter('{self._name}')"
    
    def __str__(self) -> str:
        return self._name
    
    def __hash__(self) -> int:
        return hash(self._uuid)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parameter):
            return self._uuid == other._uuid
        return False
    
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    
    # ==================== 算術演算サポート ====================
    # パラメータ式をサポート（将来の拡張用）
    
    def __add__(self, other: Union[Parameter, float]) -> 'ParameterExpression':
        return ParameterExpression(self, other, '+')
    
    def __radd__(self, other: Union[Parameter, float]) -> 'ParameterExpression':
        return ParameterExpression(other, self, '+')
    
    def __sub__(self, other: Union[Parameter, float]) -> 'ParameterExpression':
        return ParameterExpression(self, other, '-')
    
    def __rsub__(self, other: Union[Parameter, float]) -> 'ParameterExpression':
        return ParameterExpression(other, self, '-')
    
    def __mul__(self, other: Union[Parameter, float]) -> 'ParameterExpression':
        return ParameterExpression(self, other, '*')
    
    def __rmul__(self, other: Union[Parameter, float]) -> 'ParameterExpression':
        return ParameterExpression(other, self, '*')
    
    def __truediv__(self, other: Union[Parameter, float]) -> 'ParameterExpression':
        return ParameterExpression(self, other, '/')
    
    def __neg__(self) -> 'ParameterExpression':
        return ParameterExpression(-1, self, '*')


class ParameterExpression:
    """
    Expression involving parameters (e.g., theta + 0.5, 2 * phi).
    
    パラメータを含む式。評価時に具体的な値に変換される。
    """
    
    __slots__ = ('_left', '_right', '_op', '_parameters')
    
    def __init__(self, left, right, op: str):
        self._left = left
        self._right = right
        self._op = op
        
        # 含まれるパラメータを収集
        self._parameters = set()
        self._collect_parameters(left)
        self._collect_parameters(right)
    
    def _collect_parameters(self, obj):
        """再帰的にパラメータを収集"""
        if isinstance(obj, Parameter):
            self._parameters.add(obj)
        elif isinstance(obj, ParameterExpression):
            self._parameters.update(obj._parameters)
    
    @property
    def parameters(self) -> Set[Parameter]:
        """Get all parameters in this expression."""
        return self._parameters.copy()
    
    def evaluate(self, param_values: dict) -> float:
        """
        Evaluate expression with concrete parameter values.
        
        Args:
            param_values: Dict mapping Parameter to float
            
        Returns:
            Evaluated float value
        """
        left_val = self._eval_operand(self._left, param_values)
        right_val = self._eval_operand(self._right, param_values)
        
        if self._op == '+':
            return left_val + right_val
        elif self._op == '-':
            return left_val - right_val
        elif self._op == '*':
            return left_val * right_val
        elif self._op == '/':
            return left_val / right_val
        else:
            raise ValueError(f"Unknown operator: {self._op}")
    
    def _eval_operand(self, operand, param_values: dict) -> float:
        """Evaluate a single operand."""
        if isinstance(operand, Parameter):
            if operand not in param_values:
                raise ValueError(f"Parameter {operand.name} not bound")
            return param_values[operand]
        elif isinstance(operand, ParameterExpression):
            return operand.evaluate(param_values)
        else:
            return float(operand)

    def grad(self, param: Parameter) -> float:
        """
        Calculate partial derivative with respect to a parameter.
        Returns float (assuming linear coefficients for now).
        Examples:
            (2*theta).grad(theta) -> 2.0
            (theta + phi).grad(theta) -> 1.0
        """
        if param not in self._parameters:
            return 0.0
            
        # Derivative rules
        # d(u+v)/dx = du/dx + dv/dx
        # d(u*v)/dx = du/dx * v + u * dv/dx (Product rule) - Note: v must be const if linear?
        # For QAOA V1, expressions are linear combination: c * p.
        # We assume linearity for MVP simplicity, or simple product.
        
        l_grad = self._grad_operand(self._left, param)
        r_grad = self._grad_operand(self._right, param)
        
        # We need values of operands for product rule?
        # If expression is linear `2*theta`, left=2 (grad=0), right=theta (grad=1).
        # d(2*theta) = 0*theta + 2*1 = 2.
        # So we need CURRENT VALUES if non-linear?
        # But `grad` here returns float.
        # For linear expressions (QAOA), value independent.
        # For non-linear `theta*phi`, derivative depends on value.
        # MetalQ V1 QAOA uses `2*gamma`. Const * Param.
        # So we can evaluate `operand` if it is constant.
        
        # Helper to get value if constant, else raise?
        # We'll try to evaluate operands assuming they are constant relative to `param`?
        # No, simpler: 
        # d(L + R) = dL + dR
        # d(L * R) = dL*R + L*dR.
        
        # For now, let's support only Constant * Parameter or Parameter + Parameter.
        
        val_l = self._get_const_value(self._left)
        val_r = self._get_const_value(self._right)
        
        if self._op == '+':
            return l_grad + r_grad
        elif self._op == '-':
            return l_grad - r_grad
        elif self._op == '*':
            # Product rule
            # If L is const, dL=0. term is L*dR.
            # If R is const, dR=0. term is dL*R.
            # If both variable? We need values.
            # Return error for non-linear?
            term1 = l_grad * val_r if val_r is not None else 0.0 # Approximation if non-const
            term2 = val_l * r_grad if val_l is not None else 0.0
            
            if val_l is None and val_r is None:
                raise NotImplementedError("Non-linear gradient (param*param) not supported without context.")
                
            return term1 + term2
        elif self._op == '/':
            # Quotient rule d(u/v) = (du*v - u*dv)/v^2
            if val_r is None:
                raise NotImplementedError("Division by parameter not supported")
            return (l_grad * val_r - (val_l if val_l else 0) * r_grad) / (val_r * val_r)
            
        return 0.0

    def _grad_operand(self, operand, param):
        if isinstance(operand, Parameter):
            return 1.0 if operand == param else 0.0
        elif isinstance(operand, ParameterExpression):
            return operand.grad(param)
        else:
            return 0.0 # Constant
            
    def _get_const_value(self, operand) -> Optional[float]:
        """Return value if operand is constant, else None."""
        if isinstance(operand, (int, float)):
            return float(operand)
        if isinstance(operand, Parameter):
            return None
        if isinstance(operand, ParameterExpression):
            # Try to resolve if sub-expression is constant
            try:
                if len(operand.parameters) == 0:
                     return operand.evaluate({})
            except:
                pass
            return None
        return None

    
    def __repr__(self) -> str:
        return f"({self._left} {self._op} {self._right})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    # 式の連鎖をサポート
    def __add__(self, other) -> 'ParameterExpression':
        return ParameterExpression(self, other, '+')
    
    def __radd__(self, other) -> 'ParameterExpression':
        return ParameterExpression(other, self, '+')
    
    def __sub__(self, other) -> 'ParameterExpression':
        return ParameterExpression(self, other, '-')
    
    def __mul__(self, other) -> 'ParameterExpression':
        return ParameterExpression(self, other, '*')
    
    def __rmul__(self, other) -> 'ParameterExpression':
        return ParameterExpression(other, self, '*')
    
    def __truediv__(self, other) -> 'ParameterExpression':
        return ParameterExpression(self, other, '/')
    
    def __neg__(self) -> 'ParameterExpression':
        return ParameterExpression(-1, self, '*')


def is_parameterized(value) -> bool:
    """Check if a value contains unbound parameters."""
    return isinstance(value, (Parameter, ParameterExpression))


def get_parameters(value) -> Set[Parameter]:
    """Get all parameters in a value."""
    if isinstance(value, Parameter):
        return {value}
    elif isinstance(value, ParameterExpression):
        return value.parameters
    else:
        return set()


def evaluate_parameter(value, param_values: dict) -> float:
    """Evaluate a potentially parameterized value."""
    if isinstance(value, Parameter):
        if value not in param_values:
            raise ValueError(f"Parameter {value.name} not bound")
        return param_values[value]
    elif isinstance(value, ParameterExpression):
        return value.evaluate(param_values)
    else:
        return float(value)
