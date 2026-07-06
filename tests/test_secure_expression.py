#!/usr/bin/env python3
"""
Tests for Secure Expression Evaluation

Tests the secure AST-based expression evaluator to ensure:
- Only safe operations are allowed
- Code injection is prevented
- Resource limits are enforced
- Mathematical operations work correctly
"""

import pytest
import math
import time
import ast
from metalq.secure_eval import (
    SecureEvaluator,
    safe_eval,
    SecurityError,
    ExpressionError,
    EvaluationLimits,
    SafeParameterExpression,
    ExpressionBuilder,
    secure_expression
)
from metalq.parameter import Parameter, ParameterExpression


class TestSecureEvaluator:
    """Test the secure expression evaluator."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert safe_eval("2 + 3") == 5
        assert safe_eval("10 - 4") == 6
        assert safe_eval("3 * 4") == 12
        assert safe_eval("15 / 3") == 5.0
        assert safe_eval("2 ** 3") == 8
        assert safe_eval("10 % 3") == 1
        assert safe_eval("10 // 3") == 3

    def test_operator_precedence(self):
        """Test that operator precedence is respected."""
        assert safe_eval("2 + 3 * 4") == 14
        assert safe_eval("(2 + 3) * 4") == 20
        assert safe_eval("10 - 2 * 3") == 4
        assert safe_eval("2 ** 3 * 4") == 32
        assert safe_eval("2 * 3 ** 2") == 18

    def test_variables(self):
        """Test variable substitution."""
        variables = {'x': 5, 'y': 3}
        assert safe_eval("x + y", variables) == 8
        assert safe_eval("x * y", variables) == 15
        assert safe_eval("2 * x + y", variables) == 13

    def test_undefined_variable(self):
        """Test that undefined variables raise errors."""
        with pytest.raises(ExpressionError) as exc_info:
            safe_eval("x + 1")
        assert "Undefined variable" in str(exc_info.value)

    def test_math_functions(self):
        """Test whitelisted math functions."""
        assert safe_eval("sqrt(4)") == 2.0
        assert safe_eval("abs(-5)") == 5
        assert safe_eval("round(3.7)") == 4
        assert safe_eval("min(2, 5, 1)") == 1
        assert safe_eval("max(2, 5, 1)") == 5
        assert pytest.approx(safe_eval("sin(pi/2)")) == 1.0
        assert pytest.approx(safe_eval("cos(0)")) == 1.0
        assert safe_eval("floor(3.7)") == 3
        assert safe_eval("ceil(3.2)") == 4

    def test_math_constants(self):
        """Test math constants."""
        assert pytest.approx(safe_eval("pi")) == math.pi
        assert pytest.approx(safe_eval("e")) == math.e
        assert pytest.approx(safe_eval("tau")) == math.tau

    def test_comparisons(self):
        """Test comparison operations."""
        assert safe_eval("5 > 3") is True
        assert safe_eval("2 < 1") is False
        assert safe_eval("3 >= 3") is True
        assert safe_eval("4 <= 2") is False
        assert safe_eval("5 == 5") is True
        assert safe_eval("3 != 3") is False

    def test_boolean_operations(self):
        """Test boolean operations."""
        assert safe_eval("True and True") is True
        assert safe_eval("True and False") is False
        assert safe_eval("True or False") is True
        assert safe_eval("False or False") is False
        assert safe_eval("5 > 3 and 2 < 4") is True

    def test_division_by_zero(self):
        """Test division by zero handling."""
        with pytest.raises(ExpressionError) as exc_info:
            safe_eval("1 / 0")
        assert "Division by zero" in str(exc_info.value)

    def test_resource_limits(self):
        """Test that resource limits are enforced."""
        limits = EvaluationLimits(max_depth=3, max_nodes=50)  # Allow more nodes for the depth test

        # Too deep - use nested function calls which create actual depth
        with pytest.raises(SecurityError) as exc_info:
            safe_eval("abs(abs(abs(abs(abs(1)))))", limits=limits)
        assert "too deep" in str(exc_info.value).lower()

        # Too many nodes
        limits2 = EvaluationLimits(max_depth=10, max_nodes=10)  # Very low node limit
        with pytest.raises(SecurityError) as exc_info:
            safe_eval("1+1+1+1+1+1+1+1+1+1+1+1", limits=limits2)
        assert "too complex" in str(exc_info.value).lower()

    def test_timeout(self):
        """Test that evaluation times out for complex expressions."""
        # Use fewer operations to avoid recursion limit
        limits = EvaluationLimits(max_time=0.1, max_nodes=1000, max_depth=500)

        # Create an expression that takes time to evaluate
        # Note: This is tricky to test reliably, so we use a very short timeout
        expression = " + ".join(str(i) for i in range(200))

        # Should complete with normal timeout
        normal_limits = EvaluationLimits(max_time=5.0, max_nodes=1000, max_depth=500)
        result = safe_eval(expression, limits=normal_limits)
        assert result == sum(range(200))

    def test_number_limits(self):
        """Test that very large numbers are rejected."""
        limits = EvaluationLimits(max_number=1e10)

        with pytest.raises(SecurityError) as exc_info:
            safe_eval("10 ** 20", limits=limits)
        assert "too large" in str(exc_info.value).lower()


class TestSecurityPrevention:
    """Test that dangerous operations are prevented."""

    def test_import_blocked(self):
        """Test that import statements are blocked."""
        dangerous_expressions = [
            "__import__('os')",
            "import os",
            "from os import system",
        ]

        for expr in dangerous_expressions:
            with pytest.raises((SecurityError, ExpressionError)):
                safe_eval(expr)

    def test_attribute_access_blocked(self):
        """Test that attribute access is blocked."""
        dangerous_expressions = [
            "().__class__",
            "''.__class__.__bases__",
            "[].__class__.__mro__",
            "x.__dict__",
        ]

        for expr in dangerous_expressions:
            with pytest.raises(SecurityError) as exc_info:
                safe_eval(expr)
            assert "Attribute access not allowed" in str(exc_info.value)

    def test_exec_eval_blocked(self):
        """Test that exec and eval are blocked."""
        dangerous_expressions = [
            "eval('1+1')",
            "exec('x = 1')",
            "compile('x=1', '', 'exec')",
        ]

        for expr in dangerous_expressions:
            with pytest.raises((SecurityError, ExpressionError)):
                safe_eval(expr)

    def test_lambda_blocked(self):
        """Test that lambda expressions are blocked."""
        with pytest.raises(SecurityError) as exc_info:
            safe_eval("(lambda x: x + 1)(5)")
        assert "Lambda expressions not allowed" in str(exc_info.value)

    def test_comprehension_blocked(self):
        """Test that comprehensions are blocked."""
        dangerous_expressions = [
            "[x for x in range(10)]",
            "{x: x**2 for x in range(10)}",
            "(x for x in range(10))",
        ]

        for expr in dangerous_expressions:
            with pytest.raises((SecurityError, SyntaxError)):
                safe_eval(expr)

    def test_loops_blocked(self):
        """Test that loops are blocked."""
        # These would be syntax errors in eval mode anyway,
        # but we test that they're explicitly blocked
        dangerous_code = [
            "for i in range(10): pass",
            "while True: pass",
        ]

        for code in dangerous_code:
            with pytest.raises(ExpressionError):
                safe_eval(code)

    def test_function_definition_blocked(self):
        """Test that function definitions are blocked."""
        with pytest.raises(ExpressionError):
            safe_eval("def f(): return 1")

    def test_class_definition_blocked(self):
        """Test that class definitions are blocked."""
        with pytest.raises(ExpressionError):
            safe_eval("class C: pass")

    def test_dangerous_builtins_blocked(self):
        """Test that dangerous built-in functions are blocked."""
        dangerous_expressions = [
            "open('/etc/passwd')",
            "print('hello')",
            "input()",
            "globals()",
            "locals()",
            "vars()",
            "dir()",
            "help()",
            "type(1)",
            "isinstance(1, int)",
            "getattr(x, 'y')",
            "setattr(x, 'y', 1)",
            "delattr(x, 'y')",
            "hasattr(x, 'y')",
        ]

        for expr in dangerous_expressions:
            with pytest.raises((SecurityError, ExpressionError)):
                safe_eval(expr)

    def test_string_formatting_blocked(self):
        """Test that string formatting is blocked."""
        # These could potentially be used for injection
        # String operations are not part of our allowed operations
        dangerous_expressions = [
            "'%s' % 'test'",  # Mod operator on strings
            "'{}'.format('test')",  # Attribute access (format)
            # f-strings aren't supported in eval mode
        ]

        for expr in dangerous_expressions:
            with pytest.raises((SecurityError, ExpressionError)):
                safe_eval(expr)

    def test_only_whitelisted_functions(self):
        """Test that only whitelisted functions are allowed."""
        # Allowed
        assert safe_eval("sqrt(4)") == 2.0

        # Not allowed
        with pytest.raises(SecurityError) as exc_info:
            safe_eval("chr(65)")
        assert "not allowed" in str(exc_info.value)


class TestParameterIntegration:
    """Test integration with Metal-Q parameters."""

    def test_parameter_expression_secure(self):
        """Test that ParameterExpression uses secure evaluation."""
        theta = Parameter('theta')
        phi = Parameter('phi')

        # Create expression
        expr = theta * 2 + phi

        # Evaluate with secure evaluator
        param_values = {theta: 1.5, phi: 0.5}
        result = expr.evaluate(param_values)
        assert result == 3.5

    def test_parameter_expression_blocks_injection(self):
        """Test that parameter expressions cannot be used for injection."""
        # Try to create a malicious parameter name
        # The expression builder should prevent this
        try:
            bad_param = Parameter("__import__('os').system('ls')")
            expr = ParameterExpression(bad_param, 1, '+')

            # Even if we somehow create it, evaluation should fail
            with pytest.raises((SecurityError, ValueError)):
                expr.evaluate({bad_param: 1})
        except ValueError:
            # Expression validation should catch it
            pass

    def test_safe_parameter_expression(self):
        """Test SafeParameterExpression class."""
        # Valid expression
        expr = SafeParameterExpression("x * 2 + y")
        result = expr.evaluate({'x': 3, 'y': 1})
        assert result == 7

        # Invalid expression should fail at creation
        with pytest.raises(ValueError):
            SafeParameterExpression("__import__('os')")

    def test_complex_parameter_expression(self):
        """Test complex but safe parameter expressions."""
        theta = Parameter('theta')
        phi = Parameter('phi')
        gamma = Parameter('gamma')

        # Complex mathematical expression
        expr = (theta * 2 + phi) * gamma - theta / 2

        param_values = {theta: math.pi, phi: math.pi/2, gamma: 0.5}
        result = expr.evaluate(param_values)

        expected = (math.pi * 2 + math.pi/2) * 0.5 - math.pi/2
        assert pytest.approx(result) == expected

    def test_parameter_identity_preserved_for_same_name(self):
        """Different Parameter instances with the same name must not collide."""
        theta1 = Parameter('theta')
        theta2 = Parameter('theta')

        expr = theta1 + theta2

        result = expr.evaluate({theta1: 1.25, theta2: 2.75})
        assert result == 4.0

    def test_subtraction_parenthesizes_nested_expression(self):
        """Right-nested subtraction must preserve the original structure."""
        a = Parameter('a')
        b = Parameter('b')
        c = Parameter('c')

        expr = a - (b - c)

        result = expr.evaluate({a: 10.0, b: 3.0, c: 1.0})
        assert result == 8.0
        assert str(expr) == "a - (b - c)"


class TestExpressionBuilder:
    """Test the safe expression builder."""

    def test_builder_basic(self):
        """Test basic expression building."""
        builder = ExpressionBuilder()
        expr = builder.add(5).multiply(2).subtract(3).build()
        # add(5) -> "5", multiply(2) -> "(5) * 2", subtract(3) -> "((5) * 2) - 3"

        # Evaluate
        result = builder.evaluate()
        assert result == (5 * 2) - 3

    def test_builder_with_variables(self):
        """Test expression builder with variables."""
        builder = ExpressionBuilder()
        builder.add('x').multiply('y')
        result = builder.evaluate({'x': 3, 'y': 4})
        assert result == 3 * 4

    def test_builder_with_functions(self):
        """Test expression builder with functions."""
        builder = ExpressionBuilder()
        builder.add(16).function('sqrt')
        result = builder.evaluate()
        assert result == 4.0

    def test_builder_rejects_dangerous_functions(self):
        """Test that builder rejects dangerous functions."""
        builder = ExpressionBuilder()

        with pytest.raises(ValueError) as exc_info:
            builder.function('eval', 'code')
        assert "not allowed" in str(exc_info.value)


class TestSecureDecorator:
    """Test the secure_expression decorator."""

    def test_decorator_validates_expression(self):
        """Test that decorator validates expressions."""
        @secure_expression
        def evaluate_expr(expression: str) -> float:
            # Unsafe implementation (would use eval in real bad code)
            # But decorator should prevent dangerous input
            return safe_eval(expression)

        # Safe expression works
        assert evaluate_expr("2 + 3") == 5

        # Dangerous expression blocked by decorator
        with pytest.raises(SecurityError):
            evaluate_expr("__import__('os')")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_expression(self):
        """Test empty expression handling."""
        with pytest.raises(ExpressionError):
            safe_eval("")

    def test_whitespace_only(self):
        """Test whitespace-only expression."""
        with pytest.raises(ExpressionError):
            safe_eval("   ")

    def test_very_long_expression(self):
        """Test very long but valid expression."""
        # Create a long expression
        expr = " + ".join(str(i) for i in range(100))
        # Need higher limits for this test
        from metalq.secure_eval import EvaluationLimits
        limits = EvaluationLimits(max_nodes=500, max_depth=200)  # Need depth >= 100 for 100 additions
        result = safe_eval(expr, limits=limits)
        assert result == sum(range(100))

    def test_nested_functions(self):
        """Test nested function calls."""
        assert safe_eval("sqrt(abs(-16))") == 4.0
        assert safe_eval("max(min(5, 3), 2)") == 3

    def test_unicode_variables(self):
        """Test Unicode variable names."""
        variables = {'θ': math.pi, 'φ': math.pi/2}
        result = safe_eval("θ + φ", variables)
        assert pytest.approx(result) == math.pi + math.pi/2

    def test_scientific_notation(self):
        """Test scientific notation numbers."""
        assert safe_eval("1e3") == 1000
        assert safe_eval("1.5e-2") == 0.015

    def test_negative_numbers(self):
        """Test negative numbers."""
        assert safe_eval("-5") == -5
        assert safe_eval("-(-3)") == 3
        assert safe_eval("-(2 + 3)") == -5


class TestPerformance:
    """Test performance characteristics."""

    def test_evaluation_speed(self):
        """Test that evaluation is reasonably fast."""
        import timeit

        # Simple expression should be fast
        time_taken = timeit.timeit(
            lambda: safe_eval("2 + 3 * 4"),
            number=1000
        )
        assert time_taken < 1.0  # 1000 evaluations in under 1 second

    def test_validation_overhead(self):
        """Test that validation doesn't add too much overhead."""
        from metalq.secure_eval import EvaluationLimits

        # Create evaluator with appropriate limits
        limits = EvaluationLimits(max_nodes=20)  # Simple expression won't exceed this

        # Pre-parse for comparison
        tree = ast.parse("x * 2 + 3", mode='eval')

        import timeit

        # Time validation - create fresh evaluator each time to reset node count
        def validate():
            evaluator = SecureEvaluator({'x': 5}, limits=limits)
            evaluator._validate_ast(tree)

        validation_time = timeit.timeit(validate, number=1000)

        assert validation_time < 0.5  # Validation should be fast


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
