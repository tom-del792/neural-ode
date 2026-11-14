"""Translation of SBML formulas to jnp expressions.

Values of the operators, collected dependencies, local names, unsupported constructs.
"""

import math

import jax
import jax.numpy as jnp
import libsbml
import pytest
from jax.scipy import special

from sbml2jax.jaxmath import UnsupportedError, translate

VALUES = {"x": 2.0, "y": 3.0, "z": -0.5}


def evaluate(formula: str, t: float = 1.5) -> float:
    ast = libsbml.parseL3Formula(formula)
    assert ast is not None, libsbml.getLastParseL3Error()
    expr = translate(ast, lambda sid: sid)
    return float(eval(expr.code, {"jnp": jnp, "special": special, "t": t, **VALUES}))


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("x + y * 2", 8.0),
        ("-x", -2.0),
        ("x - y - 1", -2.0),
        ("x / y", 2 / 3),
        ("x^y", 8.0),
        ("pow(x, y)", 8.0),
        ("sqrt(4)", 2.0),
        ("root(3, 8)", 2.0),
        ("log(100)", 2.0),
        ("log(2, 8)", 3.0),
        ("ln(exp(x))", 2.0),
        ("abs(z)", 0.5),
        ("floor(z)", -1.0),
        ("ceil(z)", 0.0),
        ("factorial(4)", 24.0),
        ("sec(z)", 1 / math.cos(-0.5)),
        ("arccot(x)", math.atan(0.5)),
        ("min(x, y, z)", -0.5),
        ("max(x, y, z)", 3.0),
        ("rem(7, 3)", 1.0),
        ("quotient(-7, 2)", -3.0),
        ("piecewise(1, x > y, 2, x < y, 3)", 2.0),
        ("piecewise(1, x > y, 3)", 3.0),
        ("piecewise(1, x > y, 2, true)", 2.0),
        ("piecewise(10, (x < y) && (z < 0), 20)", 10.0),
        ("piecewise(10, xor(true, true), 20)", 20.0),
        ("piecewise(10, z < x < y, 20)", 10.0),
        ("piecewise(10, implies(false, false), 20)", 10.0),
        ("time * 2", 3.0),
        ("pi", math.pi),
        ("exponentiale", math.e),
        ("avogadro", 6.02214179e23),
        ("INF", math.inf),
    ],
)
def test_translate(formula: str, expected: float) -> None:
    assert evaluate(formula) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        # integer exponents are the plain power, also for a negative base
        ("z^2", 0.25),
        ("z^3", -0.125),
        ("z^(-2)", 4.0),
        ("pow(z, y)", -0.125),
        # a real power of a negative base is continued with the value at zero
        ("z^2.5", 0.0),
        ("z^(-0.5)", math.inf),
        ("sqrt(z)", 0.0),
        ("root(3, z)", 0.0),
        ("pow(z, x + 0.5)", 0.0),
        ("root(x, z)", 0.0),
        # nothing changes for a positive base
        ("x^2.5", 2**2.5),
        ("x^(-0.5)", 2**-0.5),
        ("pow(x, y + 0.5)", 2**3.5),
        ("root(y, x)", 2 ** (1 / 3)),
    ],
)
def test_power(formula: str, expected: float) -> None:
    assert evaluate(formula) == pytest.approx(expected)


@pytest.mark.parametrize("formula", ["v^2.5", "v^0.5", "sqrt(v)", "root(3, v)", "pow(v, w)", "v^2"])
@pytest.mark.parametrize("base", [-1e-12, 0.0, 2.0])
def test_power_gradient_is_finite(formula: str, base: float) -> None:
    """A base at or slightly below zero, as in a rejected solver step, gives no NaN gradient."""
    expr = translate(libsbml.parseL3Formula(formula), lambda sid: sid)

    def f(v: jax.Array, w: jax.Array) -> jax.Array:
        return eval(expr.code, {"jnp": jnp, "special": special, "v": v, "w": w})

    value, gradient = jax.value_and_grad(f, argnums=(0, 1))(base, 2.8)
    assert math.isfinite(float(value))
    assert all(math.isfinite(float(g)) for g in gradient)
    if base > 0:
        h = 1e-6
        fd = float((f(base + h, 2.8) - f(base - h, 2.8)) / (2 * h))
        assert float(gradient[0]) == pytest.approx(fd, rel=1e-6)


def test_dependencies() -> None:
    expr = translate(libsbml.parseL3Formula("k1 * S1 / (Km + S1) + time"), lambda sid: sid)
    assert expr.deps == {"k1", "S1", "Km", "t"}


def test_resolve_local() -> None:
    expr = translate(libsbml.parseL3Formula("k * S"), lambda sid: {"k": "R1_k"}.get(sid, sid))
    assert expr.deps == {"R1_k", "S"}


def test_delay_unsupported() -> None:
    with pytest.raises(UnsupportedError, match="delay"):
        translate(libsbml.parseL3Formula("delay(x, 1)"), lambda sid: sid)


def test_rate_of_without_context() -> None:
    with pytest.raises(UnsupportedError, match="rateOf"):
        translate(libsbml.parseL3Formula("rateOf(x)"), lambda sid: sid)
