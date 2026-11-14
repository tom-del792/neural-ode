"""hybrid: neural networks in mechanistic PK/PD models (work package 2).

The mechanistic model comes from `sbml2jax`; this package adds the networks,
the data generation and the training. Everything specific to a substance
(SBML model, inputs, outputs, dosing, covariates) is configuration, so the
same code applies to another model without changes.
"""

import sbml2jax  # noqa: F401  (enables double precision for the simulations)

__version__ = "0.1.0"
