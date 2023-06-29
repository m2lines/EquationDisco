# M<sup>2</sup>LInES Equation discovery package

This package implements a method for equation discovery over 2D periodic spatial fields using interleaved linear regression and symbolic regression with spatial differential operators on residuals. This approach allows us to discover parsimonious symbolic relationships between spatial fields using high-order differential terms -- as well as weighted sums of such terms, even if they have significantly different orders of magnitude.

The library is closely integrated with [pyqg](https://github.com/pyqg/pyqg), a simulator of quasigeostrophic fluid dynamics, though the symbolic regression method [can be run independently](./notebooks/example_without_pyqg.ipynb).

## Usage

Below we show a basic example of how to run symbolic regression on a dataset, then interpret the learned expression as a parameterization within pyqg, and use it to run a parameterized fluid dynamics simulation:

```python
import eqn_disco

terms_by_iter, parameterizations_by_iter = eqn_disco.hybrid_symbolic.hybrid_symbolic_regression(
  dataset,
  max_iters=3,
  target='fancy_subgrid_forcing',
  base_features=['velocity', 'potential_vorticity'],
  base_functions=['mul', 'add'],
  spatial_functions=['ddx', 'ddy', 'laplacian', 'advected'],
  **gplearn_arguments
)

parameterization = parameterizations_by_iter[-1]

pyqg_run = parameterization.run_online(**pyqg_arguments)
```

For more details, see notebooks showcasing this approach on datasets [generated from pyqg](./notebooks/hybrid_symbolic.ipynb) or [constructed independently](./notebooks/example_without_pyqg.ipynb).

To install, clone the repository, and run `pip install -e .`.

## Authors and contributors

The code in this repository was originally developed by [Andrew Ross](https://github.com/asross) and [Pavel Perezhogin](https://github.com/Pperezhogin), in [this repository](https://github.com/m2lines/pyqg_parameterization_benchmarks), for use in the publication: Benchmarking of machine learning ocean parameterizations in an idealized model (published in JAMES, <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022MS003258>).

It is currently being worked on by [Jim Denholm](https://github.com/jdenholm) and [Andrew Ross](https://github.com/asross), along with other members of the [ICCS](https://github.com/orgs/Cambridge-ICCS/dashboard) and [M<sup>2</sup>LInES](https://github.com/m2lines).

## License

The work is available under an MIT License; see [the license](LICENSE).
