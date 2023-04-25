"""Hybrid symbolic module."""
from typing import Optional, List, Dict, Any, Union, Tuple
from collections.abc import Callable

import gplearn.genetic
from gplearn.functions import _Function as Function
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from .utils import FeatureExtractor, Parameterization, ArrayLike, ModelLike, ensure_numpy


def make_custom_gplearn_functions(data_set: xr.Dataset, spatial_funcs: List[str]):
    """Define custom gplearn functions for spatial derivatives.

    Parameters
    ----------
    data_set : xarray.Dataset
        Dataset containing spatial variables.

    spatial_funcs : List[str]
        Names of spatial differential operators to explore during genetic
        programming. Must be supported by the FeatureExtractor class, e.g.
        ['ddx', 'ddy', 'laplacian', 'advected'].

    Returns
    -------
    List[gplearn.functions._Function]
        List of functions representing spatial differential operators that can
        be evaluated over the dataset during genetic programming

    """
    extractor = FeatureExtractor(data_set)

    def apply_spatial(function_name: str, flat_array: ArrayLike) -> np.ndarray:
        """Apply a `spatial_function` to an initially `flat_array`."""
        spatial_func = getattr(extractor, function_name)
        spatial_array = flat_array.reshape(extractor.spatial_shape)
        spatial_output = ensure_numpy(spatial_func(spatial_array))
        return spatial_output.reshape(flat_array.shape)

    return [
        Function(
            function=lambda x: apply_spatial(function_name, x),
            name=function_name,
            arity=1
        )
        for function_name in spatial_funcs
    ]


def run_gplearn_iteration(
    data_set: xr.Dataset,
    target: np.ndarray,
    base_features: Optional[List[str]] = None,
    base_functions: Optional[List[str]] = None,
    spatial_functions: Optional[List[str]] = None,
    **kwargs: Dict[str, Any],
):
    """Run gplearn for one iteration using custom spatial derivatives.

    Parameters
    ----------
    data_set : xarray.Dataset
        Dataset generated from pyqg.QGModel runs
    target : numpy.ndarray
        Target spatial field to be predicted from dataset attributes
    base_features : List[str]
        Features from the dataset to be used as the initial set of atomic
        inputs to genetic programming. Defaults to ['q', 'u', 'v'].
    base_functions : List[str]
        Names of gplearn built-in functions to explore during genetic
        programming (in addition to differential operators). Defaults to
        ['add', 'mul'].
    spatial_functions : List[str]
        Names of spatial differential operators to explore during genetic
        programming. Defaults to ['ddx', 'ddy', 'laplacian', 'advected'].
    **kwargs : dict
        Additional arguments to pass to gplearn.genetic.SymbolicRegressor

    Returns
    -------
    gplearn.genetic.SymbolicRegressor
        A trained symbolic regression model that predicts the ``target`` based
        on functions of the dataset's ``base_features``.

    """
    base_features = ["q", "u", "v"] if base_features is None else base_features
    base_functions = ["add", "mul"] if base_functions is None else base_functions
    spatial_functions = make_custom_gplearn_functions(data_set, spatial_functions)

    # Flatten the input and target data
    inputs = np.array([data_set[feature].data.reshape(-1) for feature in base_features]).T
    targets = target.reshape(-1)

    gplearn_kwargs = {
        "population_size": 5000,
        "generations": 50,
        "p_crossover": 0.7,
        "p_subtree_mutation": 0.1,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.1,
        "max_samples": 0.9,
        "verbose": 1,
        "parsimony_coefficient": 0.001,
        "metric": "pearson",  # IMPORTANT: fit with pearson corr and not MSE.
        "const_range": (-2, 2),
    }

    gplearn_kwargs.update(**kwargs)

    # Configure gplearn to run with a relatively small population
    # and for relatively few generations (again for performance)
    regressor = gplearn.genetic.SymbolicRegressor(
        feature_names=base_features,
        function_set=base_functions + spatial_functions,  # use our custom ops
        **gplearn_kwargs,
    )

    # Fit the model
    regressor.fit(inputs, targets)

    # Return the result
    return regressor


class LinearSymbolicRegression(Parameterization):
    """Linear symbolic regress parameterization.

    Parameters
    ----------
    models : List[sklearn.linear_model.LinearRegression]
        Linear models to predict each layer's target based on input
        expressions
    inputs : List[str]
        List of string input expressions and functions that can be
        extracted from a pyqg.QGModel or dataset using a
        ``FeatureExtractor``
    target : str
        String expression representing the target variable.

    """

    def __init__(
        self,
        models: List[LinearRegression],
        inputs: List[str],
        target: str,
    ):
        """Build ``LinearSymbolicRegression``."""
        self.models = models
        self.inputs = inputs
        self.target = target

    @property
    def targets(self) -> List[str]:
        """Return the targets.

        Returns
        -------
        List[str]
            The target.

        """
        return [self.target]

    def predict(self, model_or_dataset: ModelLike) -> Dict[str, ArrayLike]:
        """Make a prediction for a given model or dataset.

        Parameters
        ----------
        model_or_dataset : Union[pyqg.QGModel, xarray.Dataset]
            The live-running model or dataset of stored model runs for which we
            want to make subgrid forcing predictions.

        Returns
        -------
        result : Dict[str, numpy.ndarray]
            For each string target expression, the predicted subgrid forcing
            (as an array with the same shape as the corresponding inputs in the
            model or dataset).

        """
        extract = FeatureExtractor(model_or_dataset)

        x = extract(self.inputs)

        preds = []

        # Do some slightly annoying reshaping to properly apply LR coefficients
        # to data that may or may not have extra batch dimensions
        for idx, lr_model in enumerate(self.models):
            data_indices = [slice(None) for _ in x.shape]
            data_indices[-3] = idx
            x_z = x[tuple(data_indices)]
            coef_indices = [np.newaxis for _ in x_z.shape]
            coef_indices[0] = slice(None)
            c_z = lr_model.coef_[tuple(coef_indices)]
            pred_z = (x_z * c_z).sum(axis=0)
            preds.append(pred_z)

        preds = np.stack(preds, axis=-3)
        res = {}
        res[self.target] = preds
        return res

    @classmethod
    def fit(
        cls,
        data_set: xr.Dataset,
        inputs: List[str],
        target: str = "q_subgrid_forcing",
    ):
        """Fit lin-reg param on ``dataset`` in terms of symbolic ``inputs``.

        Parameters
        ----------
        data_set : xarray.Dataset
            Dataset of pyqg.QGModel runs
        inputs : List[str]
            List of expressions that can be evaluated by a
            ``FeatureExtractor``, to be used as inputs for the linear
            regression models
        target : Optional[str]
            Target variable to predict. Defaults to the subgrid forcing of
            potential vorticity.

        Returns
        -------
        LinearSymbolicRegression
            Resulting linear regression parameterization.

        """
        extractors = [FeatureExtractor(layer) for layer in each_layer(data_set)]

        models = [
            LinearRegression(fit_intercept=False).fit(
                extract(inputs, flat=True),
                extract(target, flat=True)
            )
            for extract in extractors
        ]

        return cls(models, inputs, target)


def each_layer(
    data_set: xr.Dataset
) -> List[xr.Dataset]:
    """Return a list of datasets, either for each vertical layer in `data_set`
    if it has a vertical dimension `lev`, or just `[data_set]` otherwise.

    Parameters
    ----------
    data_set : xarray.Dataset
        Dataset with or without a vertical dimension `lev`

    Returns
    -------
    List[xarray.Dataset]
        List of datasets for each vertical layer, if present

    """
    if 'lev' in data_set:
        return [data_set.isel(lev=z) for z in range(len(data_set.lev))]
    else:
        return [data_set]


def corr(spatial_data_a: xr.DataArray, spatial_data_b: xr.DataArray) -> float:
    """Return the Pearson correlation between two spatial data arrays.

    Parameters
    ----------
    a : xarray.DataArray
        First spatial data array
    b : xarray.DataArray
        Second spatial data array

    Returns
    -------
    float
        Correlation between the data arrays

    """
    return pearsonr(
        np.array(spatial_data_a.data).ravel(),
        np.array(spatial_data_b.data).ravel(),
    )[0]

def hybrid_symbolic_regression(  # pylint: disable=too-many-locals
    data_set: xr.Dataset,
    target: str = "q_subgrid_forcing",
    max_iters: int = 10,
    verbose: bool = True,
    **kw,
) -> Tuple[List[str], List[LinearRegression]]:
    """Run hybrid symbolic and linear regression.

    Uses symbolic regression to find expressions correlated wit the output,
    then fitting linear regression to get an exact expression, then running
    symbolic regression again on the resulting residuals (repeating until
    ``max_iters``).

    Parameters
    ----------
    data_set : xarray.Dataset
        Dataset of pyqg.QGModel runs.
    target : str
        String representing target variable to predict. Defaults to subgrid
        forcing of potential vorticity.
    max_iters : int
        Number of iterations to run. Defaults to 10.
    verbose : bool
        Whether to print intermediate outputs. Defaults to True.
    **kw : dict
        Additional arguments to pass to ``run_gplearn_iteration``.

    Returns
    -------
    Tuple[List[str], List[LinearSymbolicRegression]]
        List of terms discovered at each iteration alongside list of linear
        symbolic regression parameterization objects (each fit to all terms
        available at corresponding iteration)

    """
    extract = FeatureExtractor(data_set)
    residual = data_set[target]
    terms, vals, hybrid_regressors = [], [], []

    try:
        for i in range(max_iters):
            for data_set_layer, residual_layer in zip(each_layer(data_set), each_layer(residual)):
                symbolic_regressor = run_gplearn_iteration(
                    data_set_layer, target=residual_layer, **kw
                )
                new_term = str(symbolic_regressor._program)
                new_vals = extract(new_term)
                # Prevent spurious duplicates, e.g. ddx(q) and ddx(add(1,q))
                if not any(corr(new_vals, v) > 0.99 for v in vals):
                    terms.append(new_term)
                    vals.append(new_vals)
            hybrid_regressor = LinearSymbolicRegression.fit(data_set, terms, target)
            hybrid_regressors.append(hybrid_regressor)
            preds = hybrid_regressor.test_offline(data_set).load()
            residual = (data_set[target] - preds[f"{target}_predictions"]).load()
            if verbose:
                print(f"Iteration {i+1}")
                print("Discovered terms:", terms)
                print("Train correlations:", preds.correlation.data)

    except KeyboardInterrupt:
        if verbose:
            print("Exiting early due to keyboard interrupt")

    return terms, hybrid_regressors
