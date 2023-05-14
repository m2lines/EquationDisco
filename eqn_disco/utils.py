"""Utility functions."""
import operator
from typing import List, Dict, Union, Any, Tuple, Optional
import re

try:
    import pyqg

    IMPORTED_PYQG = True
except ImportError:
    IMPORTED_PYQG = False
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ModelLike = Union[pyqg.Model, xr.Dataset] if IMPORTED_PYQG else xr.Dataset
ArrayLike = Union[np.ndarray, xr.DataArray]
Numeric = Union[ArrayLike, int, float]
StringOrNumeric = Union[str, Numeric]


def ensure_numpy(array: ArrayLike) -> np.ndarray:
    """Ensure a given array-like input is converted to numpy.

    Parameters
    ----------
    array : Union[numpy.ndarray, xarray.DataArray]
        Input array

    Returns
    -------
    numpy.ndarray
        Numpy representation of input array
    """
    if isinstance(array, xr.DataArray):
        return array.data
    return array


class Parameterization(pyqg.Parameterization if IMPORTED_PYQG else object):  # type: ignore
    """Helper class for defining parameterizations.

    This extends the normal pyqg.Parameterization framework to handle
    prediction of either subgrid forcings or fluxes, as well as to apply to
    either pyqg.Models or xarray.Datasets. Can also be used without pyqg, though
    in a more limited fashion.

    """

    @property
    def targets(self) -> List[str]:
        """List the names of the quantities the parameterization predicts.

        Returns
        -------
        List[str]
            List of parameterization targets returned by this parameterization.
            If using within pyqg, valid options are "q_forcing_total",
            "q_subgrid_forcing", "u_subgrid_forcing", "v_subgrid_forcing",
            "uq_subgrid_flux", "vq_subgrid_flux", "uu_subgrid_flux",
            "vv_subgrid_flux", and "uv_subgrid_flux". See the dataset
            description notebook or the paper for more details on the meanings
            of these target fields and how they're used.

        """
        raise NotImplementedError

    def predict(self, model_or_dataset: ModelLike) -> Dict[str, ArrayLike]:
        """Subgrid forcing predictions, as a dictionary of target => array.

        Parameters
        ----------
        model_or_dataset : Union[pyqg.QGModel, xarray.Dataset]
            Model or dataset for which we are making subgrid forcing
            predictions.

        Returns
        -------
        Dict[str, Union[numpy.ndarray, xarray.DataArray]]
            Dictionary of target variable name to subgrid forcing predictions,
            either as numpy arrays (if the model is a pyqg.QGModel) or as
            xarray.DataArrays (if the model is an xarray.Dataset).

        """
        raise NotImplementedError

    @property
    def parameterization_type(self) -> str:
        """Return the parameterization type.

        Returns whether this is a potential vorticity parameterization (i.e.
        "q_parameterization") or velocity parameterization
        (i.e. "uv_parameterization"). This is needed for pyqg to properly handle parameterizations
        internally.

        Returns
        -------
        str
            Indication of whether the parameterization targets PV or velocity.

        """
        assert IMPORTED_PYQG, "pyqg must be installed to use this method"

        if any(q in self.targets[0] for q in ["q_forcing", "q_subgrid"]):
            return "q_parameterization"

        return "uv_parameterization"

    def __call__(self, model: ModelLike) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Invoke the parameterization in the format required by pyqg.

        Parameters
        ----------
        m : Union[pyqg.QGModel, xarray.Dataset]
            Model or dataset.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
           Either a single array (if ``parameterization_type`` is
           ``q_parameterization``) or a tuple of two arrays (if
           ``parameterization_type`` is ``uv_parameterization``) representing
           the subgrid forcing, with each array having the same shape and data
           type as the model's PV variable.

        """
        assert IMPORTED_PYQG, "pyqg must be installed to use this method"

        def _ensure_array(array: Numeric) -> np.ndarray:
            """Convert an array-like to numpy with model-compatible dtype."""
            return ensure_numpy(array).astype(model.q.dtype)  # type: ignore

        preds = self.predict(model)
        keys = list(sorted(preds.keys()))  # these are the same as our targets
        assert keys == self.targets

        # decide how to convert parameterization predictions to the right
        # output format
        if len(keys) == 1:
            # if there's only one target, it's a PV parameterization, and we can
            # just return the array
            return _ensure_array(preds[keys[0]])
        if keys == ["uq_subgrid_flux", "vq_subgrid_flux"]:
            # these are PV subgrid fluxes; we need to take their divergence
            extractor = FeatureExtractor(model)
            return _ensure_array(
                extractor.ddx(preds["uq_subgrid_flux"])
                + extractor.ddy(preds["vq_subgrid_flux"])
            )
        if "uu_subgrid_flux" in keys and len(keys) == 3:
            # these are velocity subgrid fluxes; we need to take two sets of
            # divergences and return a tuple
            extractor = FeatureExtractor(model)
            return (
                _ensure_array(
                    extractor.ddx(preds["uu_subgrid_flux"])
                    + extractor.ddy(preds["uv_subgrid_flux"])
                ),
                _ensure_array(
                    extractor.ddx(preds["uv_subgrid_flux"])
                    + extractor.ddy(preds["vv_subgrid_flux"])
                ),
            )
        # Otherwise, this is a "simple" velocity parameterization; return a tuple
        return tuple(_ensure_array(preds[k]) for k in keys)

    def run_online(
        self,
        sampling_freq: int = 1000,
        nx: int = 64,  # pylint: disable=invalid-name
        **kwargs,
    ) -> xr.Dataset:
        """Initialize and run a parameterized pyqg.QGModel.

        Saves snapshots periodically.

        Parameters
        ----------
        sampling_freq : int
            Number of timesteps (hours) between saving snapshots. Defaults to
            1000.
        spatial_res : int
            Number of horizontal grid points for the model. Defaults to 64.
        **kwargs : dict
            Other simulation parameters to pass to pyqg.QGModel.

        Returns
        -------
        data_set : xarray.Dataset
            Dataset of parameterized model run snapshots

        """
        assert IMPORTED_PYQG, "pyqg must be installed to run this method"

        # Initialize a pyqg model with this parameterization
        params = dict(kwargs)
        params[self.parameterization_type] = self
        params["nx"] = nx
        model = pyqg.QGModel(**params)

        # Run it, saving snapshots
        snapshots = []
        while model.t < model.tmax:
            if model.tc % sampling_freq == 0:
                snapshots.append(model.to_dataset().copy(deep=True))
            model._step_forward()  # pylint: disable=protected-access

        data_set = xr.concat(snapshots, dim="time")

        # Diagnostics get dropped by this procedure since they're only present for
        # part of the timeseries; resolve this by saving the most recent
        # diagnostics (they're already time-averaged so this is ok)
        for key, value in snapshots[-1].variables.items():
            if key not in data_set:
                data_set[key] = value.isel(time=-1)

        # Drop complex variables since they're redundant and can't be saved
        complex_vars = [k for k, v in data_set.variables.items() if np.iscomplexobj(v)]
        data_set = data_set.drop_vars(complex_vars)

        return data_set

    def test_offline(self, data_set: xr.Dataset) -> xr.Dataset:
        """Test offline performance of parameterization on existing dataset.

        Parameters
        ----------
        data_set : xarray.Dataset
            Dataset containing coarsened inputs and subgrid forcing variables
            matching this parameterization's targets.

        Returns
        -------
        test : xarray.Dataset
            Dataset of offline performance metrics specific to each predicted
            target, along with the target values themselves (subselected from
            the original dataset).

        """
        test = data_set[self.targets]

        for key, val in self.predict(data_set).items():
            truth = test[key]
            test[f"{key}_predictions"] = truth * 0 + val
            preds = test[f"{key}_predictions"]
            error = (truth - preds) ** 2
            true_var = (truth - truth.mean()) ** 2

            def dims_except(*dims, key=key):
                return [d for d in test[key].dims if d not in dims]

            time = dims_except("x", "y", "lev")
            space = dims_except("time", "lev")
            both = dims_except("lev")

            test[f"{key}_spatial_mse"] = error.mean(dim=time)
            test[f"{key}_temporal_mse"] = error.mean(dim=space)
            test[f"{key}_mse"] = error.mean(dim=both)

            test[f"{key}_spatial_skill"] = 1 - test[
                f"{key}_spatial_mse"
            ] / true_var.mean(dim=time)
            test[f"{key}_temporal_skill"] = 1 - test[
                f"{key}_temporal_mse"
            ] / true_var.mean(dim=space)
            test[f"{key}_skill"] = 1 - test[f"{key}_mse"] / true_var.mean(dim=both)

            test[f"{key}_spatial_correlation"] = xr.corr(truth, preds, dim=time)
            test[f"{key}_temporal_correlation"] = xr.corr(truth, preds, dim=space)
            test[f"{key}_correlation"] = xr.corr(truth, preds, dim=both)

        for metric in ["correlation", "mse", "skill"]:
            test[metric] = sum(test[f"{key}_{metric}"] for key in self.targets) / len(
                self.targets
            )

        return test


class FeatureExtractor:
    """Helper class for evaluating arbitrarily deep string expressions.

    For example, "laplacian(ddx(mul(u,q)))") on either pyqg.QGModel or
    xarray.Dataset instances.

    Parameters
    ----------
    model_or_dataset : Union[pyqg.QGModel, xarray.Dataset]
        Model or dataset we'll be extracting features from.

    """

    def __call__(
        self, feature_or_features: Union[str, List[str]], flat: bool = False
    ) -> np.ndarray:
        """Extract the given feature/features from underlying dataset/ model.

        Parameters
        ----------
        feature_or_features : Union[str, List[str]]
            Either a single string expression or a list of string expressions.
        flat : bool, optional
            Whether to flatten the output of each feature to an array with only
            one dimension (rather than a spatial field). Defaults to False.

        Returns
        -------
        res : numpy.ndarray
            Array of values of corresponding features.

        """
        if isinstance(feature_or_features, str):
            res = ensure_numpy(self.extract_feature(feature_or_features))  # type: ignore
            if flat:
                res = res.reshape(-1)

        else:
            res = np.array(
                [ensure_numpy(self.extract_feature(f)) for f in feature_or_features]  # type: ignore
            )
            if flat:
                res = res.reshape(len(feature_or_features), -1).T
        return res

    def __init__(
        self, model_or_dataset: ModelLike, example_realspace_input: Optional[str] = None
    ):
        """Build ``FeatureExtractor``."""
        self.model = model_or_dataset
        self.cache = {}

        assert hasattr(
            self.model, "x"
        ), "dataset must have horizontal dimension named `x`"
        assert hasattr(
            self.model, "y"
        ), "dataset must have vertical dimension named `y`"

        if example_realspace_input is None:
            if hasattr(self.model, "q"):
                example_realspace_input = "q"
            elif isinstance(self.model, xr.Dataset):
                example_realspace_input = next(
                    key for key, val in self.model.items() if "x" in val.dims
                )
        self.example_realspace_input = getattr(self.model, example_realspace_input)

        if hasattr(self.model, "_ik"):
            self.ik, self.il = np.meshgrid(  # pylint: disable=invalid-name
                self.model._ik, self.model._il
            )
        elif hasattr(self.model, "fft"):
            self.ik = 1j * self.model.k  # pylint: disable=invalid-name
            self.il = 1j * self.model.l  # pylint: disable=invalid-name
        else:
            if not hasattr(self.model, "k"):
                grid_length = self.example_realspace_input.shape[-1]
                horizontal_wavenumbers = 2 * np.pi * np.arange(0.0, grid_length / 2 + 1)
                vertical_wavenumbers = (
                    2
                    * np.pi
                    * np.append(
                        np.arange(0.0, grid_length / 2),
                        np.arange(-grid_length / 2, 0.0),
                    )
                )
                self.model["k"] = horizontal_wavenumbers
                self.model["l"] = vertical_wavenumbers

            k, l = np.meshgrid(  # pylint: disable=invalid-name
                self.model.k, self.model.l
            )
            self.ik = 1j * k  # pylint: disable=invalid-name
            self.il = 1j * l  # pylint: disable=invalid-name

        self.nx = self.ik.shape[0]  # pylint: disable=invalid-name
        self.wv2 = self.ik**2 + self.il**2

    # Helpers for taking FFTs / deciding if we need to
    def fft(self, real_array: ArrayLike) -> ArrayLike:
        """Compute the FFT of ``x``.

        Parameters
        ----------
        x : Union[numpy.ndarray, xarray.DataArray]
            An input array in real space

        Returns
        -------
        fft : Union[numpy.ndarray, xarray.DataArray]
            An output array in spectral space

        """
        try:
            # pyqg.Models will respond to `fft` (which might be pyFFTW, which is fast)
            return self.model.fft(real_array)
        except AttributeError:
            # if we got an attribute error, that means we have an xarray.Dataset.
            # use numpy FFTs and return a data array instead.
            dims = self.spectral_dims
            return xr.DataArray(
                np.fft.rfftn(real_array, axes=(-2, -1)),
                dims=dims,
                coords={d: self[d] for d in dims},
            )

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """Spatial shape of variables in real space."""
        return self.example_realspace_input.shape

    @property
    def spatial_dims(self) -> Tuple[str, ...]:
        """Names of spatial dimensions in real space."""
        return self.example_realspace_input.dims

    @property
    def spectral_dims(self) -> List[str]:
        """Names of spatial dimensions in spectral space."""
        return [{"y": "l", "x": "k"}.get(d, d) for d in self.spatial_dims]

    def ifft(self, spectral_array: ArrayLike) -> ArrayLike:
        """Compute the inverse FFT of ``x``.

        Parameters
        ----------
        x : Union[numpy.ndarray, xarray.DataArray]
            An input array in spectral space

        Returns
        -------
        ifft : Union[numpy.ndarray, xarray.DataArray]
            An output array in real space

        """
        try:
            return self.model.ifft(spectral_array)
        except AttributeError:
            # Convert numpy to xarray by adding 0 with the right dimensions
            realspace_array = np.fft.irfftn(spectral_array, axes=(-2, -1))
            zero_as_data_array = self.example_realspace_input * 0
            return zero_as_data_array + realspace_array

    def _is_real(self, arr: ArrayLike) -> bool:
        """Check if a given array is in real space."""
        return len(set(arr.shape[-2:])) == 1

    def _real(self, feature: StringOrNumeric) -> Numeric:
        """Load and convert a feature to real space, if necessary."""
        value = self[feature]
        if isinstance(value, float):
            return value
        if self._is_real(value):
            return value
        return self.ifft(value)

    def _compl(self, feature: StringOrNumeric) -> Numeric:
        """Load and convert a feature to spectral space, if necessary."""
        value = self[feature]
        if isinstance(value, float):
            return value
        if self._is_real(value):
            return self.fft(value)
        return value

    # Spectral derivatrives
    def ddxh(self, field: StringOrNumeric) -> ArrayLike:
        """Compute the horizontal derivative of ``field`` in spectral space."""
        return self.ik * self._compl(field)

    def ddyh(self, field: StringOrNumeric) -> ArrayLike:
        """Compute the vertical derivative of ``field`` in spectral space."""
        return self.il * self._compl(field)

    def divh(self, field_x: StringOrNumeric, field_y: StringOrNumeric) -> ArrayLike:
        """Compute the divergence of a vector field in spectral space."""
        return self.ddxh(field_x) + self.ddyh(field_y)

    def curlh(self, field_x: StringOrNumeric, field_y: StringOrNumeric) -> ArrayLike:
        """Compute the curl of a vector field in spectral space."""
        return self.ddxh(field_y) - self.ddyh(field_x)

    def laplacianh(self, field: StringOrNumeric) -> ArrayLike:
        """Compute the Laplacian of a field in spectral space."""
        return self.wv2 * self._compl(field)

    def advectedh(self, possibly_spectral_field: StringOrNumeric) -> ArrayLike:
        """Advect a field in spectral space."""
        assert hasattr(
            self.model, "ufull"
        ), "Model must have `ufull` and `vfull` to advect"
        assert hasattr(
            self.model, "vfull"
        ), "Model must have `ufull` and `vfull` to advect"
        field = self._real(possibly_spectral_field)
        return self.ddxh(field * self.model.ufull) + self.ddyh(field * self.model.vfull)

    # Real counterparts
    def ddx(self, field: StringOrNumeric) -> Numeric:
        """Compute the horizontal derivative of a field."""
        return self._real(self.ddxh(field))

    def ddy(self, field: StringOrNumeric) -> Numeric:
        """Compute the vertical derivative of a field."""
        return self._real(self.ddyh(field))

    def laplacian(self, field: StringOrNumeric) -> Numeric:
        """Compute the Laplacian of a field."""
        return self._real(self.laplacianh(field))

    def advected(self, field: StringOrNumeric) -> Numeric:
        """Advect a field."""
        return self._real(self.advectedh(field))

    def curl(self, field_x: StringOrNumeric, field_y: StringOrNumeric) -> Numeric:
        """Compute the curl of two fields."""
        return self._real(self.curlh(field_x, field_y))

    def div(self, field_x: StringOrNumeric, field_y: StringOrNumeric) -> Numeric:
        """Compute the divergence of two fields."""
        return self._real(self.divh(field_x, field_y))

    # Main function: interpreting a string as a feature
    def extract_feature(self, feature: str) -> Numeric:
        """Evaluate a string expression and convert it to a number or array.

        Examples of valid string expressions include:

            "1.053"
            "q"
            "u"
            "add(u, 1.053)"
            "mul(u, add(q, u))"
            "laplacian(advected(curl(u, v)))"

        More formally, features should obey the following grammar (with
        additional whitespace allowed within expressions):

            feature
                unary_expression | binary_expression | number | variable_name
            unary_expression
                unary_operator "(" feature ")"
            binary_expression
                binary_operator "(" feature "," feature ")"
            unary_operator
                "neg" | "abs" | "ddx" | "ddy" | "advected" | "laplacian"
            binary_operator
                "mul" | "add" | "sub" | "pow" | "div" | "curl"
            number
                "-"? [0-9]+ "."? [0-9]*
            variable_name
                .*

        Essentially, features should either be numbers, atomic variables (that
        get passed directly to the underlying model), or unary/binary operators
        applied to features.

        The specific operators supported are:
            - ``neg``, which negates the expression
            - ``abs``, which takes the absolute bvalue
            - ``ddx``, which takes the horizontal derivative
            - ``ddy``, which takes the vertical derivative
            - ``advected``, which applies advection to the expression
            - ``laplacian``, which takes the Laplacian of the expression
            - ``mul``, which multiplies two expressions
            - ``add``, which adds two expressions
            - ``sub``, which subtracts the second expression from the first
            - ``pow``, which takes the first expression to the power of the second
            - ``div``, which takes the divergence of the vector field whose x
                and y components are given by the two expressions, respectively
            - ``curl``, which takes the curl of the vector field whose x and y
                components are given by the two expressions, respectively

        Parameters
        ----------
        feature : str
            String representing a mathematical expression to be evaluated,
            matching the feature extraction grammar.

        Returns
        -------
        numeric : Union[numpy.ndarray, xarray.DataArray, int, float]
            Numeric or array-like expression representing the value of the
            feature.
        """

        def extract_pair(string: str) -> Tuple[Numeric, Numeric]:
            """Extract two features from a comma-separated pair."""
            depth = 0
            for i, char in enumerate(string):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    return (
                        self.extract_feature(string[:i].strip()),
                        self.extract_feature(string[i + 1 :].strip()),
                    )
            raise ValueError(f"string {string} is not a comma-separated pair")

        def real_or_spectral(arr: List[str]) -> List[str]:
            """Convert a list of strings to a list of real/spectral versions."""
            return arr + [a + "h" for a in arr]

        if not self._extracted(feature):
            # Check if the feature looks like "function(expr1, expr2)"
            # (better would be to write a grammar + use a parser,
            # but this is a very simple DSL)
            match = re.search(r"^([a-z]+)\((.*)\)$", feature)
            if match:
                op_name, inner = match.group(1), match.group(2)
                if op_name in ["mul", "add", "sub", "pow"]:
                    self.cache[feature] = getattr(operator, op_name)(
                        *extract_pair(inner)
                    )
                elif op_name in ["neg", "abs"]:
                    self.cache[feature] = getattr(operator, op_name)(
                        self.extract_feature(inner)
                    )
                elif op_name in real_or_spectral(["div", "curl"]):
                    self.cache[feature] = getattr(self, op_name)(*extract_pair(inner))
                elif op_name in real_or_spectral(
                    ["ddx", "ddy", "advected", "laplacian"]
                ):
                    self.cache[feature] = getattr(self, op_name)(
                        self.extract_feature(inner)
                    )
                else:
                    raise ValueError(f"could not interpret {feature}")
            elif feature == "streamfunction":
                # hack to make streamfunctions work in both datasets & pyqg.Models
                self.cache[feature] = self.ifft(self["ph"])
            elif re.search(r"^\-?\d+\.?\d*$", feature):
                # ensure numbers still work
                return float(feature)
            else:
                raise ValueError(f"could not interpret {feature}")

        return self[feature]

    def _extracted(self, key: str) -> bool:
        """Check if a feature has already been extracted."""
        return key in self.cache or hasattr(self.model, key)

    # A bit of additional hackery to allow for the reading of features or properties
    def __getitem__(self, attribute: StringOrNumeric) -> Any:
        """Read an attribute from the model or cache, or echo back if numeric."""
        if isinstance(attribute, str):
            if attribute in self.cache:
                return self.cache[attribute]
            if re.search(r"^[\-\d\.]+$", attribute):
                return float(attribute)
            return getattr(self.model, attribute)
        if any(
            isinstance(attribute, kls) for kls in [xr.DataArray, np.ndarray, int, float]
        ):
            return attribute
        raise KeyError(attribute)


def energy_budget_term(model, term):
    """Compute an energy budget term, handling parameterization contributions if present."""
    val = model[term]
    if "paramspec_" + term in model:
        val += model["paramspec_" + term]
    return val.sum("l")


def energy_budget_figure(models, skip=0):
    """Plot the energy budget for a set of models."""
    fig = plt.figure(figsize=(12, 5))
    vmax = 0
    for i, term in enumerate(["KEflux", "APEflux", "APEgenspec", "KEfrictionspec"]):
        plt.subplot(1, 4, i + 1, title=term)
        plt.axhline(0, color="gray", ls="--")

        for model, label in models:
            spec = energy_budget_term(model, term)
            plt.semilogx(
                model.k[skip:],
                spec[skip:],
                label=label,
                lw=3,
                ls=("--" if "+" in label else "-"),
            )
            vmax = max(vmax, spec[skip:].max())
        plt.grid(which="both", alpha=0.25)
        if i == 0:
            plt.ylabel("Energy transfer $[m^2 s^{-3}]$")
        else:
            plt.gca().set_yticklabels([])
        if i == 3:
            plt.legend()
        plt.xlabel("Zonal wavenumber $[m^{-1}]$")
    for axis in fig.axes:
        axis.set_ylim(-vmax, vmax)
    plt.tight_layout()
    return fig


def example_non_pyqg_data_set(
    grid_length: int = 8, num_samples: int = 20
) -> xr.Dataset:
    """Create a simple xarray dataset for testing the library without `pyqg`.

    This dataset has a single variable called `inputs` with `x`, `y`, and
    `batch` coordinates.  It can be used in various methods of the library
    without needing to invoke `pyqg`.

    Parameters
    ----------
    grid_length : int
        The length of the grid in each dimension.
    num_samples : int
        The number of samples in the dataset.

    Returns
    -------
    xr.Dataset
        The dataset.

    """
    grid = np.linspace(0, 1, grid_length)
    inputs = np.random.normal(size=(num_samples, grid_length, grid_length))

    return xr.Dataset(
        data_vars={
            "inputs": (("batch", "y", "x"), inputs),
        },
        coords={
            "x": grid,
            "y": grid,
            "batch": np.arange(num_samples),
        },
    )
