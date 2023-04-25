"""Utility functions."""
import operator
from typing import List, Dict, Union, Any, Tuple, Optional
import re
import pyqg
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ModelLike = Union[pyqg.Model, xr.Dataset]
ArrayLike = Union[np.ndarray, xr.DataArray]
Numeric = Union[ArrayLike, int, float]
StringOrNumeric = Union[str, Numeric]


def ensure_numpy(array: ArrayLike) -> np.ndarray:
    """Helper to ensure that a given array-like input (numpy ndarray or xarray
    DataArray) is converted to numpy format.

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
    else:
        return array


class Parameterization(pyqg.Parameterization):
    """Helper class for defining parameterizations.

    This extends the normal pyqg parameterization framework to handle
    prediction of either subgrid forcings or fluxes, as well as to apply to
    either pyqg.Models orxarray.Datasets.

    """

    @property
    def targets(self) -> List[str]:
        """List the names of the quantities the parameterization predicts.

        Returns
        -------
        List[str]
            List of parameterization targets returned by this parameterization.
            Valid options are "q_forcing_total", "q_subgrid_forcing",
            "u_subgrid_forcing", "v_subgrid_forcing", "uq_subgrid_flux",
            "vq_subgrid_flux", "uu_subgrid_flux", "vv_subgrid_flux", and
            "uv_subgrid_flux". See the dataset description notebook or the
            paper for more details on the meanings of these target fields and
            how they're used.

        """
        raise NotImplementedError

    def predict(self) -> Dict[str, ArrayLike]:
        """Subgrid forcing predictions, as a dictionary of target => array.

        Parameters
        ----------
        model : Union[pyqg.QGModel, xarray.Dataset]
            Model for which we are making subgrid forcing predictions.

        Returns
        -------
        Dict[str, Union[numpy.ndarray, xarray.DataArray]]
            Dictionary of target variable name to subgrid forcing predictions,
            either as numpy arrays (if the model is a pyqg.QGModel) or as
            xarray.DataArrays (if the model is an xarray.Dataset).

        """
        raise NotImplementedError

    @property
    def spatial_res(self) -> int:
        """Spatial res of pyqg.QGModel to which this parameterization applies.

        Currently only supports 64 to replicate the paper, but could be easily
        extended.

        Returns
        -------
        int
            Spatial resolution, i.e. pyqg.QGModel.nx
        """
        return 64  # Future work should generalize this.

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
        if any(q in self.targets[0] for q in ["q_forcing", "q_subgrid"]):
            return "q_parameterization"

        return "uv_parameterization"

    def __call__(
        self, model: ModelLike
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
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
        ensure_array = lambda x: ensure_numpy(x).astype(model.q.dtype)
        preds = self.predict(model)
        keys = list(sorted(preds.keys()))  # these are the same as our targets
        assert keys == self.targets

        # decide how to convert parameterization predictions to the right
        # output format
        if len(keys) == 1:
            # if there's only one target, it's a PV parameterization, and we can
            # just return the array
            return ensure_array(preds[keys[0]])
        elif keys == ["uq_subgrid_flux", "vq_subgrid_flux"]:
            # these are PV subgrid fluxes; we need to take their divergence
            extractor = FeatureExtractor(model)
            return ensure_array(
                extractor.ddx(preds["uq_subgrid_flux"])
                + extractor.ddy(preds["vq_subgrid_flux"])
            )
        elif "uu_subgrid_flux" in keys and len(keys) == 3:
            # these are velocity subgrid fluxes; we need to take two sets of
            # divergences and return a tuple
            extractor = FeatureExtractor(model)
            return (
                ensure_array(
                    extractor.ddx(preds["uu_subgrid_flux"])
                    + extractor.ddy(preds["uv_subgrid_flux"])
                ),
                ensure_array(
                    extractor.ddx(preds["uv_subgrid_flux"])
                    + extractor.ddy(preds["vv_subgrid_flux"])
                ),
            )
        else:
            # this is a "simple" velocity parameterization; return a tuple
            return tuple(ensure_array(preds[k]) for k in keys)

    def run_online(self, sampling_freq: int = 1000, **kwargs) -> xr.Dataset:
        """Initialize and run a parameterized pyqg.QGModel.

        Saves snapshots periodically.

        Parameters
        ----------
        sampling_freq : int
            Number of timesteps (hours) between saving snapshots.
        **kwargs : dict
            Simulation parameters to pass to pyqg.QGModel.

        Returns
        -------
        data_set : xarray.Dataset
            Dataset of parameterized model run snapshots

        """
        # Initialize a pyqg model with this parameterization
        params = dict(kwargs)
        params[self.parameterization_type] = self
        params["nx"] = self.spatial_res
        model = pyqg.QGModel(**params)

        # Run it, saving snapshots
        snapshots = []
        while model.t < m.tmax:
            if model.tc % sampling_freq == 0:
                snapshots.append(model.to_dataset().copy(deep=True))
            model._step_forward()

        data_set = xr.concat(snapshots, dim="time")

        # Diagnostics get dropped by this procedure since they're only present for
        # part of the timeseries; resolve this by saving the most recent
        # diagnostics (they're already time-averaged so this is ok)
        for k, v in snapshots[-1].variables.items():
            if k not in data_set:
                data_set[k] = v.isel(time=-1)

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

            true_centered = truth - truth.mean()
            pred_centered = preds - preds.mean()
            true_var = true_centered**2

            def dims_except(*dims):
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

    def __call__(self, feature_or_features: Union[str, List[str]], flat: bool = False):
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
            res = ensure_numpy(self.extract_feature(feature_or_features))
            if flat:
                res = res.reshape(-1)

        else:
            res = np.array([ensure_numpy(self.extract_feature(f)) for f in feature_or_features])
            if flat:
                res = res.reshape(len(feature_or_features), -1).T
        return res

    def __init__(self, model_or_dataset: ModelLike, example_realspace_input: Optional[str] = None):
        """Build ``FeatureExtractor``."""
        self.m = model_or_dataset
        self.cache = {}

        assert hasattr(self.m, "x"), "dataset must have horizontal realspace dimension"
        assert hasattr(self.m, "k"), "dataset must have horizontal spectral dimension"
        assert hasattr(self.m, "y"), "dataset must have vertical realspace dimension"
        assert hasattr(self.m, "l"), "dataset must have vertical spectral dimension"

        if example_realspace_input is None:
            if hasattr(self.m, "q"):
                example_realspace_input = "q"
            elif isinstance(self.m, xr.Dataset):
                example_realspace_input = next(key for key, val in self.m.items() if 'x' in val.dims)
        self.example_realspace_input = getattr(self.m, example_realspace_input)

        if hasattr(self.m, "_ik"):
            self.ik, self.il = np.meshgrid(self.m._ik, self.m._il)
        elif hasattr(self.m, "fft"):
            self.ik = 1j * self.m.k
            self.il = 1j * self.m.l
        else:
            k, l = np.meshgrid(self.m.k, self.m.l)
            self.ik = 1j * k
            self.il = 1j * l

        self.nx = self.ik.shape[0]
        self.wv2 = self.ik**2 + self.il**2

    # Helpers for taking FFTs / deciding if we need to
    def fft(self, x: ArrayLike) -> ArrayLike:
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
            return self.m.fft(x)
        except AttributeError:
            # if we got an attribute error, that means we have an xarray.Dataset.
            # use numpy FFTs and return a data array instead.
            dims = self.spectral_dims
            coords = dict([(d, self[d]) for d in dims])
            return xr.DataArray(
                np.fft.rfftn(x, axes=(-2, -1)), dims=dims, coords=coords
            )

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return self.example_realspace_input.shape

    @property
    def spatial_dims(self) -> Tuple[str, ...]:
        return self.example_realspace_input.dims

    @property
    def spectral_dims(self) -> Tuple[str, ...]:
        return [dict(y="l", x="k").get(d, d) for d in self.spatial_dims]

    def ifft(self, x: ArrayLike) -> ArrayLike:
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
            return self.m.ifft(x)
        except AttributeError:
            return self.example_realspace_input * 0 + np.fft.irfftn(x, axes=(-2, -1))

    def is_real(self, arr: ArrayLike) -> bool:
        return len(set(arr.shape[-2:])) == 1

    def real(self, feature: StringOrNumeric) -> ArrayLike:
        arr = self[feature]
        if isinstance(arr, float):
            return arr
        if self.is_real(arr):
            return arr
        return self.ifft(arr)

    def compl(self, feature: StringOrNumeric) -> ArrayLike:
        arr = self[feature]
        if isinstance(arr, float):
            return arr
        if self.is_real(arr):
            return self.fft(arr)
        return arr

    # Spectral derivatrives
    def ddxh(self, f: StringOrNumeric) -> ArrayLike:
        return self.ik * self.compl(f)

    def ddyh(self, f: StringOrNumeric) -> ArrayLike:
        return self.il * self.compl(f)

    def divh(self, x: StringOrNumeric, y: StringOrNumeric) -> ArrayLike:
        return self.ddxh(x) + self.ddyh(y)

    def curlh(self, x: StringOrNumeric, y: StringOrNumeric) -> ArrayLike:
        return self.ddxh(y) - self.ddyh(x)

    def laplacianh(self, x: StringOrNumeric) -> ArrayLike:
        return self.wv2 * self.compl(x)

    def advectedh(self, x_: StringOrNumeric) -> ArrayLike:
        x = self.real(x_)
        return self.ddxh(x * self.m.ufull) + self.ddyh(x * self.m.vfull)

    # Real counterparts
    def ddx(self, f: StringOrNumeric) -> ArrayLike:
        return self.real(self.ddxh(f))

    def ddy(self, f: StringOrNumeric) -> ArrayLike:
        return self.real(self.ddyh(f))

    def laplacian(self, x: StringOrNumeric) -> ArrayLike:
        return self.real(self.laplacianh(x))

    def advected(self, x: StringOrNumeric) -> ArrayLike:
        return self.real(self.advectedh(x))

    def curl(self, x: StringOrNumeric, y: StringOrNumeric) -> ArrayLike:
        return self.real(self.curlh(x, y))

    def div(self, x: StringOrNumeric, y: StringOrNumeric) -> ArrayLike:
        return self.real(self.divh(x, y))

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
                [\-\d\.]+
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
            - ``div``, which takes the divergence of the vector field whose x and y components are given by the two expressions, respectively
            - ``curl``, which takes the curl of the vector field whose x and y components are given by the two expressions, respectively

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
        # Helper to recurse on each side of an arity-2 expression
        def extract_pair(s):
            depth = 0
            for i, char in enumerate(s):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    return self.extract_feature(s[:i].strip()), self.extract_feature(
                        s[i + 1 :].strip()
                    )
            raise ValueError(f"string {s} is not a comma-separated pair")

        real_or_spectral = lambda arr: arr + [a + "h" for a in arr]

        if not self.extracted(feature):
            # Check if the feature looks like "function(expr1, expr2)"
            # (better would be to write a grammar + use a parser,
            # but this is a very simple DSL)
            match = re.search(r"^([a-z]+)\((.*)\)$", feature)
            if match:
                op, inner = match.group(1), match.group(2)
                if op in ["mul", "add", "sub", "pow"]:
                    self.cache[feature] = getattr(operator, op)(*extract_pair(inner))
                elif op in ["neg", "abs"]:
                    self.cache[feature] = getattr(operator, op)(
                        self.extract_feature(inner)
                    )
                elif op in real_or_spectral(["div", "curl"]):
                    self.cache[feature] = getattr(self, op)(*extract_pair(inner))
                elif op in real_or_spectral(["ddx", "ddy", "advected", "laplacian"]):
                    self.cache[feature] = getattr(self, op)(self.extract_feature(inner))
                else:
                    raise ValueError(f"could not interpret {feature}")
            elif re.search(f"^[\-\d\.]+$", feature):
                # ensure numbers still work
                return float(feature)
            elif feature == "streamfunction":
                # hack to make streamfunctions work in both datasets & pyqg.Models
                self.cache[feature] = self.ifft(self["ph"])
            else:
                raise ValueError(f"could not interpret {feature}")

        return self[feature]

    def extracted(self, key: str) -> bool:
        return key in self.cache or hasattr(self.m, key)

    # A bit of additional hackery to allow for the reading of features or properties
    def __getitem__(self, attribute: StringOrNumeric) -> Any:
        if isinstance(attribute, str):
            if attribute in self.cache:
                return self.cache[attribute]
            elif re.search(r"^[\-\d\.]+$", attribute):
                return float(attribute)
            else:
                return getattr(self.m, attribute)
        elif any(
            [
                isinstance(attribute, kls)
                for kls in [xr.DataArray, np.ndarray, int, float]
            ]
        ):
            return attribute
        else:
            raise KeyError(attribute)


def energy_budget_term(model, term):
    val = model[term]
    if "paramspec_" + term in model:
        val += model["paramspec_" + term]
    return val.sum("l")


def energy_budget_figure(models, skip=0):
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
