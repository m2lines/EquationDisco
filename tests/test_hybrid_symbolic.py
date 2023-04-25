import pyqg
import numpy as np
import xarray as xr
from eqn_disco.hybrid_symbolic import LinearSymbolicRegression, run_gplearn_iteration
from eqn_disco.utils import FeatureExtractor

def test_linear_symbolic():
    model = pyqg.QGModel()
    model._step_forward()
    dataset = model.to_dataset()
    extractor = FeatureExtractor(dataset)

    dataset['q_forcing_target'] = dataset.u + dataset.v + 2*extractor.ddx('q')

    parameterization = LinearSymbolicRegression.fit(
        dataset,
        inputs=['u','v','q','ddx(q)','ddy(q)'],
        target='q_forcing_target'
    )

    for lin_reg in parameterization.models:
        np.testing.assert_allclose(lin_reg.coef_, np.array([1, 1, 0, 2, 0]), atol=1e-2)

    preds = parameterization.test_offline(dataset)

    np.testing.assert_allclose(preds.skill.mean(), 1.0, atol=1e-2)

    model2 = pyqg.QGModel(parameterization=parameterization)
    model2._step_forward()

def test_run_gplearn_iteration():
    grid_length = 8
    num_samples = 20
    grid = np.linspace(0, 1, grid_length)
    x, y = np.meshgrid(grid, grid)
    inputs = np.random.normal(size=(num_samples, grid_length, grid_length))
    l = 2 * np.pi * np.append(np.arange(0., grid_length/2), np.arange(-grid_length/2, 0.))
    k = 2 * np.pi * np.arange(0., grid_length/2 + 1)

    data_set = xr.Dataset(
        data_vars=dict(
            inputs=(('batch', 'y', 'x'), inputs),
        ),
        coords=dict(
            x=grid,
            y=grid,
            l=l,
            k=k,
            batch=np.arange(num_samples)
        )
    )

    extractor = FeatureExtractor(data_set, example_realspace_input="inputs")

    target = extractor.extract_feature('ddx(inputs)').data

    regressor = run_gplearn_iteration(
       data_set,
       target,
       base_features=['inputs'],
       base_functions=[],
       spatial_functions=['ddx'],
       population_size=100,
       generations=10,
       metric='mse',
       random_state=42
    )

    result = str(regressor._program)

    assert result == 'ddx(inputs)'
