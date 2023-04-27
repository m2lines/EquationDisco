import pyqg
import numpy as np
import xarray as xr
from eqn_disco.hybrid_symbolic import LinearSymbolicRegression, run_gplearn_iteration, hybrid_symbolic_regression
from eqn_disco.utils import FeatureExtractor, example_non_pyqg_data_set

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
    data_set = example_non_pyqg_data_set()

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


def test_hybrid_symbolic_regression():
    data_set = example_non_pyqg_data_set()

    extractor = FeatureExtractor(data_set, example_realspace_input="inputs")

    data_set['target'] = extractor.extract_feature('ddx(inputs)')

    terms, hybrid_regressors = hybrid_symbolic_regression(
        data_set,
        target='target',
        max_iters=2,
        verbose=False,
        base_features=['inputs'],
        base_functions=[],
        spatial_functions=['ddx'],
        population_size=100,
        generations=10,
        metric='mse',
        random_state=42
    )

    assert terms == ['ddx(inputs)']

    regressor = hybrid_regressors[-1]

    assert len(regressor.models) == 1

    np.testing.assert_allclose(regressor.models[0].coef_[0], 1.0)