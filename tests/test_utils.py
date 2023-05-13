import pyqg
import numpy as np
import xarray as xr
import eqn_disco.utils as utils

def test_feature_extractor():
    model = pyqg.QGModel()
    model._step_forward()

    extractor1 = utils.FeatureExtractor(model)
    extractor2 = utils.FeatureExtractor(model.to_dataset())
    lap_adv_q1 = extractor1.extract_feature('laplacian(advected(q))')
    lap_adv_q2 = extractor2.extract_feature('laplacian(advected(q))')

    np.testing.assert_allclose(lap_adv_q1, lap_adv_q2.data[0])


def test_parameterization_with_pyqg():
    class MyParameterization(utils.Parameterization):
        @property
        def targets(self):
            return ['q_subgrid_forcing']

        def predict(self, model_or_dataset: utils.ModelLike):
            return dict(q_subgrid_forcing=model_or_dataset.q * 0.0 + 1e-8)

    parameterization = MyParameterization()

    # Test that the parameterization can be used within pyqg
    model = pyqg.QGModel(parameterization=parameterization)
    model._step_forward()

    # Test that pyqg-invoking methods work
    data_set = parameterization.run_online(dt=7200.0, tmax=7200.0, tavestart=0.0)
    assert len(data_set.time) == 1
    assert len(data_set.lev) == 2
    assert len(data_set.x) == len(data_set.y) == 64


def test_parameterization_without_pyqg():
    class MyParameterization(utils.Parameterization):
        @property
        def targets(self):
            return ['target']

        def predict(self, model_or_dataset: utils.ModelLike):
            return dict(
                target=getattr(model_or_dataset, 'inputs') ** 2
            )

    data_set = utils.example_non_pyqg_data_set()

    extractor = utils.FeatureExtractor(data_set)

    data_set['target'] = extractor.extract_feature('inputs') ** 2

    parameterization = MyParameterization()
    result = parameterization.test_offline(data_set)

    np.testing.assert_allclose(result.mse, 0.0)
    np.testing.assert_allclose(result.skill, 1.0)
    np.testing.assert_allclose(result.correlation, 1.0)

    np.testing.assert_allclose(result.target_spatial_mse, np.zeros((8, 8)))
    np.testing.assert_allclose(result.target_spatial_skill, np.ones((8, 8)))
    np.testing.assert_allclose(result.target_spatial_correlation, np.ones((8, 8)))