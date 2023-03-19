import pyqg
import numpy as np
import xarray as xr
from eqn_disco.hybrid_symbolic import LinearSymbolicRegression
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
