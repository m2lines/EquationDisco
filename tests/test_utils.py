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
