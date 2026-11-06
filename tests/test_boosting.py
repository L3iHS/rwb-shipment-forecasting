import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData,HORIZON
from rwb_forecasting.boosting import BoostConfig, pooled_features, predict_boosting


def panel():
    times=pd.date_range('2025-01-01',periods=500,freq='30min')
    target=np.arange(500,dtype=float)[:,None]*100+np.arange(3)[None,:]*10+1000
    status=np.repeat(target[:,:,None],6,axis=2)*np.arange(1,7)
    return ForecastData(times,np.array([9,1,20]),status,target)


def test_source_features_ignore_every_later_observation():
    data=panel(); indices=np.array([360])
    for normalize in (False,True):
        before,scale=pooled_features(data,indices,normalize)
        after,scale2=pooled_features(data.history(data.timestamps[360]),indices,normalize)
        pd.testing.assert_frame_equal(before,after)
        np.testing.assert_array_equal(scale,scale2)


def test_seasonal_lags_align_to_future_step_and_route():
    data=panel();indices=np.array([360,370])
    features,_=pooled_features(data,indices)
    for step in range(1,9):
        block=features.iloc[(step-1)*6:step*6]
        np.testing.assert_array_equal(block['aligned_target_48'],data.target[indices+step-48].ravel())
        np.testing.assert_array_equal(block['aligned_target_336'],data.target[indices+step-336].ravel())
        np.testing.assert_array_equal(block['horizon'],np.full(6,step))


def test_normalization_is_source_causal_and_reversible():
    data=panel();indices=np.array([360])
    raw,scales=pooled_features(data,indices)
    norm,_=pooled_features(data,indices,True)
    np.testing.assert_allclose(norm['target_lag_0']*scales,raw['target_lag_0'],rtol=1e-6)
    np.testing.assert_allclose(norm['status_2_mean_6']*scales,raw['status_2_mean_6'],rtol=1e-6)


@pytest.mark.parametrize('normalize,loss,calibrate',[(False,'squared_error',False),(True,'squared_error',False),(False,'absolute_error',True)])
def test_prediction_contract_and_label_cutoff(normalize,loss,calibrate):
    data=panel()
    config=BoostConfig(normalize=normalize,loss=loss,calibrate=calibrate,max_iter=3,max_leaf_nodes=3)
    pred,audit=predict_boosting(data,config)
    assert pred.shape==(3,HORIZON)
    assert np.isfinite(pred).all() and (pred>=0).all()
    assert pd.Timestamp(audit['latest_label_timestamp'])<=data.timestamps[-1]
    assert pd.Timestamp(audit['last_source_timestamp'])<=data.timestamps[-9]
    assert audit['fit_early_stopping'] is False


def test_insufficient_history_is_rejected():
    data=panel()
    with pytest.raises(ValueError,match='Insufficient'):
        predict_boosting(data.history(data.timestamps[100]))


def test_advanced_profiles_and_status_lags_are_causal():
    data=panel(); indices=np.array([360,370])
    features,_=pooled_features(data,indices,False,True)
    for step in range(1,9):
        block=features.iloc[(step-1)*6:step*6]
        expected=np.stack([data.target[indices+step-day*48].ravel() for day in range(1,8)]).mean(axis=0)
        np.testing.assert_allclose(block['aligned_daily_mean_7'],expected,rtol=1e-6)
        np.testing.assert_array_equal(block['status_2_lag_12'],data.statuses[indices-12,:,1].ravel())
    complete,_=pooled_features(data,np.array([360]),True,True)
    prefix,_=pooled_features(data.history(data.timestamps[360]),np.array([360]),True,True)
    pd.testing.assert_frame_equal(complete,prefix)


def test_direct_residual_prediction_contract():
    data=panel()
    config=BoostConfig(normalize=True,advanced=True,direct=True,residual=True,max_iter=3,max_leaf_nodes=3)
    pred,audit=predict_boosting(data,config)
    assert pred.shape==(3,HORIZON)
    assert np.isfinite(pred).all() and (pred>=0).all()
    assert pd.Timestamp(audit['latest_label_timestamp'])==data.timestamps[-1]


def test_calibration_zero_predictions_remains_finite():
    data=panel();data.target[:]=0
    config=BoostConfig(calibrate=True,max_iter=3,max_leaf_nodes=3)
    pred,audit=predict_boosting(data,config)
    np.testing.assert_array_equal(pred,np.zeros((3,8)))
    assert audit['calibration_factors']==[1.]*8
