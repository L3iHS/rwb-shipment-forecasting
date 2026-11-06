import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData
from rwb_forecasting.boosting import BoostConfig, pooled_features, predict_boosting
from rwb_forecasting.context_features import global_context_frame, halfhour_feature_frame
from rwb_forecasting.halfhour import halfhour_bounds


def panel():
    times=pd.date_range('2025-01-01',periods=500,freq='30min')
    t=np.arange(500)[:,None]
    flow=10000.+30*t+np.arange(4)[None,:]*1000+500*np.sin(t/3)
    flow[100:102]=0
    previous=np.vstack([np.zeros((1,4)),flow[:-1]])
    target=np.round(flow+previous)
    statuses=np.repeat((flow+1000)[:,:,None],6,axis=2)*np.arange(1,7)
    return ForecastData(times,np.array([8,2,90,13]),statuses,target)


def test_rounding_tolerance_preserves_feasible_interval():
    y=np.array([[1.],[3.],[1.]])
    with pytest.raises(ValueError,match='No feasible'):
        halfhour_bounds(y,0.)
    lower,upper=halfhour_bounds(y,.5)
    assert (lower<=upper).all() and (lower>=0).all()
    with pytest.raises(ValueError,match='No feasible'):
        halfhour_bounds(np.array([[1.],[10.],[1.]]),.5)


def test_forward_bounds_never_revisit_past_with_future_zero_anchor():
    y=np.array([[20.],[30.],[20.],[0.],[10.]])
    full=halfhour_bounds(y)
    prefix=halfhour_bounds(y[:2])
    for complete,initial in zip(full,prefix):
        np.testing.assert_array_equal(complete[:2],initial)


def test_global_context_matches_explicit_source_means_and_route_order():
    data=panel();i=360
    frame=global_context_frame(data,np.array([i]))
    scale=max(data.target[i-47:i+1].mean(),1000.)
    np.testing.assert_allclose(frame.global_target_current_ratio,data.target[i].mean()/scale)
    np.testing.assert_allclose(frame.global_status_2_level,data.statuses[i,:,1].mean()/scale)
    for row in range(1,len(data.routes)):
        np.testing.assert_array_equal(frame.iloc[0],frame.iloc[row])
    order=np.array([3,0,2,1])
    shuffled=ForecastData(data.timestamps,data.routes[order],data.statuses[:,order],data.target[:,order])
    pd.testing.assert_frame_equal(frame,global_context_frame(shuffled,np.array([i])))


def test_global_and_halfhour_features_are_source_prefix_invariant():
    data=panel();indices=np.array([360])
    prefix=data.history(data.timestamps[360])
    for function in [global_context_frame,halfhour_feature_frame]:
        pd.testing.assert_frame_equal(function(data,indices),function(prefix,indices))
    complete,scale=pooled_features(data,indices,True,True,True,True)
    truncated,scale2=pooled_features(prefix,indices,True,True,True,True)
    pd.testing.assert_frame_equal(complete,truncated)
    np.testing.assert_array_equal(scale,scale2)


def test_halfhour_midpoint_lags_and_width_align_to_source_route():
    data=panel();indices=np.array([360,370]);scale=np.arange(8,dtype=float)+1000
    lower,upper=halfhour_bounds(data.target)
    features=halfhour_feature_frame(data,indices,scale)
    np.testing.assert_allclose(features.halfhour_midpoint_lag_3,((lower[indices-3]+upper[indices-3])/2).ravel()/scale)
    np.testing.assert_allclose(features.halfhour_width,(upper[indices]-lower[indices]).ravel()/scale)


@pytest.mark.parametrize('halfhour',[False,True])
def test_new_candidate_prediction_contract_and_audit(halfhour):
    data=panel()
    cfg=BoostConfig(normalize=True,advanced=True,direct=True,global_context=True,halfhour=halfhour,max_iter=3,max_leaf_nodes=3)
    prediction,audit=predict_boosting(data,cfg)
    assert prediction.shape==(4,8)
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    assert pd.Timestamp(audit['latest_label_timestamp'])==data.timestamps[-1]
    if halfhour:
        assert audit['earliest_feature_timestamp']==str(data.timestamps[0])
        assert audit['halfhour_rounding_tolerance']==.5


def test_future_poison_does_not_affect_new_feature_blocks():
    data=panel();indices=np.array([360])
    poisoned=ForecastData(data.timestamps,data.routes,data.statuses.copy(),data.target.copy())
    poisoned.statuses[361:]=-1e12
    poisoned.target[361:]=-1e12
    for function in [global_context_frame,halfhour_feature_frame]:
        pd.testing.assert_frame_equal(function(data,indices),function(poisoned,indices))
