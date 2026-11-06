import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rwb_forecasting.block_ridge import fit_block_ridge


def toy(seed=42, rows=513, numeric_count=12):
    rng=np.random.default_rng(seed)
    raw=rng.normal(size=(rows,numeric_count))
    raw[:,0]=4.0  # StandardScaler turns this constant column into zeros.
    raw[:,1]=raw[:,2]  # Exact numerical rank deficiency.
    z=StandardScaler().fit_transform(raw)
    categories=np.column_stack([rng.integers(0,17,rows),rng.integers(0,7,rows)])
    c=OneHotEncoder(handle_unknown='ignore').fit_transform(categories)
    # Every complete one-hot group sums to one, making the design rank deficient
    # with the unpenalized intercept even without duplicate numerical features.
    dense=np.column_stack([z,c.toarray()])
    y=dense@rng.normal(size=(dense.shape[1],8))+rng.normal(size=(rows,8))*.05+np.arange(8)*3.7
    return z,c,y


@pytest.mark.parametrize('alpha',[.01,4.,1000.])
def test_coefficients_intercept_and_forecast_match_dense_cholesky(alpha):
    z,c,y=toy()
    actual=fit_block_ridge(z,c,y,alpha)
    expected=Ridge(alpha=alpha,solver='cholesky',fit_intercept=True).fit(np.column_stack([z,c.toarray()]),y)
    np.testing.assert_allclose(actual.coef_,expected.coef_,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(actual.intercept_,expected.intercept_,rtol=1e-8,atol=1e-9)
    # Include unseen categorical inputs represented by all-zero one-hot rows.
    current_z=z[:19]+.137
    current_c=c[:19].copy();current_c[:3]=0
    np.testing.assert_allclose(actual.predict(current_z,current_c),expected.predict(np.column_stack([current_z,current_c.toarray()])),rtol=1e-9,atol=1e-9)


@pytest.mark.parametrize('numeric_count,categorical_count',[(0,12),(12,0),(90,70)])
def test_empty_blocks_and_wide_rank_deficient_design(numeric_count,categorical_count):
    rng=np.random.default_rng(20)
    z=rng.normal(size=(47,numeric_count))
    c=sparse.csr_matrix(rng.integers(0,2,size=(47,categorical_count)))
    y=rng.normal(size=(47,8))
    actual=fit_block_ridge(z,c,y,alpha=4.)
    x=np.column_stack([z,c.toarray()])
    expected=Ridge(alpha=4.,solver='cholesky').fit(x,y)
    np.testing.assert_allclose(actual.coef_,expected.coef_,atol=1e-12)
    np.testing.assert_allclose(actual.predict(z,c),expected.predict(x),atol=1e-12)


def test_all_constant_features_have_zero_coefficients_and_mean_prediction():
    z=np.full((30,4),3.7);c=sparse.csr_matrix(np.ones((30,2)))
    y=np.arange(240,dtype=float).reshape(30,8)
    model=fit_block_ridge(z,c,y,4.)
    np.testing.assert_allclose(model.coef_,0,atol=1e-11)
    np.testing.assert_allclose(model.intercept_,y.mean(axis=0),atol=1e-11)
    np.testing.assert_allclose(model.predict(z,c),np.tile(y.mean(axis=0),(30,1)),atol=1e-11)


def test_realistic_large_target_scale_and_row_permutation():
    z,c,y=toy(seed=10,rows=1001,numeric_count=73)
    y=(y+200)*1e5
    expected=Ridge(alpha=4.,solver='cholesky').fit(np.column_stack([z,c.toarray()]),y)
    actual=fit_block_ridge(z,c,y,4.)
    np.testing.assert_allclose(actual.coef_,expected.coef_,rtol=1e-8,atol=1e-6)
    np.testing.assert_allclose(actual.predict(z[:23],c[:23]),expected.predict(np.column_stack([z[:23],c[:23].toarray()])),rtol=1e-12,atol=1e-6)
    order=np.random.default_rng(12).permutation(len(z))
    shuffled=fit_block_ridge(z[order],c[order],y[order],4.)
    np.testing.assert_allclose(actual.predict(z[:23],c[:23]),shuffled.predict(z[:23],c[:23]),rtol=1e-12,atol=1e-6)


@pytest.mark.parametrize('alpha',[0.,-1.,np.inf,np.nan])
def test_invalid_alpha_rejected(alpha):
    z,c,y=toy()
    with pytest.raises(ValueError,match='alpha'):
        fit_block_ridge(z,c,y,alpha)
