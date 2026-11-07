import unittest
import numpy as np
from rwb_forecasting.route_reconciliation import reconcile_temporal,ReconciliationConvergenceError

class Tests(unittest.TestCase):
 def assert_margins(self,a,v,z):
  row=a[:,1:].sum(1);column=a[:,1:].sum(0);total=column.sum()
  scaled=v/v.max();alt=scaled/scaled.sum()*total
  np.testing.assert_allclose(z[:,1:].sum(1),.5*row+.5*alt,rtol=1e-10,atol=1e-12)
  np.testing.assert_allclose(z[:,1:].sum(0),column,rtol=1e-10,atol=1e-12)
  np.testing.assert_array_equal(z[:,0].view('u8'),a[:,0].view('u8'))
  self.assertTrue(np.isfinite(z).all());self.assertTrue((z>=0).all())
 def test_dense_margins_and_no_mutation(self):
  rng=np.random.default_rng(21);a=rng.gamma(2.,50.,(1000,8));v=rng.gamma(2.,200.,1000);original=a.copy();alt=v.copy()
  z,receipt=reconcile_temporal(a,v);self.assert_margins(a,v,z)
  np.testing.assert_array_equal(a,original);np.testing.assert_array_equal(v,alt)
  self.assertTrue(receipt['converged'])
 def test_new_row_and_zero_columns_get_feasible_support(self):
  a=np.zeros((4,8));a[:,0]=[1.,2.,3.,4.];a[0,1]=20.;a[1,2]=10.;v=np.array([0.,0.,30.,0.])
  z,r=reconcile_temporal(a,v);self.assert_margins(a,v,z)
  self.assertGreater(z[2,1:3].sum(),0.);np.testing.assert_array_equal(z[:,3:],0.)
  np.testing.assert_array_equal(z[3,1:],0.)
 def test_structural_diagonal_needs_cross_cells(self):
  a=np.zeros((2,8));a[0,1]=1.;a[1,2]=1.;v=np.array([2.,0.])
  z,r=reconcile_temporal(a,v);self.assert_margins(a,v,z);self.assertGreater(z[0,2],.49)
 def test_empty_alternative_zero_total_empty_panel_and_fixed_point(self):
  for a,v in [(np.ones((3,8)),np.zeros(3)),(np.zeros((3,8)),np.ones(3)),(np.empty((0,8)),np.empty(0))]:
   z,r=reconcile_temporal(a,v);np.testing.assert_array_equal(z.view('u8'),a.view('u8'));self.assertEqual(r['iterations'],0)
  a=np.arange(24,dtype=float).reshape(3,8);z,r=reconcile_temporal(a,a[:,1:].sum(1))
  np.testing.assert_array_equal(z.view('u8'),a.view('u8'))
 def test_no_silent_nonconvergence(self):
  a=np.zeros((2,8));a[0,1]=1.;a[1,2]=1.
  with self.assertRaises(ReconciliationConvergenceError) as caught:reconcile_temporal(a,np.array([2.,0.]),max_iter=1)
  self.assertFalse(caught.exception.audit['converged'])
 def test_invalid_inputs(self):
  a=np.ones((2,8))
  for v in [np.ones(3),np.array([-1.,2.]),np.array([np.nan,1.])]:
   with self.assertRaises(ValueError):reconcile_temporal(a,v)
  for a in [np.ones((2,7)),np.ones((2,8),dtype=np.float32),np.full((2,8),-1.)]:
   with self.assertRaises(ValueError):reconcile_temporal(a,np.ones(2))

if __name__=='__main__':unittest.main()
