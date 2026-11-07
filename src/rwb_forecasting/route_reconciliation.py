"""Fixed0.5 row-mass blend with incumbent temporal margins, using NumPy IPF."""
import numpy as np

class ReconciliationConvergenceError(RuntimeError):
    """No partially reconciled forecast is returned when margins fail tolerance."""
    def __init__(self,audit):
        self.audit=audit
        super().__init__(f"IPF failed after {audit['iterations']} iterations; "
                         f"row error={audit['maximum_row_absolute_error']:.6g}, "
                         f"column error={audit['maximum_column_absolute_error']:.6g}")

def reconcile_temporal(incumbent,predicted_row_total,*,seed_fraction=1e-6,
                       max_iter=20000,rtol=1e-10,atol=1e-12):
    """Return `(forecast, audit)`; inputs are never changed.

    `incumbent`: float64(n,8), finite/nonnegative. H1 is copied bitwise.
    `predicted_row_total`: finite/nonnegative(n,), each prediction covers H2-H8.
    Alternate row masses are normalized to the incumbent H2-H8 grand total.
    Desired rows = .5*incumbent rows + .5*normalized alternate rows.
    Desired columns = the incumbent's seven horizon sums.

    Active support means every positive desired row crossed with every positive
    incumbent column, including formerly zero cells. A positive independence
    seed of total mass `seed_fraction` (relative to the grand total) makes this
    support feasible. IPF minimizes KL relative to this seeded matrix subject
    to the margins. If margins do not converge, raise; no silent fallback.

    With no incumbent H2-H8 mass or an all-zero alternative, return an exact
    copy. A matching alternative row profile is also an exact fixed point.
    `atol` is expressed in original forecast units; `rtol` is per-margin.
    """
    a=np.asarray(incumbent);v=np.asarray(predicted_row_total,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=8 or a.dtype!=np.float64:
        raise ValueError('incumbent must have float64 shape(n,8)')
    if v.shape!=(len(a),) or not np.isfinite(a).all() or not np.isfinite(v).all() or (a<0).any() or (v<0).any():
        raise ValueError('Aligned finite nonnegative predictions are required')
    if not np.isfinite(seed_fraction) or seed_fraction<=0 or not np.isfinite(rtol) or rtol<=0 or not np.isfinite(atol) or atol<0:
        raise ValueError('Positive seed/rtol and nonnegative atol required')
    if type(max_iter) is not int or max_iter<1:raise ValueError('max_iter must be positive integer')
    original=a[:,1:];row=original.sum(1);column=original.sum(0);total=float(column.sum())
    if not np.isfinite(total) or not np.isfinite(row).all():raise ValueError('Nonfinite incumbent margin sum')
    audit=dict(shrink=.5,seed_fraction=float(seed_fraction),max_iter=max_iter,rtol=rtol,atol=atol,
               incumbent_h2_h8_total=total,iterations=0,converged=True,h1_bitwise_preserved=True,
               active_rows=0,active_columns=0,maximum_row_absolute_error=0.,maximum_column_absolute_error=0.)
    if total==0 or len(v)==0 or not (v>0).any():
        audit['fallback']='zero_incumbent_total' if total==0 else 'zero_alternative'
        return a.copy(),audit
    # Scaling by max first prevents overflow when summing very large alternatives.
    scaled=v/float(v.max());scaled/=scaled.sum();alternate=scaled*total
    desired=.5*row+.5*alternate
    if np.array_equal(v,row):
        audit['fallback']='already_matching_row_totals'
        return a.copy(),audit
    active_r=desired>0;active_c=column>0
    audit.update(active_rows=int(active_r.sum()),active_columns=int(active_c.sum()),fallback=None)
    # Normalize numerical work to unit grand mass; return physical margins.
    r=desired[active_r]/total;c=column[active_c]/total
    work=original[np.ix_(active_r,active_c)].copy()/total
    seed=seed_fraction*r[:,None]*c[None,:]
    if not (seed>0).all():raise ValueError('Active independence seed underflow; margin dynamic range too large')
    work+=seed
    row_tol=atol/total+rtol*r;column_tol=atol/total+rtol*c
    for iteration in range(1,max_iter+1):
        work*= (r/work.sum(1))[:,None]
        work*= (c/work.sum(0))[None,:]
        er=abs(work.sum(1)-r);ec=abs(work.sum(0)-c)
        if np.all(er<=row_tol) and np.all(ec<=column_tol):break
    else:
        audit.update(converged=False,iterations=max_iter,maximum_row_absolute_error=float(er.max()*total),
                     maximum_column_absolute_error=float(ec.max()*total))
        raise ReconciliationConvergenceError(audit)
    result=np.zeros_like(a);result[:,0]=a[:,0]
    result[:,1:][np.ix_(active_r,active_c)]=work*total
    re=abs(result[:,1:].sum(1)-desired);ce=abs(result[:,1:].sum(0)-column)
    # Check physical output too, rather than relying only on normalized workspace.
    if (not np.isfinite(result).all() or (result<0).any()
        or not np.all(re<=atol+rtol*desired) or not np.all(ce<=atol+rtol*column)):
        audit.update(converged=False,iterations=iteration,maximum_row_absolute_error=float(re.max()),maximum_column_absolute_error=float(ce.max()))
        raise ReconciliationConvergenceError(audit)
    np.testing.assert_array_equal(result[:,0].view('u8'),a[:,0].view('u8'))
    audit.update(iterations=iteration,maximum_row_absolute_error=float(re.max(initial=0)),
                 maximum_column_absolute_error=float(ce.max(initial=0)),output_h2_h8_total=float(result[:,1:].sum()))
    return result,audit
