import itertools
import numpy as np


def lap(arr):
    """Simple brute force, but exact, solution of the linear assignment problem
    (maximization). 

    Enumerates all possible solutions so this has rather bad scaling. Only
    feasible for matrix size nn < 7. 

    | Other solutions: 
    |     Hungarian algorithm
    |         scipy.optimize.linear_sum_assignment
    |         github.com/hrldcpr/hungarian
    |     Jonker-Volgenant
    |         github.com/gatagat/lap
    |         github.com/hrldcpr/pyLAPJV
    |     ... etc etc
        
    Parameters
    ----------
    arr : numpy array, (nn,nn)
        "cost" matrix
    
    Returns
    -------
    irow, icol : index arrays
        same as scipy's version

    Examples
    --------
    # of course 42!
    >>> irow,icol=scipy.optimize.linear_sum_assignment(-arr)
    >>> irow,icol
    (array([0, 1, 2, 3, 4]), array([1, 0, 2, 4, 3]))

    >>> arr[irow,icol].sum()
    42.0

    >>> lap.lap(arr)
    (array([0, 1, 2, 3, 4]), array([1, 0, 2, 4, 3]))
    """
    assert len(set(arr.shape)) == 1, "arr is not square"
    assert (arr >= 0).all(), "arr must be positive"
    nn = arr.shape[0]
    
    # All nn x nn (i,j) pairs
    ijs = itertools.product(range(nn), repeat=2)

    # Brute force search over possible solutions [nn-tuples of (i,j) pairs].
    # Exclude tuples where (i,j)-points are in the same col or row. Loop over
    # combinations avoids storage, which would kill us for nn=6 already, as in 
    #   np.array([x for x in itertools.combinations(ijs, nn)])
    best_max = 0
    best_points = None
    for pp in itertools.combinations(ijs, nn):
        # listcomps are the bottleneck with ~75% of the total runtime, but no
        # other list-based method is faster. For example here
        #   ppl = list(itertools.chain(*pp))
        #   ilst = ppl[::2]
        #   jlst = ppl[1::2]
        # the chain() call is fast (5% of total runtime), but the slicing is
        # slow.
        ilst = [ii[0] for ii in pp]
        jlst = [jj[1] for jj in pp]
        if len(set(ilst)) < nn or len(set(jlst)) < nn:
            continue
        else:
            this_max = arr[ilst, jlst].sum()
            if this_max > best_max:
                best_max = this_max
                best_points = (ilst, jlst)
    return best_points


def argmax(arr):
    """argmax for nd array. Works transparently for masked arrays as well."""
    ##row_idx = arr.max(axis=1).argmax()
    ##col_idx = arr[row_idx,:].argmax()
    ##return row_idx, col_idx 
    return np.unravel_index(arr.argmax(), arr.shape)


def lap_approx(arr):
    """Fast, approximate solution of the linear assignment problem. 
    
    Uses masked arrays to select sub-arrays.
    """
    # Should also work for nd arrays with minor modifications
    assert len(set(arr.shape)) == 1, "arr is not square"
    assert arr.ndim == 2, "arr is not a matrix"
    icol=[]
    irow=[]
    marr = np.ma.array(arr, mask=None) 
    mask = marr.mask.copy()
    for ii in range(arr.shape[0]):
        row_idx, col_idx = argmax(marr)
        icol.append(col_idx)
        irow.append(row_idx)
        mask[row_idx,:] = True
        mask[:,col_idx] = True
        marr = np.ma.array(marr.data, mask=mask)
    return irow, icol


# For profiling only:
#   $ pprofile3 this.py
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        arr = np.loadtxt(sys.argv[1])
    else:
        arr = np.random.rand(5,5)
    print(arr[lap(arr)].sum())
