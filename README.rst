Two simple toy implementations for solving the linear assignment problem
(maximization).

Some timings in ipython:

.. code-block:: python

    >>> hungarian=scipy.optimize.linear_sum_assignment
    >>> import lap

    >>> arr = np.random.randint(0, 99, (5,5))
    >>>
    >>> %timeit lap.lap(arr)
    99.2 ms ± 635 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    >>> %timeit lap.lap_approx(arr)
    518 µs ± 1.41 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    >>> %timeit hungarian(arr)
    298 µs ± 17.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    >>> # repeat runs
    >>> def run(nn):
    ...     arr = np.random.randint(0, 99, (nn,nn))
    ...     d = {'lap': arr[lap.lap(arr)].sum(),
    ...          'hungarian': arr[hungarian(-arr)].sum(),
    ...          'lap_approx': arr[lap.lap_approx(arr)].sum()}
    ...     print(' '.join(f"{k}: {v}" for k,v in d.items()))
    ...
    >>> for ii in range(10):
    ...     run(5)

To find The Answer To Life, The Universe - And Everything, use

.. code-block:: shell

    $ python3 lap.py arr.txt
