
at pyinla directory 

run

python -m pip install -e .


pyINLA Test Instructions
========================

Prerequisites
-------------
- Python 3.9 or newer available on your PATH (``python``/``python3``).
- Required Python packages installed (``pip install -e .`` in this repository works if you have a pyproject/setup; alternatively install the dependencies you use in development).
- Optional: ``Rscript`` if you want to run the comparison steps that execute the reference INLA code bundled with the tests.
- Optional but recommended: point ``INLA_PATH`` to the bundled binary folder (for Linux this repository ships ``bin/linux/64bit``), or to your own INLA installation.

Running the Full Test Suite
---------------------------
1. From the project root create/activate the Python environment you prefer.
2. Execute:
   ```
   python tests/run_all_tests.py
   ```
   or equivalently ``python -m tests.run_all_tests``.
3. The runner:
   - generates any synthetic data needed for a scenario,
   - runs the matching R-based INLA script when ``Rscript`` is available,
   - runs the pyINLA script(s) for the scenario,
   - compares the outputs (py vs. R) where comparison tooling is present.

Targeting Specific Tests
------------------------
- Supply folder names to restrict the run: ``python tests/run_all_tests.py test1 test4``.
- Use the convenience flags (e.g., ``--test1``, ``--test2``) to select individual suites.
- Add ``--details`` to stream the full stdout/stderr from each step, or ``--details-light`` to print the commands without the full output.
- Add ``--print-pass`` to show command output even when a step succeeds.

Artifacts
---------
- Each scenario writes its outputs under ``tests/test*/py_out`` and ``tests/test*/r_out``.
- If you request comparisons, the runner will try to diff the latest ``inla.model`` folders produced by R and pyINLA.

Troubleshooting
---------------
- Ensure the INLA binary is executable on your platform; adjust ``INLA_PATH`` or update ``PATH`` if the runner cannot locate it.
- Confirm ``Rscript`` is installed if you need reference R runs; otherwise the runner skips that phase.
- Delete or move previous ``inla.model*`` directories inside a scenario folder to avoid comparing against stale results.
