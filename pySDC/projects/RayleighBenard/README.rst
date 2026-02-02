SDC for Rayleigh-Benard convection
==================================

Benchmarks
----------
The benchmarks use JUBE.
Please run them using commands like

.. code-block:: bash
   module load JUBE
   cd pySDC/projects/RayleighBenard/benchmarks
   OUT=JUSUF_RBC3DG4R4SDC44Ra1e5 jube run jube_script.yaml -t JUSUF RBC3DG4R4SDC44Ra1e5
   jube result bench_run_JUSUF_RBC3DG4R4SDC44Ra1e5 -a > results/JUSUF_RBC3DG4R4SDC44Ra1e5.txt

Use tags `JUSUF` of `BOOSTER` for running on JUSUF or JUWELS booster respectively. The tags for configurations are `RBC3DG4R4SDC44Ra1e5` and `RBC3DG4R4SDC44Ra1e6`

Once you have run all the benchmarks, plot them with

.. code-block:: bash
    cd pySDC/projects/RayleighBenard
    python analysis_scripts/plot_benchmarks.py
