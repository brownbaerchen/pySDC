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
