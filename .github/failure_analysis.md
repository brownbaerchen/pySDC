# Automated Test Failure Analysis

**Generated:** 2026-03-09T06:10:34.804444+00:00
**Workflow Run:** https://github.com/brownbaerchen/pySDC/actions/runs/22840424822

## Summary

- Total Jobs: 30
- Failed Jobs: 1

## Failed Jobs

### 1. user_cpu_tests_linux (base, 3.12)

- **Job ID:** 66245337439
- **Started:** 2026-03-09T05:57:24Z
- **Completed:** 2026-03-09T06:00:33Z
- **Logs:** [View Job Logs](https://github.com/brownbaerchen/pySDC/actions/runs/22840424822/job/66245337439)

#### Error Details

**Error 1:**
```
2026-03-09T05:58:40.8393574Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-3] PASSED [ 15%]
2026-03-09T05:58:40.8649001Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-4] PASSED [ 15%]
2026-03-09T05:58:40.9560570Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:58:40.9843696Z pySDC/test
```

**Error 2:**
```
2026-03-09T05:58:40.8649001Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-4] PASSED [ 15%]
2026-03-09T05:58:40.9560570Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:58:40.9843696Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:58:41.0129166Z pySDC/te
```

**Error 3:**
```
2026-03-09T05:58:40.9560570Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:58:40.9843696Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:58:41.0129166Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:58:41.0411549Z pySDC/
```

**Error 4:**
```
2026-03-09T05:58:40.9843696Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:58:41.0129166Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:58:41.0411549Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-5] FAILED [ 15%]
2026-03-09T05:58:41.0712166Z pySDC/
```

**Error 5:**
```
2026-03-09T05:58:41.0129166Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:58:41.0411549Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-5] FAILED [ 15%]
2026-03-09T05:58:41.0712166Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-GAUSS-2] FAILED [ 15%]
2026-03-09T05:58:41.0999569Z pySDC/tests/
```

## Recommended Actions

1. Review the error messages above
2. Check if this is a known issue in recent commits
3. Review the full logs linked above for complete context
4. Consider if this is related to:
   - Dependency updates (check recent dependency changes)
   - Environment configuration issues
   - Test infrastructure problems
   - Flaky tests that need to be fixed
5. If needed, manually investigate and apply fixes to this PR

## How to Use This PR

This PR was automatically created to help investigate test failures. You can:

- Use this PR to track the investigation
- Add commits with fixes directly to this branch
- Close this PR if the issue is resolved elsewhere
- Convert this to an issue if it needs more discussion
