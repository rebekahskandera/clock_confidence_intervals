# Clock Confidence Intervals
The primary purpose of this repository is to calculate and plot error bars for fractional frequency deviations given phase or frequency data, including for three-cornered hat measurements.

The Python package is clock_confidence_intervals,
and the Jupyter notebook `ADEV_Error_Bars_Workflow.ipynb` demonstrates in detail
the main uses of the module, including:
- Calculate required sample time to achieve sufficient Chi-squared
distribution degrees of freedom for accurate error bars (specifically for three-cornered hat measurements)
- Print chart of calculated degrees of freedom vs. runtime for each noise type
- Plot three-cornered hat (TCH) or 2-port/reference clock OADEV with Error Bars from clock phase differences
- Print list of calculated error bars/confidence intervals

See `examples/ADEV_Error_Bars_Workflow.ipynb` for a more detailed explanation of the processes
used to make the calculations.

## License

Copyright (C) 2026 Rebekah Skandera

This project is licensed under the GNU Lesser General
Public License v3.0 or later (LGPL-3.0-or-later).

[LGPL v3+ license](https://www.gnu.org/licenses/lgpl-3.0.html)

See LICENSE.txt for details.

## Requirements

- Python 3.10 or newer

## Installation

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/rebekahskandera/three_corner_hat.git"
```

## Example workflow

See [`examples/ADEV_Error_Bars_Workflow.ipynb`](examples/ADEV_Error_Bars_Workflow.ipynb) for a complete workflow using phase-difference data, TCH calculations, runtime planning, tables, and plots.


## Example use of main functions
### Calculate required runtime
```python

from clock_confidence_intervals import required_runtime
tau = 1000  
rate = 1.0
#    None calculates runtime for all noise types.
# alternatively, put power law exponent of dominant noise type
dominant_noise_type = 'all'
relative_sigmas = (0.1, 1,1)    # Relative allan deviations

df = 5  # desired minimum Chi-squared df
target_clock = 1    # which clock's df to solve for (1,2, 3)

alphas = dominant_noise_type

planning = required_runtime(target_df = df, tau=tau, rate=rate, alphas=alphas,
reference_clock=False, target_clock=target_clock, relative_sigmas=relative_sigmas)
print(f"Target: {df} degrees of freedom at tau = {tau} s, rate = {rate} Hz\n")
for r in planning['per_alpha']:
    print(f"{r['noise_type']:<16} N = {r['n_required']:>10,}   "
          f"runtime = {r['runtime_hours']:>10.2f} h   "
          f"({r['tau_multiples']:.0f}x tau)   "
          f"achieved DOF = {r['degrees_of_freedom']:>8.2f}   "
          f"binding: {r['binding_constraint']}")

if len(planning['per_alpha']) > 1:
    w = planning['worst_case']
    print(f"\nWorst case (plan around this if the dominant noise type is unknown): "
          f"{w['noise_type']}, {w['runtime_hours']:.2f} hours")

```
### Plot TCH OADEV with Error Bars using error_bars_tch_list and error_bars_plot
```python
import allantools
from clock_confidence_intervals import (error_bars_tch_list, plot_error_bars)
import matplotlib.pyplot as plt

# Generate simple phase difference data
phase_1 = allantools.noise.white(1000)
phase_2 = allantools.noise.white(1000)
phase_3 = allantools.noise.white(1000)

d12 = phase_1 - phase_2
d23 = phase_2 - phase_3
d31 = phase_3 - phase_1

rate = 1.0
test_taus = 'decade'

#   Defaults to OADEV if no dev_function is entered
error_bars_1, error_bars_2, error_bars_3 = error_bars_tch_list([d12, d23, d31], rate = rate, 
                                                           taus = test_taus, confidence = 0.95,
                                                           dev_function=allantools.oadev, min_df = 10, df_type = "greenhall")

fig, ax = plt.subplots(figsize = (10,6))

#   Clock Names (Clock 1 corresponds to Ch 1, etc.):
clock_1 = 'Clock 1'
clock_2 = 'Clock 2'
clock_3 = 'Clock 3'
plot_error_bars(error_bars_1, error_bars_2, error_bars_3, ax = ax,
                     labels = [clock_1, clock_2, clock_3], title = 'TCH Overlapping Allan Deviation Error Bars', title_font_size = 20,
                     axis_labels= ('Averaging Interval Tau (s)', 'Overlapping Allan Deviation'),
                     axis_label_font_size = 18, axis_tick_font_size = 16)


plt.show()


# After running error_bars_list(), print list of calculated OADEV and confidence intervals
#   Print Tau, ADEV, and Error Bars for Each Clock (after running error_bars_list() function)

def print_error_bar_table(clock_data, clock_name):
    header = f"{'Tau (s)':>14}{'ADEV':>16}{'Lower':>16}{'Upper':>16}"
    print(clock_name)
    print(header)
    print('-' * len(header))
    for tau, dev, (lower, upper), df in clock_data:
        print(f"{tau:>14.4g}{dev:>16.6e}{lower:>16.6e}{upper:>16.6e}")
    print()

print_error_bar_table(error_bars_1, clock_1)
print_error_bar_table(error_bars_2, clock_2)
print_error_bar_table(error_bars_3, clock_3)

```
## Support
Creation of this software was supported in part by DEVCOM Army Research Laboratory.



## References

* **Ekstrom, C. R., & Koppang, P. A. (2006).** Error bars for three-cornered hats. *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 53(5), 876-879.
* **Greenhall, C.A., & Riley, W.J. (2003).** Uncertainty of stability variances based on finite differences. *Proceedings of the 35th Annual Precise Time and Interval Systems and Applications Meeting, San Diego, CA, 2003*, pp. 267-280.
* **Howe, D. A., Allan, D. W., & Barnes, J. A. (1981).** Properties of signal sources and measurement methods. *Proceedings of the 35th Annual Symposium on Frequency Control*, 1-47.
* **Riley, W. J., & Greenhall, C. A. (2004).** Power law noise identification using the lag 1 autocorrelation. *18th European Frequency and Time Forum (EFTF 2004)*, 576-580.

 
