"""
Calculate confidence intervals for reference-clock
and three-cornered hat (TCH) stability measurements.

Includes functions for confidence intervals, noise identificaion,
and equivalent degrees of freedom.

# Copyright (C) 2026 Rebekah Skandera

License
-------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""


from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Callable, Sequence

import allantools
from . import ci
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats as stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Warning configuration and shared constants
# ---------------------------------------------------------------------------

def _compact_warning_format(message, category, filename, lineno, line=None):
    """One-line `file:line: Category: message`, no echoed source line."""
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"



# Formats every warnings.warn() call in this process as a single 
# file:line: Category: message line.
warnings.formatwarning = _compact_warning_format

# Power-law exponent (alpha) to conventional noise-type name.
_NOISE_TYPE_NAMES = {
    2: "White PM",
    1: "Flicker PM",
    0: "White FM",
    -1: "Flicker FM",
    -2: "Random Walk FM",
}

ONE_SIGMA_CI = scipy.special.erf(1 / np.sqrt(2))

# ---------------------------------------------------------------------------
# Error bars functions
# ---------------------------------------------------------------------------

def error_bars(
    phase_data,
    dev_data,
    rate: float = 1.0,
    tau: float = 1.0,
    confidence: float = 0.95,
    dev_function: Callable = allantools.oadev,
    min_df: float = 10,
    min_length: int = 5,
    verbose: bool = True,
    df_fallback: bool = True,
    overlapping_noise_id: bool = False,
    df_type: str = "greenhall",
    data_type: str = "phase",
):
    """Compute reference-clock (non-TCH) deviation error bars for 2-sample
    and 3-sample deviations. Using input phase data and the deviation value, confidence intervals
    are calculated at each tau by identifying the noise type, applying a fallback
    if applicable, and calcualating effective chi-squared degrees of freedom (EDF).

    Parameters
    ----------
    phase_data: numpy.ndarray
        Phase or fractional-frequency data against the reference clock.
    dev_data: float
        Deviation at ``tau``.
    rate, tau, confidence: float, optional
        Sample rate, averaging time, and confidence level.
    dev_function: Callable. Default is allantools.oadev.
    df_fallback: bool, optional
        df_fallback is used when N/m is too small for confidence noise identification.
    overlapping_noise_id: bool, optional
        overlapping_noise_id uses all of the data to produce a noise identification estimate.
    min_df: float, optional
        Minimum remaining degrees of freedom.
    min_length: int, optional
        Minimum decimated length for noise identification.
    verbose: bool, optional
        Whether to emit warnings for noise-id failures
            and low degrees-of-freedom.
    df_type: str, optional
        Degrees-of-freedom calculation method: "greenhall" or "simple".
        Simple valid only for OADEV.
    data_type: {"phase", "freq"}, optional
        Type of input data. Defaults to "phase".

    Returns
    -------
    tuple[float, float, float, float]
        Lower and upper bounds, remaining degrees of freedom, and alpha.
    """
    if data_type not in ("phase", "freq"):
        raise ValueError("data_type must be 'phase' or 'freq'.")
    if data_type == "freq" and overlapping_noise_id:
        raise NotImplementedError(
            "overlapping_noise_id is not implemented for frequency data."
        )
    if tau is None:
        warnings.warn("Tau is None. Did you pass tau value of None?", stacklevel=2)

    ns = len(phase_data)

    # Determine the maximum order of differencing used for noise
    # (2-sample or 3-sample statistic)
    d_max = 3 if (dev_function == allantools.hdev or dev_function == allantools.ohdev) else 2

    tau_0 = 1 / rate
    M = tau / tau_0

    if not np.isclose(M, round(M)):
        raise ValueError(f"Desired tau ({tau}) must be an exact integer multiple of tau_0 ({tau_0}).")
    M = int(round(M))

    # Bool to check if enough values for noise id
    noise_id_length = -(-ns // M) if data_type == "phase" else ns // M
    skip_noise_id = noise_id_length < min_length
    if skip_noise_id:
        if verbose:
            warnings.warn(
            f"N/m is too short for noise identification at tau={tau}. "
            "Defaulting to N/m - 1 degrees of freedom.",
            stacklevel=2,
        )
        rem_df_perfect_ref = ns/M-1
        alpha = np.nan
        dev = dev_data

    if not skip_noise_id:

        dev = dev_data  # 2-port deviation (e.g. OADEV).
        if overlapping_noise_id:
            alpha_est = noise_id_overlapping(phase_data, rate, tau, 0, d_max, round_p=False, min_length=min_length, verbose=verbose) + 2
        else:
            p = noise_id(
                phase_data, 0, d_max, round_p=False, verbose=verbose,
                data_type=data_type, af=M,
            )
            alpha_est = p + 2 if data_type == "phase" else p
        if not np.isfinite(alpha_est):
            if verbose:
                warnings.warn(
                    f"Noise-id estimation failed for tau={tau}. Skipping this tau value.",
                    stacklevel=2,
                )
            return np.nan, np.nan, np.nan, np.nan
        alpha = round(alpha_est)

        if alpha not in (2, 1, 0, -1, -2):
            if verbose:
                warnings.warn(
                    f"Identified alpha not in accepted range (2, 1, 0, -1, -2); defaulting to RWFM for conservative df estimate.",
                    stacklevel=2,
                )
            alpha = -2

        if df_fallback:
            #   Calculate edf_formula with alpha corresponding to smaller d.f.
            #   if sample size too small
            alpha = fallback_alpha(ns, tau, alpha, rate=rate)

        if df_type == 'simple':
            if dev_function != allantools.oadev:
                raise ValueError(f"df_type 'simple' is only valid for function=allantools.oadev; got {dev_function!r}.")
            rem_df_perfect_ref = edf_formula(ns, M, alpha)
        elif df_type == 'greenhall':
            rem_df_perfect_ref = greenhall_by_name(alpha, M, ns, dev_type=dev_function)
        else:
            rem_df_perfect_ref = greenhall_by_name(alpha, M, ns, dev_type=dev_function)
            if verbose:
                warnings.warn(
                    f"Unrecognized df_type {df_type!r}; defaulting to Greenhall method.",
                    stacklevel=2,
                )

    rem_df = rem_df_perfect_ref

    # Alternatively, we could defalt to N/m-1 or RWFM df here (still often < 95% coverage)
    if not np.isfinite(rem_df) or rem_df <= 0:
        if verbose:
            warnings.warn(
                f"Invalid remaining dof ({rem_df}) at tau={tau} \n— the averaging "
                f"factor is likely too large for the data length (m > ns/2). Returning NaN.",
                stacklevel=2,
            )
        return np.nan, np.nan, rem_df, np.nan

    # Alternatively, we could default to N/m-1 or RWFM df here (still often < 95% coverage)
    if rem_df < min_df:
        if verbose:
            warnings.warn(
                f"Remaining degrees of freedom of {rem_df:.1f} is below the minimum df {min_df} "
                f"at tau={tau}. Returning NaN error bars.",
                stacklevel=2,
            )
        return np.nan, np.nan, rem_df, alpha

    if rem_df < 10:
        if verbose:
            warnings.warn(
                f"Only {rem_df:.1f} remaining degrees of freedom (<10) at tau={tau}. \n"
                f"The confidence interval calculation parameters are degraded. Use these bars with caution.",
                stacklevel=2)


    alpha_chi2 = 1 - confidence
    chi2_lower = stats.chi2.ppf(alpha_chi2 / 2, rem_df)
    chi2_upper = stats.chi2.ppf(1 - alpha_chi2 / 2, rem_df)
    dev_lower = (rem_df * dev**2 / chi2_upper) ** (1 / 2)
    dev_upper = (rem_df * dev**2 / chi2_lower) ** (1 / 2)

    return dev_lower, dev_upper, rem_df, alpha

def error_bars_tch(
    phase_data,
    dev_data,
    clock_num: int,
    rate: float = 1.0,
    tau: float = 1.0,
    confidence: float = 0.95,
    dev_function: Callable = allantools.oadev,
    bias_correct: bool = False,
    min_df: float = 10,
    min_length: int = 5,
    verbose: bool = True,
    df_fallback: bool = True,
    overlapping_noise_id: bool = False,
    df_type: str = 'greenhall',
    data_type: str = "phase",
):
    """Compute three-cornered-hat (TCH) deviation error bars. Using input phase data and the deviation value, 
    confidence intervals are calculated at each tau by identifying the noise type, applying a fallback
    if applicable, calcualating reference effective chi-squared degrees of freedom (EDF), and multiplying 
    the calculated EDF by a deviation-specific gamma factor.


    Parameters
    ----------
    phase_data: sequence[numpy.ndarray]
        Three phase or fractional-frequency difference series ``[d12, d23, d31]``.
    dev_data: sequence[float]
        Three TCH deviations ``[sigma_1, sigma_2, sigma_3]`` at ``tau``.
    clock_num: int
        Clock number to calculate (1, 2, or 3).
    rate, tau, confidence: float, optional
        Sample rate, averaging time, and confidence level.
    dev_function: Callable. Default is allantools.oadev.
    bias_correct: bool, optional
    df_fallback: bool, optional
        df_fallback is used when N/m is too small for confidence noise identification.
    overlapping_noise_id: bool, optional
        overlapping_noise_id uses all of the data to produce a noise identification estimate.
    min_df: float, optional
        Minimum remaining degrees of freedom. Defaults to 10.
    min_length: int, optional
        Minimum decimated length for noise identification. Defaults to 5
    verbose: bool, optional
        Whether to emit warnings for noise-id failures
            and low degrees-of-freedom. Defaults to True.
    df_type: str, optional
        Degrees-of-freedom calculation method: "greenhall" or "simple".
        Defaults to "greenhall". Simple valid only for OADEV.
    data_type: {"phase", "freq"}, optional
        Type of input data. Defaults to "phase".


    Returns
    -------
    tuple[float, float, float, float]
        Lower and upper bounds, remaining degrees of freedom, and alpha.
    """
    if data_type not in ("phase", "freq"):
        raise ValueError("data_type must be 'phase' or 'freq'.")
    if data_type == "freq" and overlapping_noise_id:
        raise NotImplementedError(
            "overlapping_noise_id is not implemented for frequency data."
        )

    reference_clock = False
    if tau is None:
        warnings.warn("Tau is None. Did you pass tau value of None?", stacklevel=2)

    if reference_clock:
        ns = len(phase_data)
    else:
        if len(phase_data) != 3:
            raise ValueError(
                f"TCH requires exactly 3 phase-difference series [d12, d23, d31]; got {len(phase_data)}."
            )
        ns = len(phase_data[clock_num - 1]) - 1
        if len(dev_data) != 3:
            raise ValueError(f"TCH requires 3 clock deviations [sigma1, sigma2, sigma3]; got {len(dev_data)}.")

    # Determine the maximum order of differencing used for noise
    # (2-sample or 3-sample statistic)
    d_max = 3 if (dev_function == allantools.hdev or dev_function == allantools.ohdev) else 2
    
    tau_0 = 1 / rate
    M = tau / tau_0

    if not np.isclose(M, round(M)):
        raise ValueError(f"Desired tau ({tau}) must be an exact integer multiple of tau_0 ({tau_0}).")
    M = int(round(M))

    # Bool to check if enough values for noise id
    noise_id_length = -(-ns // M) if data_type == "phase" else ns // M
    skip_noise_id = noise_id_length < min_length
    if skip_noise_id:
        if verbose:
            warnings.warn(
            f"N/m is too short for noise identification at tau={tau}. "
            "Defaulting to N/m - 1 degrees of freedom.",
            stacklevel=2,
        )
        rem_df_perfect_ref = ns/M-1
        alpha = np.nan
        if reference_clock:
            dev = dev_data
        else:
            dev = dev_data[clock_num - 1]

    if not skip_noise_id:
        
        dev = dev_data[clock_num - 1]  # TCH deviation for this clock.

        # For TCH, both clocks are assumed independent with the same
        # noise type, so their phase differences should share a noise type.
        if overlapping_noise_id:
            alpha_1 = noise_id_overlapping(phase_data[clock_num - 1], rate, tau, 0, d_max, round_p=False, min_length=min_length, verbose=verbose) + 2
            alpha_2 = noise_id_overlapping(phase_data[clock_num - 2], rate, tau, 0, d_max, round_p=False, min_length=min_length, verbose=verbose) + 2
        else:
            p_1 = noise_id(
                phase_data[clock_num - 1], 0, d_max, round_p=False,
                verbose=verbose, data_type=data_type, af=M,
            )
            p_2 = noise_id(
                phase_data[clock_num - 2], 0, d_max, round_p=False,
                verbose=verbose, data_type=data_type, af=M,
            )
            phase_offset = 2 if data_type == "phase" else 0
            alpha_1 = p_1 + phase_offset
            alpha_2 = p_2 + phase_offset
        if not np.isfinite(alpha_1) or not np.isfinite(alpha_2):
            if verbose:    
                warnings.warn(
                    f"Noise-id estimation failed for tau={tau}. Skipping this tau value.",
                    stacklevel=2,
                )
            return np.nan, np.nan, np.nan, np.nan

        if abs(alpha_1 - alpha_2) >= 0.99:
            # Alpha estimates disagree: take the noise type corresponding
            # to the fewer degrees of freedom (the more conservative one).
            if verbose:
                warnings.warn(
                    f"Clock differences for {clock_num} vs other clocks have different noise types \n"
                    f"at tau={tau}, with alpha_1={alpha_1} and alpha_2={alpha_2}. Taking noise type "
                    f"corresponding to least degrees of freedom.",
                    stacklevel=2,
                )
            M = int(round(tau * rate))
            df_1 = edf_formula(ns, M, round(alpha_1))
            df_2 = edf_formula(ns, M, round(alpha_2))
            alpha = round(alpha_1 if df_1 <= df_2 else alpha_2)
        else:
            alpha = round((alpha_1 + alpha_2) / 2)

        if alpha not in (2, 1, 0, -1, -2):
            if verbose:
                warnings.warn(
                    f"Identified alpha not in accepted range (2, 1, 0, -1, -2); defaulting to RWFM for conservative df estimate.",
                    stacklevel=2,
                )
            alpha = -2

        if df_fallback:
            #   Calculate edf_formula with alpha corresponding to smaller d.f.
            #   if sample size too small
            alpha = fallback_alpha(ns, tau, alpha, rate=rate)

        if df_type == 'greenhall':
            rem_df_perfect_ref = greenhall_by_name(alpha, M, ns, dev_type=dev_function)
        elif df_type == 'simple':
            if dev_function != allantools.oadev:
                raise ValueError(f"df_type 'simple' is only valid for function=allantools.oadev; got {dev_function!r}.")
            rem_df_perfect_ref = edf_formula(ns, M, alpha)
        else:
            rem_df_perfect_ref = greenhall_by_name(alpha, M, ns, dev_type=dev_function)
    gamma = None


    sigmas = np.asarray(dev_data, dtype=float)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
        if verbose:    
            warnings.warn(
                f"TCH deviations contain a negative or non-finite value ({sigmas}) at tau={tau}: \n"
                f"Returning NaN error bars.",
                stacklevel=2,
            )
        return np.nan, np.nan, np.nan, alpha
    gamma = gamma_factor(clock_num, sigmas[0], sigmas[1], sigmas[2])
    rem_df = rem_df_perfect_ref * gamma

    # Alternatively, we could defalt to N/m-1 or RWFM df here (still often < 95% coverage)
    if not np.isfinite(rem_df) or rem_df <= 0:
        if verbose:
            warnings.warn(
                f"Invalid remaining dof ({rem_df}) at tau={tau} for clock {clock_num} \n— the averaging "
                f"factor is likely too large for the data length (m > ns/2). Returning NaN.",
                stacklevel=2,
            )
        return np.nan, np.nan, rem_df, np.nan

    # Alternatively, we could defalt to N/m-1 or RWFM df here (still often < 95% coverage)
    if not reference_clock and rem_df < min_df:
        if verbose:
            warnings.warn(
                f"Remaining degrees of freedom of {rem_df:.1f} is below the minimum df {min_df} "
                f"at tau={tau} for clock {clock_num}. Returning NaN error bars.",
                stacklevel=2,
            )
        return np.nan, np.nan, rem_df, alpha

    if rem_df < 10:
        if verbose:
            warnings.warn(
                f"Only {rem_df:.1f} remaining degrees of freedom (<10) at tau={tau} for clock {clock_num} \n"
                f"The confidence interval calculation parameters are degraded, \n"
                f"and the TCH may return a negative variance. Use these bars with caution.",
                stacklevel=2)
        #       and the deviation is biased ~{0.15 / rem_df:.1%}. "
        #     f"Use these bars with caution.",
        #     skip_file_prefixes=(__file__,),
        # )

    # Bias correction is useful for TCH estimates with poor degrees of freedom
    if not reference_clock and bias_correct:
        dev = dev / math.sqrt(1 + 0.3 / rem_df)

    alpha_chi2 = 1 - confidence
    chi2_lower = stats.chi2.ppf(alpha_chi2 / 2, rem_df)
    chi2_upper = stats.chi2.ppf(1 - alpha_chi2 / 2, rem_df)
    dev_lower = (rem_df * dev**2 / chi2_upper) ** (1 / 2)
    dev_upper = (rem_df * dev**2 / chi2_lower) ** (1 / 2)

    return dev_lower, dev_upper, rem_df, alpha

def error_bars_list(
    phase_data,
    rate: float = 1.0,
    dev_data=None,
    taus=None,
    confidence: float = 0.95,
    dev_function: Callable = allantools.oadev,
    min_df: float = 10,
    min_length: int = 5,
    verbose: bool = True,
    df_fallback: bool = True,
    overlapping_noise_id: bool = False,
    df_type: str = "greenhall",
    data_type: str = "phase",
):
    """Compute reference-clock (non-TCH) deviation error bars over many taus. 
    See :func:`error_bars` for details on the error bars calculations.

    Parameters
    ----------
    phase_data: numpy.ndarray
        Phase or fractional-frequency data against the reference clock.
    dev_data: numpy.ndarray or None, optional
        Precomputed deviations corresponding to ``taus``.
    taus: array-like or None, optional
        Averaging times to calculate.
    rate, confidence: float, optional
        Sample rate and confidence level.
    dev_function: Callable. Default is allantools.oadev.
    df_fallback: bool, optional
        df_fallback is used when N/m is too small for confidence noise identification.
    overlapping_noise_id: bool, optional
        overlapping_noise_id uses all of the data to produce a noise identification estimate.
    dev_function: callable, str, or None, optional
        Deviation function to use when ``dev_data`` is not supplied. Defaults to "oadev".
    min_df: float, optional
        Minimum remaining degrees of freedom.
    min_length: int, optional
        Minimum decimated length for noise identification.
    verbose: bool, optional
        Whether to print noise-type changes and emit warnings for noise-id failures
            and low degrees-of-freedom.
    df_type: str or None, optional
        Degrees-of-freedom calculation method.
    data_type: {"phase", "freq"}, optional
        Type of input data. Defaults to "phase".

    Returns
    -------
    list[tuple[float, float, tuple[float, float], float]]
        Error-bar results as ``(tau, deviation, (lower, upper), dof)`` tuples.
    """
    if data_type not in ("phase", "freq"):
        raise ValueError("data_type must be 'phase' or 'freq'.")
    if data_type == "freq" and overlapping_noise_id:
        raise NotImplementedError(
            "overlapping_noise_id is not implemented for frequency data."
        )
    if dev_function is None:
        dev_function = allantools.oadev
    elif isinstance(dev_function, str):
        try:
            dev_function = getattr(allantools, dev_function)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown deviation function name: {dev_function!r}. "
                f"Expected a callable from allantools."
            ) from exc

    if dev_data is None:
        taus, dev_data, _, _ = dev_function(
            data=phase_data, rate=rate, data_type=data_type, taus=taus
        )

    order = np.argsort(taus)
    taus = taus[order]
    dev_data = dev_data[order]

    confidence_intervals = []
    current_alpha = None
    for i, dev in enumerate(dev_data):
        M = int(round(taus[i] * rate))

        n = len(phase_data)

        dev_lower, dev_upper, df, alpha = error_bars(
            phase_data,
            dev,
            rate=rate,
            tau=taus[i],
            confidence=confidence,
            dev_function=dev_function,
            min_df=min_df,
            min_length=min_length,
            verbose=verbose,
            df_fallback=df_fallback,
            overlapping_noise_id=overlapping_noise_id,
            df_type=df_type,
            data_type=data_type,
        )
        if np.isnan(dev_lower) or np.isnan(dev_upper):
            warnings.warn(
                f"Skipping tau={taus[i]} because error bars could not be computed.",
                stacklevel=2,
            )
            continue

        if np.isfinite(alpha) and alpha != current_alpha:
            if verbose:
                print(
                    f"Noise type changed to {string_noise_type(round(alpha))} at tau = {taus[i]}"
                    f" for 2-port deviation."
                )
            current_alpha = alpha

        confidence_intervals.append((taus[i], dev, (dev_lower, dev_upper), df))
    return confidence_intervals

def three_cornered_hat(
    data_ab,
    data_bc,
    data_ca,
    rate,
    taus,
    function,
    data_type: str = "phase",
):
    """Compute one TCH deviation series for phase or frequency data.

    Parameters
    ----------
    data_ab, data_bc, data_ca: numpy.ndarray
        Pairwise clock-difference data.
    rate: float
        Sampling rate in Hz.
    taus: array-like
        Averaging times to evaluate.
    function: Callable
        AllanTools deviation function.
    data_type: {"phase", "freq"}, optional
        Type of input data. Defaults to "phase".

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Tau values and TCH deviations for clock A.
    """
    if data_type not in ("phase", "freq"):
        raise ValueError("data_type must be 'phase' or 'freq'.")
    tau_ab, dev_ab, _, _ = function(
        data_ab, data_type=data_type, rate=rate, taus=taus
    )
    _, dev_bc, _, _ = function(
        data_bc, data_type=data_type, rate=rate, taus=taus
    )
    _, dev_ca, _, _ = function(
        data_ca, data_type=data_type, rate=rate, taus=taus
    )
    var_a = 0.5 * (dev_ab**2 + dev_ca**2 - dev_bc**2)
    var_a[var_a < 0] = 0
    return tau_ab, np.sqrt(var_a)


def error_bars_tch_list(
    phase_data,
    rate: float = 1.0,
    taus=None,
    confidence: float = 0.95,
    dev_function: Callable = allantools.oadev,
    bias_correct: bool = False,
    min_df: float = 10,
    min_length: int = 5,
    verbose: bool = True,
    df_fallback: bool = True,
    overlapping_noise_id: bool = False,
    df_type: str = "greenhall",
    data_type: str = "phase",
):
    """Compute three-cornered-hat (TCH) deviation error bars over many taus.
    See :func:`error_bars_tch` for details on the error bars calculations.


    Parameters
    ----------
    phase_data: sequence[numpy.ndarray]
        Three phase or fractional-frequency difference series ``[d12, d23, d31]``.
    taus: array-like or None, optional
        Averaging times to calculate.
    rate, confidence: float, optional
        Sample rate and confidence level.
    dev_function: Callable. Default is allantools.oadev.
    bias_correct: bool, optional
    df_fallback: bool, optional
        df_fallback is used when N/m is too small for confidence noise identification.
    overlapping_noise_id: bool, optional
        overlapping_noise_id uses all of the data to produce a noise identification estimate.
    dev_function: callable, str, or None, optional
        Deviation function used for TCH deviations. Defaults to "oadev".
    min_df: float, optional
        Minimum remaining degrees of freedom.
    min_length: int, optional
        Minimum decimated length for noise identification.
    verbose: bool, optional
        Whether to print noise-type changes and emit noise-id failures
        and low degrees-of-freedom warnings.
    df_type: str or None, optional
        Degrees-of-freedom calculation method.
    data_type: {"phase", "freq"}, optional
        Type of input data. Defaults to "phase".

    Returns
    -------
    tuple[list, list, list]
        One list of ``(tau, deviation, (lower, upper), dof)`` tuples per clock.
    """
    if data_type not in ("phase", "freq"):
        raise ValueError("data_type must be 'phase' or 'freq'.")
    if data_type == "freq" and overlapping_noise_id:
        raise NotImplementedError(
            "overlapping_noise_id is not implemented for frequency data."
        )
    if dev_function is None:
        dev_function = allantools.oadev
    elif isinstance(dev_function, str):
        try:
            dev_function = getattr(allantools, dev_function)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown deviation function name: {dev_function!r}. "
                f"Expected a callable from allantools."
            ) from exc
    # Three-cornered hat deviations for each clock.
    taus_1, dev_1 = three_cornered_hat(
        phase_data[0], phase_data[1], phase_data[2], rate, taus,
        dev_function, data_type=data_type,
    )
    taus_2, dev_2 = three_cornered_hat(
        phase_data[1], phase_data[2], phase_data[0], rate, taus,
        dev_function, data_type=data_type,
    )
    taus_3, dev_3 = three_cornered_hat(
        phase_data[2], phase_data[0], phase_data[1], rate, taus,
        dev_function, data_type=data_type,
    )
    dev_data = [dev_1, dev_2, dev_3]
    taus_by_clock = [taus_1, taus_2, taus_3]

    confidence_intervals_by_clock = ([], [], [])
    for clock_num in (1, 2, 3):
        clock_taus = taus_by_clock[clock_num - 1]
        intervals = confidence_intervals_by_clock[clock_num - 1]
        current_alpha = None

        order = np.argsort(clock_taus)
        clock_taus = clock_taus[order]
        dev_data[clock_num - 1] = dev_data[clock_num - 1][order]


        for i, dev in enumerate(dev_data[clock_num - 1]):
            M = int(round(clock_taus[i] * rate))
            n = len(phase_data[0])

            dev_lower, dev_upper, rem_df, alpha = error_bars_tch(
                phase_data,
                [dev_data[0][i], dev_data[1][i], dev_data[2][i]],
                clock_num=clock_num,
                rate=rate,
                tau=clock_taus[i],
                confidence=confidence,
                dev_function=dev_function,
                bias_correct=bias_correct,
                min_df=min_df,
                min_length=min_length,
                verbose=verbose,
                df_fallback=df_fallback,
                overlapping_noise_id=overlapping_noise_id,
                df_type=df_type,
                data_type=data_type,
            )
            if np.isnan(dev_lower) or np.isnan(dev_upper):
                warnings.warn(
                    f"Skipping tau={clock_taus[i]} for clock {clock_num} because "
                    f"error bars could not be computed.",
                    stacklevel=2,
                )
                continue

            if np.isfinite(alpha) and alpha != current_alpha:
                if verbose:
                    print(
                        f"Noise type for Clock {clock_num} at tau = {clock_taus[i]} changed to "
                        f"{string_noise_type(round(alpha))}."
                    )
                current_alpha = alpha

            intervals.append((clock_taus[i], dev, (dev_lower, dev_upper), rem_df))

    return confidence_intervals_by_clock

# ---------------------------------------------------------------------------
# Post-error-bar processing
# ---------------------------------------------------------------------------

def print_error_bars_table(
    error_bars,
    clock_name: str | None = None,
    deviation_name: str = "OADEV",
    output_file: str | Path | None = None,
):
    """Print a table of error-bar results and optionally write it to a file.

    Parameters
    ----------
    error_bars: list[tuple[float, float, tuple[float, float], float]]
        Tuples of ``(tau, deviation, (lower, upper), degrees_of_freedom)``.
    clock_name: str, optional
        Clock name included in the deviation column headers.
    deviation_name: str, optional
        Deviation name included in the column headers.
    output_file: str or pathlib.Path, optional
        Text file to write the formatted table to.

    Returns
    -------
    pandas.DataFrame
        Table containing the error-bar results.
    """
    label = f"{clock_name} {deviation_name}" if clock_name else deviation_name
    lower_column = f"{label} (lower)"
    upper_column = f"{label} (upper)"
    rows = [
        {
            "Tau (s)": tau,
            lower_column: lower,
            upper_column: upper,
            "Degrees of Freedom": degrees_of_freedom,
        }
        for tau, _dev, (lower, upper), degrees_of_freedom in error_bars
    ]
    results_table = pd.DataFrame(
        rows,
        columns=["Tau (s)", lower_column, upper_column, "Degrees of Freedom"],
    )
    table_text = results_table.to_string(
        index=False,
        formatters={
            "Tau (s)": lambda value: f"{value:.10g}",
            lower_column: lambda value: f"{value:.2e}",
            upper_column: lambda value: f"{value:.2e}",
            "Degrees of Freedom": lambda value: f"{value:.2f}",
        },
    )
    print(table_text)
    if output_file is not None:
        Path(output_file).write_text(table_text + "\n", encoding="utf-8")
    return results_table

def string_noise_type(exponent: int) -> str:
    """Return the noise-type name for a power law exponent.

    Parameters
    ----------
    exponent: Power law exponent; must be one of -2, -1, 0, 1, 2.

    Returns
    -------
    The noise-type name (e.g. "White PM") or invalid message.
    """
    if exponent not in _NOISE_TYPE_NAMES:
        return "Invalid exponent for noise type. Power law exponent must be -2, -1, 0, 1, or 2."
    return _NOISE_TYPE_NAMES[exponent]

def error_bars_dict(error_bars):
    """Create a dictionary mapping `tau` values to corresponding error bars.

    Parameters
    ----------
    error_bars: Output of :func:`error_bars_list` (a list of
        `(tau, dev, (lower, upper), df)` tuples)

    Returns
    -------
    dict of {tau: ((lower, upper), df)}: A dictionary mapping `tau` values to corresponding error bars.
    """
    by_tau = {}
    for tau, dev, bars, df in error_bars:
        by_tau[tau] = (bars, df)
    return by_tau

def non_overlapping_tau(error_bars_1, error_bars_2):
    """Find tau values where two sets of error bars do not overlap.

    Parameters
    ----------
    error_bars_1: Output of :func:`error_bars_list` (a list of
        `(tau, dev, (lower, upper), df)` tuples) or of
        :func:`error_bars_dict` (a `{tau: ((lower, upper), df)}` dict).
    error_bars_2: Same shape as `error_bars_1`, for the series to
        compare against.

    Returns
    -------
    A list of tau values for which the error bars do not overlap.
    """
    if isinstance(error_bars_1, list):
        error_bars_1 = error_bars_dict(error_bars_1)
    if isinstance(error_bars_2, list):
        error_bars_2 = error_bars_dict(error_bars_2)

    non_overlapping = []
    for tau, ((lower_1, upper_1), _), in error_bars_1.items():
        (lower_2, upper_2), _ = error_bars_2[tau]
        if lower_1 > upper_2 or upper_1 < lower_2:
            non_overlapping.append(tau)
    return non_overlapping

def plotallan_phase(plt, y, rate, taus, style):
    """Plot OADEV computed from a phase time series.

    Parameters
    ----------
    plt: A `matplotlib.pyplot`-compatible plotting interface.
    y: Phase time series.
    rate: Sampling rate of the data, in Hz.
    taus: Averaging times to evaluate, or an allantools tau selector
        (e.g. `'all'`).
    style: Matplotlib style string, e.g. `'bo'`.

    Returns
    -------
    None. The OADEV series is added to `plt`.
    """
    t2, ad, ade, adn = allantools.oadev(y, rate=rate, taus=taus)
    plt.loglog(t2, ad, style, label="OADEV from phase data")

def plotallan_frequency(plt, y, rate, taus, style):
    """Plot OADEV computed from a frequency time series.

    Parameters
    ----------
    plt: A `matplotlib.pyplot`-compatible plotting interface.
    y: Frequency time series.
    rate: Sampling rate of the data, in Hz.
    taus: Averaging times to evaluate, or an allantools tau selector
        (e.g. `'all'`).
    style: Matplotlib style string, e.g. `'r.'`.

    Returns
    -------
    None. The OADEV series is added to `plt`.
    """
    t2, ad, ade, adn = allantools.oadev(y, rate=rate, data_type="freq", taus=taus)
    plt.loglog(t2, ad, style)

def plot_error_bars(
    error_bars_1,
    error_bars_2=None,
    error_bars_3=None,
    ax=None,
    labels=None,
    colors=None,
    axis_labels = None,
    axis_label_font_size = None, 
    axis_tick_font_size = None,
    capsize=3,
    marker="o",
    linestyle="-",
    title=None,
    title_font_size = None
):
    """Plot tau vs. deviation with error bars using output of :func:`error_bars_list`.

    Parameters
    ----------
    error_bars_1: List of `(tau, dev, (lower, upper), df)` tuples for
        clock 1, as returned by :func:`error_bars_list`.
    error_bars_2: Same shape, for clock 2. Omitted if None.
    error_bars_3: Same shape, for clock 3. Omitted if None.
    ax: Matplotlib Axes to plot on. Defaults to the current axes.
    labels: Legend labels, one per clock. Defaults to
        `['Clock 1', 'Clock 2', 'Clock 3']`.
    colors: Line/marker colors, one per clock. Defaults to
        `['r', 'g', 'b']`.
    axis_labels: Labels for the x and y axes. Defaults to 
        (xlabel, ylabel) = `('Tau (s)', 'Deviation')`.
    axis_label_font_size: Font size for axis labels. Defaults to 16.
    axis_tick_font_size: Font size for axis tick labels. Defaults to 12.
    capsize: Error bar cap size, in points.
    marker: Matplotlib marker style for the deviation points.
    linestyle: Matplotlib line style connecting the points.
    title: Optional plot title.
    title_font_size: Font size for the title. Defaults to 16 if title is provided.

    Returns
    -------
    The matplotlib Axes instance that was plotted on.
    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = ["Clock 1", "Clock 2", "Clock 3"]
    if colors is None:
        colors = ["r", "g", "b"]
    if axis_labels is None:
        axis_labels = ("Tau (s)", "Deviation")
    if axis_label_font_size is None:
        axis_label_font_size = 16
    if axis_tick_font_size is None:
        axis_tick_font_size = 12
    if title is not None:
        if title_font_size is None:
            title_font_size = 16
    

    series = [(error_bars_1, labels[0], colors[0])]
    if error_bars_2 is not None:
        series.append((error_bars_2, labels[1], colors[1]))
    if error_bars_3 is not None:
        series.append((error_bars_3, labels[2], colors[2]))

    prepared = []
    for entries, label, color in series:
        sorted_entries = sorted(entries, key=lambda entry: entry[0])
        taus = np.asarray([entry[0] for entry in sorted_entries], dtype=float)
        dev = np.asarray([entry[1] for entry in sorted_entries], dtype=float)
        lower = np.asarray([entry[2][0] for entry in sorted_entries], dtype=float)
        upper = np.asarray([entry[2][1] for entry in sorted_entries], dtype=float)

        #   Drop NaN/non-positive points here
        valid = np.isfinite(taus) & (taus > 0) & np.isfinite(dev) & (dev > 0)
        taus, dev, lower, upper = taus[valid], dev[valid], lower[valid], upper[valid]
        yerr = np.vstack((dev - lower, upper - dev))

        prepared.append((taus, dev, yerr, label, color))

    if all(taus.size == 0 for taus, dev, yerr, label, color in prepared):
        ax.text(
            0.5, 0.5, "No valid ADEV calculated", ha="center", va="center",
            transform=ax.transAxes, fontsize=axis_label_font_size,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title, fontsize=title_font_size)
        return ax

    for taus, dev, yerr, label, color in prepared:
        if taus.size == 0:
            continue
        ax.errorbar(
            taus, dev, yerr=yerr, fmt=marker, linestyle=linestyle,
            color=color, ecolor=color, capsize=capsize, label=label,
        )

    xlabel = axis_labels[0]
    ylabel = axis_labels[1]
    ax.tick_params(axis = 'both', labelsize = axis_tick_font_size)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)
    ax.set_ylabel(ylabel, fontsize = axis_label_font_size)
    if title is not None:
        ax.set_title(title, fontsize = title_font_size)
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    return ax

# ---------------------------------------------------------------------------
# Noise identification functions
# ---------------------------------------------------------------------------

def noise_id(
    z,
    dmin: int = 0,
    dmax: int = 2,
    round_p: bool = True,
    verbose: bool = True,
    *,
    data_type: str | None = None,
    af: int = 1,
):
    """Estimate the dominant power-law noise exponent via lag-1 autocorrelation function algorithm.

    Implementation of the Lag 1 Autocorrelation Power Law Noise
    Identification Algorithm from Riley and Greenhall (2004).

    Parameters
    ----------
    z: Input data. When ``data_type`` is None, preprocessed data.
    dmin: Minimum order of differencing.
    dmax: Maximum order of differencing (2 or 3 for Allan/Hadamard
        2- or 3-sample variance, respectively).
    round_p: Whether to round the estimated exponent `p` to the
        nearest integer.
    verbose: Whether to emit insufficient sample size warnings.
    data_type: {"phase", "freq"} or None, optional
        Type of raw input data. When provided, data is preprocessed before
        noise identification. Defaults to None.
    af: int, optional
        Averaging factor used for preprocessing. Defaults to 1.

    Returns
    -------
    Estimate for the power-law noise exponent `p`, where
    `alpha = p + 2` for phase data or `alpha = p` for frequency data.
    """
    z = np.asarray(z, dtype=float)
    if data_type is not None:
        if data_type not in ("phase", "freq"):
            raise ValueError("data_type must be 'phase', 'freq', or None.")
        if not np.isclose(af, round(af)) or af < 1:
            raise ValueError("af must be a positive integer.")
        af = int(round(af))
        if data_type == "phase":
            z = z[::af]
            if len(z) < 3:
                if verbose:
                    warnings.warn(
                        "Noise-id estimation did not receive sufficient sample size. "
                        "Returning NaN.",
                        stacklevel=2,
                    )
                return np.nan
            z = detrend(z, deg=2)
        else:
            z = z[:len(z) - (len(z) % af)]
            z = z.reshape((-1, af)).mean(axis=1) if len(z) else z
            if len(z) < 2:
                if verbose:
                    warnings.warn(
                        "Noise-id estimation did not receive sufficient sample size. "
                        "Returning NaN.",
                        stacklevel=2,
                    )
                return np.nan
            z = detrend(z, deg=1)

    # catch empty or small arrays
    if len(z) < 2:
        if verbose:
            warnings.warn(
                f"Noise-id estimation did not receive sufficient sample size. "
                f"Returning NaN.",
                stacklevel=2,
            )
        return np.nan
 
    d = 0
    p = None
    done = False
    
    while not done:
        z_bar = np.mean(z)
        numerator = np.sum((z[:-1] - z_bar) * (z[1:] - z_bar))
        denominator = np.sum((z - z_bar) ** 2)

        r1 = numerator / denominator if denominator != 0 else 0
        delta = r1 / (1 + r1) if (1 + r1) != 0 else float("inf")

        if d >= dmin and (delta < 0.25 or d >= dmax):
            n=len(z)
            #   compare to bias-corrected autocorr
            # r1_classes = np.array([0.0, -1/3, -1/2])
            # r1_bias = np.array([1.0/n, (-0.735+2.75/n)/n,
            #                      (-1.007+1.31/n)/n])
            # k = np.argmin(np.abs(r1-(r1_classes-r1_bias)))
            # r1_corr = r1 - r1_bias[k]
            # delta = r1_corr/(1+r1_corr)
            p = -2 * (delta + d)
            done = True
        else:
            z = np.diff(z)
            d += 1

    return (round(p) if round_p else p)

def decimate_phase_data_overlapping(phase_data, tau_0: float, tau: float, min_length: int = 5, verbose: bool = True):
    """Decimate phase data into overlapping, shifted arrays at every offset.

    Used by :func:`noise_id_overlapping` to make use of nearly all the
    available data, rather than a single non-overlapping decimation.

    Parameters
    ----------
    phase_data: Raw phase data array.
    tau_0: Original sampling interval of the data.
    tau: Desired averaging interval.
    min_length: Minimum length for a decimated slice to be kept.
    verbose: Whether to emit insufficient sample size warnings.

    Returns
    -------
    A list of decimated phase arrays, one per starting offset, with
    any slice shorter than `min_length` dropped. Empty if none of the
    M possible offsets produce a long enough slice.

    Raises
    ------
    ValueError
        If tau is not an exact integer multiple of tau_0.
    """
    M = tau / tau_0
    # np.isclose avoids floating point precision issues (e.g. 0.05 / 0.01 = 4.9999999).
    if not np.isclose(M, round(M)):
        raise ValueError(f"Desired tau ({tau}) must be an exact integer multiple of tau_0 ({tau_0}).")
    M = int(round(M))

    phase_array = np.asarray(phase_data)
    N = len(phase_array)

    # Check that input data length meets required min_length
    longest_possible = -(-N // M)  # ceil(N/M)
    if longest_possible < min_length:
        if verbose:
            warnings.warn(
                f"No decimated slice can reach min_length={min_length} at tau={tau} "
                f"(N={N}, M={M}, longest possible slice={longest_possible}). \n"
                f"Skipping this tau value for noise identification.",
                stacklevel=7,
            )
        return []

    decimated_arrays = [
        phase_array[start::M] for start in range(M) if len(phase_array[start::M]) >= min_length
    ]

    if not decimated_arrays:
        if verbose:
            warnings.warn(
                f"No decimated slice reaches min_length={min_length} at tau={tau} \n"
                f"(N={N}, M={M}, slice length ≈ {N // M}). \n"
                f"Skipping this tau value for noise identification.",
                stacklevel=7,
            )
        return []

    return decimated_arrays

def _noise_id_batch(z, dmin: int = 0, dmax: int = 2):
    """Vectorized :func:`noise_id`: estimate p for every row of z (shape (K, L)) at once.

    Same lag-1-autocorrelation algorithm as :func:`noise_id`, batched across rows for efficiency.
    Used by :func:`noise_id_overlapping`.

    Parameters
    ----------
    z: Batch of equal-length rows, shape (K, L).
    dmin: Minimum order of differencing.
    dmax: Maximum order of differencing.

    Returns
    -------
    Array of shape (K,) with the unrounded p estimate for each row.
    """
    z = np.asarray(z, dtype=float)
    K = z.shape[0]
    L=z.shape[1]
    qualifies = np.empty((dmax + 1, K), dtype=bool)
    deltas = np.empty((dmax + 1, K), dtype=float)

    for d in range(dmax + 1):
        z_bar = z.mean(axis=1, keepdims=True)
        numerator = np.sum((z[:, :-1] - z_bar) * (z[:, 1:] - z_bar), axis=1)
        denominator = np.sum((z - z_bar) ** 2, axis=1)

        r1 = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=r1, where=denominator != 0)

        #   Apply minimal bias correction +1/n
        # r1 = r1+1.0/z.shape[1]

        denom2 = 1 + r1
        delta = np.full_like(r1, np.inf)
        np.divide(r1, denom2, out=delta, where=denom2 != 0)

        deltas[d] = delta
        qualifies[d] = (d >= dmin) & ((delta < 0.25) | (d >= dmax))
        if d < dmax:
            z = np.diff(z, axis=1)

    chosen_d = np.argmax(qualifies, axis=0)  # first qualifying stage per row
    chosen_delta = deltas[chosen_d, np.arange(K)]

    # #   Apply minimal bias correction +1/n
    # chosen_r1=chosen_delta / (1-chosen_delta)
    # chosen_r1 = chosen_r1 + 1.0 / (L-chosen_d)
    # chosen_delta = chosen_r1 / (1+chosen_r1)

    return (-2 * (chosen_delta + chosen_d))

def noise_id_overlapping(phase_data, rate: float, tau: float, dmin: int = 0, dmax: int = 2, round_p: bool = True,
                          min_length: int = 5, verbose: bool = True):
    """Estimate the dominant noise exponent using overlapping decimated data.

    Uses M decimated slices from :func:`decimate_phase_data_overlapping`, where 
    M = tau / tau_0, and averages each slice noise id calculated with :func:`_noise_id_batch`.
    Adjustment to :func:`noise_id` to use all available data.
    
    Parameters
    ----------
    phase_data: Raw phase data array (not decimated).
    rate: Sampling rate of the data, in Hz (tau_0 = 1 / rate).
    tau: Averaging interval to decimate to.
    dmin: Minimum order of differencing (default 0).
    dmax: Maximum order of differencing (2 or 3 for Allan/Hadamard
        2- or 3-sample variance, respectively).
    round_p: Whether to round the estimated value `p` to the nearest
        integer.
    min_length: Minimum length for a decimated slice to be used.

    Returns
    -------
    Estimate for the power-law noise exponent `p` (`alpha = p + 2`
    for phase data), or NaN if no valid estimate could be produced.

    Warns
    -----
    UserWarning
        If a decimated slice has fewer than 2 points, that slice is skipped;
        a normal (non-NaN) estimate is still returned from the remaining
        slices, if any are valid.
    UserWarning
        If no valid decimated arrays or estimates exist, warning is emitted.

    Raises
    ------
    ValueError
        If `tau` is not an exact integer multiple of `tau_0` (propagated
        from `decimate_phase_data_overlapping`).
    """
    tau_0 = 1 / rate
    decimated_data = decimate_phase_data_overlapping(phase_data, tau_0, tau, min_length, verbose=verbose)
    if len(decimated_data) == 0:
        if verbose:
            warnings.warn(
                f"No valid overlapping decimated arrays available for tau={tau}. Skipping noise-id estimation.",
                stacklevel=3,
            )
        return np.nan

    by_length = {}
    for array in decimated_data:
        array = np.asarray(array, dtype=float)
        if array.size < 2:
            if verbose:
                warnings.warn(
                    f"Decimated data length must be >=2 for noise identification; got {array.size}. "
                    f"Skipping this decimated array.",
                    stacklevel=3,
                )
            continue
        by_length.setdefault(array.size, []).append(array)

    if not by_length:
        if verbose:
            warnings.warn(
                f"No valid noise-id estimates were produced for tau={tau}. Skipping this tau value.",
                stacklevel=3,
            )
        return np.nan

    p_estimates = [_noise_id_batch(np.stack(arrays), dmin, dmax) for arrays in by_length.values()]
    p = np.mean(np.concatenate(p_estimates))
    return (round(p) if round_p else p)

def decimate_phase_data(phase_data, rate: float = 1.0, tau: float | None = None, min_length: int | None = None):
    """Decimate phase data to a desired averaging interval, keeping every Mth sample.

    Non-overlapping decimation, sufficient for noise identification. Call
    before :func:`noise_id`.

    Parameters
    ----------
    phase_data: Raw phase data array.
    rate: Sampling rate of the data, in Hz (tau_0 = 1 / rate).
    tau: Desired averaging interval, e.g. 3 seconds. Defaults to tau = 1/rate.
    min_length: Minimum length for a slice to be returned. 
        If None min_length is not checked.

    Returns
    -------
    The decimated phase data as an `np.ndarray`. 

    Raises
    ------
    ValueError
        If tau is not an exact integer multiple of tau_0.
    """

    tau_0 = 1 / rate
    if tau == None:
        tau = tau_0
    M = tau / tau_0

    if not np.isclose(M, round(M)):
        raise ValueError(f"Desired tau ({tau}) must be an exact integer multiple of tau_0 ({tau_0}).")
    M = int(round(M))

    if min_length is not None and -(-len(phase_data) // M) < min_length:  # ceil(len(phase_data)/M)
        warnings.warn(
            f"Decimated slice length ({-(-len(phase_data) // M)}) is below min_length ({min_length}) "
            f"at tau={tau}. Skipping this tau value for noise identification.",
            stacklevel=3,
        )
        return np.array([])
    return np.asarray(phase_data)[::M]

# ---------------------------------------------------------------------------
# Degrees of freedom calculation functions
# ---------------------------------------------------------------------------

def fallback_alpha(N, tau, alpha, rate=1.0):
    """Return the conservative fallback noise exponent, when applicable.

    Parameters
    ----------
    N: Number of phase-difference samples.
    m: Averaging factor.
    alpha: Estimated power-law noise exponent.

    Returns
    -------
    The original or conservatively substituted noise exponent.
    """

    m = int(round(tau * rate))
    if alpha not in [2, 1, 0, -1, -2]:
        return -2

    if alpha == 1 or alpha == 2:
        if N / m <= 80 and tau > 10:
            return 0
    return alpha

def greenhall_by_name(alpha, af, N, dev_type="adev"):
    """Return Greenhall equivalent degrees of freedom for ``dev_type``.

    Parameters
    ----------
    alpha: Power-law noise exponent.
    af: Averaging factor.
    N: Number of phase-difference samples.
    dev_type: Deviation type: `adev` or allantools.adev, `oadev` or allantools.oadev,
      `mdev` or allantools.mdev, `tdev` or allantools.tdev, `hdev` or allantools.hdev,
        or `ohdev` or allantools.ohdev.

    Returns
    -------
    The Greenhall equivalent degrees of freedom.

    Raises
    ------
    NotImplementedError
        If `dev_type` is not supported.
    """
    if dev_type == "adev" or dev_type == allantools.adev:
        return ci.edf_greenhall(
            alpha=round(alpha), d=2, m=af, N=N, overlapping=False, modified=False
        )
    if dev_type == "oadev" or dev_type == allantools.oadev:
        return ci.edf_greenhall(
            alpha=round(alpha), d=2, m=af, N=N, overlapping=True, modified=False
        )
    if dev_type == "mdev" or dev_type == "tdev" or dev_type == allantools.mdev or dev_type == allantools.tdev:
        return ci.edf_greenhall(
            alpha=round(alpha), d=2, m=af, N=N, overlapping=True, modified=True
        )
    if dev_type == "hdev" or dev_type == allantools.hdev:
        return ci.edf_greenhall(
            alpha=round(alpha), d=3, m=af, N=N, overlapping=False, modified=False
        )
    if dev_type == "ohdev" or dev_type == allantools.ohdev:
        return ci.edf_greenhall(
            alpha=round(alpha), d=3, m=af, N=N, overlapping=True, modified=False
        )
    raise NotImplementedError

def edf_formula(N, M: float, alpha: int):
    """Compute equivalent degrees of freedom for a clock vs. a perfect reference.

    Fixes typos present in `allantools.edf_simple()` and Howe (1981), 
    "Properties of Signal Sources and Measurement Methods".

    Parameters
    ----------
    N: Number of phase difference samples. May be a length-1
        array/list, which is normalized to a scalar.
    M: Averaging factor (tau / tau_0). Must be an integer, or very
        close to one.
    alpha: Power-law exponent of the frequency PSD:
        2 = White PM, 1 = Flicker PM, 0 = White FM, -1 = Flicker FM,
        -2 = Random Walk FM. Unrecognized values default to N/M - 1.

    Returns
    -------
    The equivalent degrees of freedom for the given noise type.

    Raises
    ------
    ValueError
        If M is not (approximately) an integer.
    """
    if not np.isclose(M, round(M)):
        raise ValueError("Desired tau must be an exact integer multiple of tau_0.")

    # Normalize N
    if isinstance(N, np.ndarray) and N.size == 1:
        N =  N.item()
    elif isinstance(N, (list, tuple)) and len(N) == 1:
        N =  N[0]

    m = int(round(M))

    if alpha == 2:  # White PM
        return (N + 1) * (N - 2 * m) / (2 * (N - m))
    if alpha == 0:  # White FM
        return (3 * (N - 1) / (2 * m) - 2 * (N - 2) / N) * (4 * m**2) / (4 * m**2 + 5)
    if alpha == 1:  # Flicker PM
        a = (N - 1) / (2 * m)
        b = (2 * m + 1) * (N - 1) / 4
        return math.exp(math.sqrt(math.log(a) * math.log(b)))
    if alpha == -1:  # Flicker FM
        if m == 1:
            return 2 * (N - 2) ** 2 / (2.3 * N - 4.9)
        return 5 * N**2 / (4 * m * (N + 3 * m))
    if alpha == -2:  # Random Walk FM
        return ((N - 2) / m) * ((N - 1) ** 2 - 3 * m * (N - 1) + 4 * m**2) / (N - 3) ** 2
    # Default. Close to RWFM df. Assumes close to no additional information gained from overlapping ADEV
    return N/M - 1  

def gamma_factor(i: int, sigma_1: float, sigma_2: float, sigma_3: float) -> float:
    """Compute the TCH gamma factor for clock `i` (1, 2, or 3).

    The adjusted remaining degrees of freedom for a TCH confidence
    interval is the product of :func:`gamma_factor` and :func:`edf_formula`.

    Parameters
    ----------
    i: Which clock (1, 2, or 3) to compute the gamma factor for.
    sigma_1: Deviation of clock 1.
    sigma_2: Deviation of clock 2.
    sigma_3: Deviation of clock 3.

    Returns
    -------
    The gamma factor for clock `i`.

    Raises
    ------
    ValueError
        If `i` is not 1, 2, or 3.
    """
    sigmas = (sigma_1, sigma_2, sigma_3)
    if i not in (1, 2, 3):
        raise ValueError("Invalid clock index. Must be 1, 2, or 3.")
    sigma_i = sigmas[i - 1]
    return (2 * sigma_i**4) / (
        2 * sigma_i**4 + sigma_1**2 * sigma_2**2 + sigma_1**2 * sigma_3**2 + sigma_2**2 * sigma_3**2
    )


def detrend(x, deg=1):
    """
    remove polynomial from data
    used in :func:`noise_id`

    Parameters
    ----------
    x: numpy.array
        time-series
    deg: int
        degree of polynomial to remove from x

    Returns
    -------
    z: numpy.array
        detrended time-series
    """
    t = range(len(x))
    p = np.polyfit(t, x, deg)
    residual = x - np.polyval(p, t)
    return residual
# ---------------------------------------------------------------------------
# Experiment planning
# ---------------------------------------------------------------------------

def _required_n_for_dof(target_df: float, m: int, alpha: int, gamma: float = 1.0, starting_value: int = None) -> int:
    """Smallest integer N such that :func:`edf_formula` times gamma is at least target_df.

    Assumes :func:`edf_formula` is non-decreasing in N (true for all
    noise types for m large enough and N large enough) (checked for m>100 and
    N>4m for RWFM, other don't require large values)

    Parameters
    ----------
    target_df: Desired minimum degrees of freedom.
    m: Averaging factor.
    alpha: Power-law noise exponent.
    gamma: TCH gamma-factor multiplier; 1.0 for the reference-clock case.
    starting_value: Optional starting value for N, assumed above domain floor
    of :func:`edf_formula`. If None, function will search for domain
    floor and start from there.

    Returns
    -------
    The smallest valid N above starting_valuemeeting the target.

    Raises
    ------
    RuntimeError
        If no N up to an internal search cap satisfies the target, or
        bisection fails to converge.
    """

    if starting_value is not None:
        lo = starting_value
    else:
        # Find domain limit
        if alpha in (2, 1):
                algebraic = 2 * m + 1
        #   For alpha = 0, domain limit < 1.34*m will always be dominated
        #  by need for > 2 samples in noise_id
        elif alpha == -1 and m == 1:
            algebraic = 3  # need denominator > 0
        else:
            algebraic = 4
        domain_floor = max(m + 1, algebraic)
        
        lo = domain_floor


    if edf_formula(lo, m, alpha) * gamma >= target_df:
        return lo

    # Exponential search for a valid upper bound that meets target df
    hi = lo
    search_cap = max(lo * 1000, 1e13)
    while True:
        hi = min(hi * 2, search_cap)
        try:
            value = edf_formula(hi, m, alpha) * gamma
        except (ValueError, ZeroDivisionError):
            value = float("nan")
        if np.isfinite(value) and value > 0 and value >= target_df:
            break
        if hi >= search_cap:
            raise RuntimeError(
                f"No N up to {search_cap:,}*tau_0 reaches target_df={target_df} "
                f"for alpha={alpha}, m={m}."
            )

    # Integer bisection for smallest N in (lo, hi] with target df
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            value = edf_formula(mid, m, alpha) * gamma
        except (ValueError, ZeroDivisionError):
            value = float("nan")
        if np.isfinite(value) and value > 0 and value >= target_df:
            hi = mid
        else:
            lo = mid + 1

    # Ensure lo meets target df due to floating point inaccuracy
    n = lo
    for _ in range(10):
        if edf_formula(n, m, alpha) * gamma >= target_df:
            return n
        n += 1
    raise RuntimeError(
        f"Verification failed to converge for target_df={target_df}, alpha={alpha}, "
        f"m={m} after bisection + 10 nudges."
    )

def required_runtime(
    target_df: float,
    tau: float,
    rate: float,
    alphas: int | Sequence[int] | str = "all",
    reference_clock: bool = True,
    target_clock: int = 1,
    relative_sigmas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_length: int = 5,
):
    """Estimate the runtime needed to reach a target DOF at a given tau.

    Solves the inverse of :func:`edf_formula`: given minimum
    degrees-of-freedom, finds the smallest sample count N (and corresponding
    runtime) that reaches it, subject to two additional constraints:
    the noise-id step's data-sufficiency wall, and each df formula's
    own domain. Applies only to OADEV case.

    This does NOT include a safety margin for noise-id misidentification or
    formula approximation error at low DOF.

    Parameters
    ----------
    target_df: Desired minimum remaining degrees of freedom.
    tau: Averaging interval of interest, in seconds.
    rate: Sampling rate, in Hz.
    alphas: Which noise type(s) to solve for. A single alpha, a sequence
        of alphas, or "all" (default) for all 5 known types
        (2, 1, 0, -1, -2).
    reference_clock: If True (default), solve for the reference-clock
        (2-port) case. If False, solve for TCH, applying the gamma
        factor for `target_clock` given `relative_sigmas`.
    target_clock: Which clock (1, 2, or 3) to solve for in TCH mode.
        Ignored if reference_clock is True.
    relative_sigmas: Relative noise levels of the three clocks in TCH
        mode, e.g. (1.0, 2.0, 2.0) if clock 1 is expected to be twice
        as stable as clocks 2 and 3. Only the ratios matter (:func:`gamma_factor`
        is scale-invariant); defaults to the symmetric case. Ignored if
        reference_clock is True.
    min_length: Minimum decimated-slice length the noise-id step needs;
        must match whatever :func:`error_bars` or :func:`error_bars_list` will actually
        be called with later.

    Returns
    -------
    A dict with:
        "per_alpha": a list of dicts, one per requested alpha, each with
            "alpha", "noise_type", "n_required", "runtime_seconds",
            "runtime_hours", "binding_constraint" (one of "dof",
            "noise_id_wall", "domain"), "tau_multiples"
            (runtime_seconds / tau), and "degrees_of_freedom" -- the DOF
            actually achieved at n_required
            (the result of :func:`edf_formula` times gamma). Equal to
            target_df when binding_constraint is "dof"; larger than
            target_df when the noise-id wall or domain floor pushed
            n_required above what the DOF target alone required.
        "worst_case": the entry from "per_alpha" with the largest
            n_required -- the one to plan around if the true noise type
            is unknown.

    Raises
    ------
    ValueError
        If target_df <= 0 or an unrecognized alpha is requested.
    RuntimeError
        If the per-alpha DOF target can't be verified, or if
        :func:`_required_n_for_dof` can't find a valid N up to its internal search
        cap or fails to converge.
    """
    if target_df <= 0:
        raise ValueError(f"target_df must be positive; got {target_df}.")

    if isinstance(alphas, str):
        if alphas != "all":
            raise ValueError(f"Unrecognized alphas string {alphas!r}; expected 'all' or a list of alphas.")
        alpha_list = list(_NOISE_TYPE_NAMES.keys())
    elif isinstance(alphas, int):
        alpha_list = [alphas]
    else:
        alpha_list = list(alphas)

    for a in alpha_list:
        if a not in _NOISE_TYPE_NAMES:
            raise ValueError(f"Unrecognized alpha {a}; expected one of {_NOISE_TYPE_NAMES.keys()}.")

    tau_0 = 1 / rate
    m = tau / tau_0
    # use integer multiple of tau_0
    m = int(round(m))

    gamma = 1.0 if reference_clock else gamma_factor(target_clock, *relative_sigmas)
    n_noise_id = m * (min_length - 1) + 1  # smallest N with ceil(N/m) >= min_length

    per_alpha = []
    for alpha in alpha_list:

        # Find domain limit
        if alpha in (2, 1):
                algebraic = 2 * m + 1
        #   For alpha = 0, domain limit < 1.34*m will always be dominated
        #  by need for > 2 samples in noise_id
        elif alpha == -1 and m == 1:
            algebraic = 3  # need denominator > 0
        else:
            algebraic = 4
        domain_floor = max(m + 1, algebraic)
        
        n_domain = domain_floor
        # while True:
        #     try:
        #         value = edf_formula(n, m, alpha)
        #     except (ValueError, ZeroDivisionError):
        #         value = float("nan")
        #     if np.isfinite(value) and value > 0:
        #         n_domain = n
        #         break
        #     n += 1


        n_dof = _required_n_for_dof(target_df, m, alpha, gamma=gamma, starting_value = n_domain)
        n_required = max(n_dof, n_noise_id, n_domain)

        # A noise_id/domain floor above n_dof is sufficient
        # under monotonicity, but confirm and allow for floating point innacuracy 
        for _ in range(10):
            if edf_formula(n_required, m, alpha) * gamma >= target_df:
                break
            n_required += 1
        else:
            raise RuntimeError(f"Verification failed to converge for alpha={alpha}, m={m}.")

        if n_required == n_dof:
            binding = "dof"
        elif n_required == n_noise_id:
            binding = "noise_id_wall"
        else:
            binding = "domain"

        runtime_seconds = n_required / rate
        per_alpha.append(dict(
            alpha=alpha,
            noise_type=string_noise_type(alpha),
            n_required=n_required,
            runtime_seconds=runtime_seconds,
            runtime_hours=runtime_seconds / 3600,
            binding_constraint=binding,
            tau_multiples=runtime_seconds / tau,
            degrees_of_freedom=edf_formula(n_required, m, alpha) * gamma,
        ))

    worst_case = max(per_alpha, key=lambda r: r["n_required"])
    return dict(per_alpha=per_alpha, worst_case=worst_case)
