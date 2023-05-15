from typing import Literal

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize


def vectorized_outlier_detection(
    data: np.ndarray, calibdata: np.ndarray, mad_ini_0: float, min_mad: float
) -> np.ndarray:
    # data : Original data to filter.
    #           Single column: the data to be filtered
    # calibdata : Data used to calibrate the filter.
    #           Single column: the data used for calibration.

    # OUTPUT:
    # The Output are explained in each function below:

    # ###############Automatic calibration of some paramseters.#################

    lambda_z, lambda_mad, mad_ini, min_mad = lambda_determination(calibdata, mad_ini_0)

    # #######################Find the outliers#################################
    newdata = vectorized_ewma(data, lambda_z, lambda_mad, mad_ini, min_mad)
    return newdata


def calibrate_ewma_model(
    data, mad_ini_0, calibrate_min_mad: bool
) -> tuple(float, float, float, float):
    # New optimization procedure implemented in alpha_z_determination which
    # does both lambda_z and lambda_MAD together

    # This function takes a data time serie and auto-calibrate the outlier
    # filter. The dataset is assumed to be of good quality (no fault sensor and
    # limited amount of outliers) and without significant gaps (missing time
    # steps or NaN).
    # This smoother is highly inspired from the textbook Introduction to
    # Statistical Quality Control, 6th edition by Douglas Montgomery, section
    # 10.4.2
    # DATA :        The input data matrice of format Nx2. First column is a
    #               date vector compliant with Matlab internal representation
    #               of date (use of DATENUM returns the proper format). The
    #               second column is the signal to be treated.
    # PARAM :       A structure containing all parameters (general and
    #               advanced) to be set by the user. See "DefaultParam.m"
    lambda_z, lambda_mad, mad_ini, min_mad = calibrate_ewma_model(
        data, mad_ini_0, calibrate_min_mad
    )

    return lambda_z, lambda_mad, mad_ini, min_mad


@njit
def objFun_alpha_z(log_alpha_z: np.ndarray, db: np.ndarray) -> float:
    log_alpha_z_unwrapped: float = log_alpha_z[0]
    size = db.shape[0]
    alpha_z = np.exp(-(log_alpha_z_unwrapped**2))
    z = np.zeros([size, 3])  # Smoothed values
    a = np.zeros(
        size,
    )  # Model parameter values
    b = np.zeros(
        size,
    )  # Model parameter values
    c = np.zeros(
        size,
    )  # Model parameter values
    forecast = np.zeros(
        size,
    )  # Forecast values

    # Model initialisation
    z[1, 0] = db[0]
    z[1, 1] = db[0]
    z[1, 2] = db[0]

    for i in range(1, size):
        z[i, 0] = alpha_z * db[i] + (1 - alpha_z) * z[i - 1, 0]
        z[i, 1] = alpha_z * z[i, 0] + (1 - alpha_z) * z[i - 1, 1]
        z[i, 2] = alpha_z * z[i, 1] + (1 - alpha_z) * z[i - 1, 2]

    a[1:] = 3 * z[1:, 0] - 3 * z[1:, 1] + z[1:, 2]
    b[1:] = (alpha_z / (2 * (1 - alpha_z) ** 2)) * (
        (6 - 5 * alpha_z) * z[1:, 0]
        - 2 * (5 - 4 * alpha_z) * z[1:, 1]
        + (4 - 3 * alpha_z) * z[1:, 2]
    )
    c[1:] = (alpha_z / (1 - alpha_z)) ** 2 * (z[1:, 0] - 2 * z[1:, 1] + z[1:, 2])

    forecast[2:] = a[1:-1] + b[1:-1] + 0.5 * c[1:-1]

    # square error between the measured  and the forecast values
    square_err = (db - forecast) ** 2

    rmse_z = np.sqrt((square_err / size).sum())  # Calculation of the RMSE
    return rmse_z


@njit
def objFun_alpha_MAD(
    log_alpha_MAD: np.ndarray, alpha_z: np.ndarray, db: np.ndarray
) -> float:
    log_alpha_mad_unwrapped: float = log_alpha_MAD[0]
    alpha_z_unwrapped: float = alpha_z[0]
    size = db.shape[0]

    alpha_MAD = np.exp(-(log_alpha_mad_unwrapped**2))

    z = np.zeros([size, 3])  # Smoothed values
    a = np.zeros(size)  # Model parameter values
    b = np.zeros(size)  # Model parameter values
    c = np.zeros(size)  # Model parameter values
    forecast = np.zeros(size)  # Forecast values
    MAD = np.zeros(size)
    forecast_MAD = np.zeros(size)

    # Model initialisation
    z[0, 0] = db[0]
    z[0, 1] = db[0]
    z[0, 2] = db[0]

    z[1:, 0] = alpha_z_unwrapped * db[1:] + (1 - alpha_z_unwrapped) * z[:-1, 0]
    z[1:, 1] = alpha_z_unwrapped * z[1:, 0] + (1 - alpha_z_unwrapped) * z[:-1, 1]
    z[1:, 2] = alpha_z_unwrapped * z[1:, 1] + (1 - alpha_z_unwrapped) * z[:-1, 2]

    for i in range(1, size - 1):
        z[i, 0] = alpha_z_unwrapped * db[i] + (1 - alpha_z_unwrapped) * z[i - 1, 0]
        z[i, 1] = alpha_z_unwrapped * z[i, 0] + (1 - alpha_z_unwrapped) * z[i - 1, 1]
        z[i, 2] = alpha_z_unwrapped * z[i, 1] + (1 - alpha_z_unwrapped) * z[i - 1, 2]

    a[1:] = 3 * z[1:, 0] - 3 * z[1:, 1] + z[1:, 2]
    b[1:] = (alpha_z_unwrapped / (2 * (1 - alpha_z_unwrapped) ** 2)) * (
        (6 - 5 * alpha_z_unwrapped) * z[1:, 0]
        - 2 * (5 - 4 * alpha_z_unwrapped) * z[1:, 1]
        + (4 - 3 * alpha_z_unwrapped) * z[1:, 2]
    )
    c[1:] = (alpha_z_unwrapped / (1 - alpha_z_unwrapped)) ** 2 * (
        z[1:, 0] - 2 * z[1:, 1] + z[1:, 2]
    )

    forecast[2:] = a[1:-1] + b[1:-1] + 0.5 * c[1:-1]

    # square error between the measured  and the forecast values
    err = db - forecast
    square_err = err**2

    # Calculation of the forecast MAD
    for i in range(2, size - 1):

        MAD[i] = (
            abs(alpha_MAD * err[i]) + (1 - alpha_MAD) * MAD[i - 1]
        )  # Calculation of the MAD
        forecast_MAD[i + 1] = MAD[i]  # Calculation of the forecast MAD
        square_err[i] = (forecast_MAD[i] - abs(err[i])) ** 2
        # Calculation of the square error
        # between the forecast MAD and the absolute deviation between the forecast value and the
        # measured value

    RMSE_mad = (square_err / size).sum()  # Calculation of the RMSE
    return np.sqrt(RMSE_mad)


@njit
def objFun_alpha_MADini(
    MADini: float,
    alpha_MAD: float,
    alpha_z: float,
    db: np.ndarray,
    obj: Literal["err", "min_MAD"],
) -> float:
    size = db.shape[0]
    z = np.zeros([size, 3])  # Smoothed values
    a = np.zeros(size)  # Model parameter values
    b = np.zeros(size)  # Model parameter values
    c = np.zeros(size)  # Model parameter values
    forecast = np.zeros(size)  # Forecast values
    MAD = np.zeros(size)

    MAD[0] = MADini
    MAD[1] = MADini

    # Model initialisation
    z[0, 0] = db[0]
    z[0, 1] = db[0]
    z[0, 2] = db[0]

    z[1:, 0] = alpha_z * db[1:] + (1 - alpha_z) * z[:-1, 0]
    z[1:, 1] = alpha_z * z[1:, 0] + (1 - alpha_z) * z[:-1, 1]
    z[1:, 2] = alpha_z * z[1:, 1] + (1 - alpha_z) * z[:-1, 2]

    for i in range(1, size - 1):
        z[i, 0] = alpha_z * db[i] + (1 - alpha_z) * z[i - 1, 0]
        z[i, 1] = alpha_z * z[i, 0] + (1 - alpha_z) * z[i - 1, 1]
        z[i, 2] = alpha_z * z[i, 1] + (1 - alpha_z) * z[i - 1, 2]

    a[1:] = 3 * z[1:, 0] - 3 * z[1:, 1] + z[1:, 2]
    b[1:] = (alpha_z / (2 * (1 - alpha_z) ** 2)) * (
        (6 - 5 * alpha_z) * z[1:, 0]
        - 2 * (5 - 4 * alpha_z) * z[1:, 1]
        + (4 - 3 * alpha_z) * z[1:, 2]
    )
    c[1:] = (alpha_z / (1 - alpha_z)) ** 2 * (z[1:, 0] - 2 * z[1:, 1] + z[1:, 2])

    forecast[2:] = a[1:-1] + b[1:-1] + 0.5 * c[1:-1]

    # error between the measured  and the forecast values
    err = db - forecast
    # Calculation of the forecast MAD
    for i in range(2, size):
        MAD[i] = (
            abs(alpha_MAD * err[i]) + (1 - alpha_MAD) * MAD[i - 1]
        )  # Calculation of the MAD

    absolute_error = np.abs(MADini - np.mean(MAD[2:]))
    min_MAD = float(np.median(MAD))
    if obj == "err":
        return absolute_error
    else:
        return min_MAD


@njit
def alpha_MAD_determination(data: np.ndarray, lambda_z: float) -> tuple[float, float]:
    # This function calculates the right value of lambda_MAD in the model used
    # to calculate the forecast value of the standard desviation of the
    # forecast error

    # The RMSE is calculated with the forecast mean absolute devitaion (forecast MAD) and the absolute deviation between the forecast and the measured values to    choose the
    # optmimum value of the alpha_MAD.  This is done with the lambda_z choosen in the lambda_z_determination function

    db = data.flatten()
    size = db.shape[0]
    # Creation of the matrixes
    z = np.zeros((size, 3))  # Smoothed values
    a = np.zeros(size)  # Model parameter values
    b = np.zeros(size)  # Model parameter values
    c = np.zeros(size)  # Model parameter values
    forecast = np.zeros(size)  # Forecast value
    err = np.zeros(
        size
    )  # The error between the measured and the forecast values for lambda_z value from 0.01 to 1 with an interval of 0.01
    MAD = np.zeros((size, 100))  # MAD
    forecast_MAD = np.zeros((size, 100))  # The forecast MAD
    square_err = np.zeros(
        (size, 100)
    )  # The square error between the measured and the forecast values for lambda_z value from 0.01 to 1 with an interval of 0.01
    RMSE_mads = np.zeros(100)  # RMSE

    # Models initialisation

    # Forecast value model
    z[0, 0] = db[0]  # Smoothed value
    z[0, 1] = db[0]  # Smoothed value
    z[0, 2] = db[0]  # Smoothed value

    z[1, 0] = lambda_z * db[1] + (1 - lambda_z) * z[0, 0]  # Smoothed value
    z[1, 1] = lambda_z * z[1, 0] + (1 - lambda_z) * z[0, 1]  # Smoothed value
    z[1, 2] = lambda_z * z[1, 1] + (1 - lambda_z) * z[0, 2]  # Smoothed value

    a[1] = 3 * z[1, 0] - 3 * z[1, 1] + z[1, 2]  # Model parameter value
    b[1] = (lambda_z / (2 * (1 - lambda_z) ** 2)) * (
        (6 - 5 * lambda_z) * z[1, 0]
        - 2 * (5 - 4 * lambda_z) * z[1, 1]
        + (4 - 3 * lambda_z) * z[1, 2]
    )  # Model parameter value
    c[1] = (lambda_z / (1 - lambda_z)) ** 2 * (
        z[1, 0] - 2 * z[1, 1] + z[1, 2]
    )  # Model parameter value

    forecast[2] = a[1] + b[1] * 1 + (1 / 2) * c[1] ** 2  # Forecast value

    # Calculation of the forecast values
    i = 2
    while i < size - 1:
        z[i, 0] = (
            lambda_z * db[i] + (1 - lambda_z) * z[i - 1, 0]
        )  # Calculation of the smoothed value
        z[i, 1] = (
            lambda_z * z[i, 0] + (1 - lambda_z) * z[i - 1, 1]
        )  # Calculation of the smoothed value
        z[i, 2] = (
            lambda_z * z[i, 1] + (1 - lambda_z) * z[i - 1, 2]
        )  # Calculation of the smoothed value

        a[i] = (
            3 * z[i, 0] - 3 * z[i, 1] + z[i, 2]
        )  # Calculation of the model parameter value
        b[i] = (lambda_z / (2 * (1 - lambda_z) ** 2)) * (
            (6 - 5 * lambda_z) * z[i, 0]
            - 2 * (5 - 4 * lambda_z) * z[i, 1]
            + (4 - 3 * lambda_z) * z[i, 2]
        )  # Calculation of the model parameter value
        c[i] = (lambda_z / (1 - lambda_z)) ** 2 * (
            z[i, 0] - 2 * z[i, 1] + z[i, 2]
        )  # Calculation of the model parameter value

        forecast[i + 1] = (
            a[i] + b[i] * 1 + (1 / 2) * c[i] ** 2
        )  # Calculation of the forecast value

        err[i] = (
            db[i] - forecast[i]
        )  # Calculation of the error between the measured and the forecast values
        i = i + 1

    n = 1
    while n < 101:
        alpha_MAD = (
            n / 100
        )  # The RMSE is calculated for alpha_MAD value from 0.01 to 1 with an interval of 0.01

        # Calculation of the forecast MAD
        i = 2
        while i < size - 1:
            MAD[i, n] = (
                abs(alpha_MAD * err[i]) + (1 - alpha_MAD) * MAD[i - 1, n]
            )  # Calculation of the MAD
            forecast_MAD[i + 1, n] = MAD[i, n]  # Calculation of the forecast MAD
            square_err[i, n] = (
                forecast_MAD[i, n] - abs(err[i])
            ) ** 2  # Calculation of the square error between the forecast MAD and the absolute deviation between the  forecast value and the measured value

        RMSE_mads[n] = np.sum(square_err[:, n]) / size  # Calculation of the RMSE

    min_rmse_idx = np.array(np.argmin(RMSE_mads))[0]
    lambda_MAD = 0.01 * min_rmse_idx
    RMSE_mad = RMSE_mads[min_rmse_idx]

    return lambda_MAD, RMSE_mad


def lambda_determination(
    data: np.ndarray, mad_ini_0: float
) -> tuple[float, float, float, float]:
    # To do : add comments on the optimisation procedure!

    # This function calculates the right value of lamda_z in the model used to
    # calculate the forecast value of the variable

    # The RMSE is calculated on the forecast and the measured value to choose the
    # optmimum value of the alpha_z

    # Raw data selection
    size = data.shape[0]
    db = np.array(data).flatten()

    # The fminsearch provides an unconstrained optimization of lambda_z which
    # is faster and more precise than the previous systematic method. However,
    # in some particular cases, it might be necessary to return to the original
    # method (i.e. optimal lambda_z not between 0 and 1).

    lambda_0 = np.array(1).reshape(1, 1)
    optimization_failed = False
    try:
        result = minimize(objFun_alpha_z, lambda_0, db, method="Nelder-Mead")
        log_lambda_z = result.x[0]

        lambda_z = np.exp(-(log_lambda_z**2))
        result = minimize(
            objFun_alpha_MAD, lambda_0, args=(lambda_z, db), method="Nelder-Mead"
        )
        log_lambda_MAD = result.x[0]

        lambda_MAD = np.exp(-(log_lambda_MAD**2))

        mad_ini_0_vec = np.array(mad_ini_0).reshape(1, 1)
        result = minimize(
            objFun_alpha_MADini, mad_ini_0_vec, args=(lambda_MAD, lambda_z, db, "err")
        )
        MAD_ini = result.x[0]

        min_MAD = objFun_alpha_MADini(MAD_ini, lambda_MAD, lambda_z, db, "min_MAD")
    except Exception as e:
        optimization_failed = True
        print("Optimization failed", e)

    # If the fminsearch optimization failed, the following procedure ensures
    # that lambda_z and lambda_MAD are properly computed. The following
    # procedure is slower on the average case.
    if optimization_failed:
        # Creation of the matrixes
        z = np.zeros((size, 3, 100))
        a = np.zeros(size)
        b = np.zeros(size)
        c = np.zeros(size)
        forecast = np.zeros(size)
        square_err = np.zeros((size, 100))
        rmse_z = np.zeros((100, 1))

        alpha_z_vect = np.linspace(0.01, 0.01, 1)
        n = 0
        while n < 100:
            # The RMSE is calculated for alpha_z values from 0.01 to 1 with an interval of 0.01
            alpha_z = alpha_z_vect[n]

            # Model initialisation
            # Smoothed value
            z[0, 0] = db[0]
            z[0, 1] = db[0]
            z[0, 2] = db[0]

            z[1, 0] = alpha_z * db[1] + (1 - alpha_z) * z[0, 0]
            z[1, 1] = alpha_z * z[1, 0] + (1 - alpha_z) * z[0, 1]
            z[1, 2] = alpha_z * z[1, 1] + (1 - alpha_z) * z[0, 2]

            # Model parameter value
            a[1] = 3 * z[1, 0] - 3 * z[1, 1] + z[1, 2]
            b[1] = (alpha_z / (2 * (1 - alpha_z) ^ 2)) * (
                (6 - 5 * alpha_z) * z[1, 0]
                - 2 * (5 - 4 * alpha_z) * z[1, 1]
                + (4 - 3 * alpha_z) * z[1, 2]
            )
            c[1] = (alpha_z / (1 - alpha_z)) ^ 2 * (z[1, 0] - 2 * z[1, 1] + z[1, 2])

            # Forecast value
            forecast[2] = a[1] + b[1] * 1 + (1 / 2) * c[1] ^ 2

            # Data on which the filter is applied

            # Calculation of the smoothed value
            z[2 : size - 1, 0, n] = (
                alpha_z * db[2 : size - 1] + (1 - alpha_z) * z[2 : size - 1 - 1, 0, n]
            )
            z[2 : size - 1, 1, n] = (
                alpha_z * z[2 : size - 1, 0, n]
                + (1 - alpha_z) * z[2 : size - 1 - 1, 1, n]
            )
            z[2 : size - 1, 2, n] = (
                alpha_z * z[2 : size - 1, 1, n]
                + (1 - alpha_z) * z[2 : size - 1 - 1, 2, n]
            )

            # Calculation of the model parameter value
            a[2 : size - 1] = (
                3 * z[2 : size - 1, 1, n]
                - 3 * z[2 : size - 1, 2, n]
                + z[2 : size - 1, 3, n]
            )
            b[2 : size - 1] = (alpha_z / (2 * (1 - alpha_z) ** 2)) * (
                (6 - 5 * alpha_z) * z[2 : size - 1, 1, n]
                - 2 * (5 - 4 * alpha_z) * z[2 : size - 1, 2, n]
                + (4 - 3 * alpha_z) * z[2 : size - 1, 3, n]
            )
            c[2 : size - 1] = (alpha_z / (1 - alpha_z)) ** 2 * (
                z[2 : size - 1, 1, n]
                - 2 * z[2 : size - 1, 2, n]
                + z[2 : size - 1, 3, n]
            )

            # Calculation of the forecast value
            forecast[3:size, n] = (
                a[2 : size - 1] + b[2 : size - 1] * 1 + (1 / 2) * c[2 : size - 1] ** 2
            )

            # square error between the measured  and the forecast values
            square_err[2 : size - 1, n] = np.square(
                db[2 : size - 1] - forecast[2 : size - 1, n]
            )

            # Calculation of the RMSE
            rmse_z[n] = np.sum(square_err[:, n]) / size
            n = n + 1

        min_idx = np.argmin(rmse_z)
        lambda_z = alpha_z_vect[min_idx]
        lambda_MAD, _ = alpha_MAD_determination(data, lambda_z)
        MAD_ini = mad_ini_0

    return lambda_z, lambda_MAD, MAD_ini, min_MAD


@njit
def calc_z(datapoint: float, alpha: float, z_prev: np.ndarray) -> np.ndarray:
    z = np.zeros(3)
    z[0] = (alpha * datapoint) + ((1 - alpha) * z_prev[0])
    z[1] = alpha * z[0] + (1 - alpha) * z_prev[1]
    z[2] = alpha * z[1] + (1 - alpha) * z_prev[2]
    return z


@njit
def calc_forecast(alpha: float, z: np.ndarray) -> float:
    a = 3 * z[0] - 3 * z[1] + z[2]
    b = (alpha / (2 * (1 - alpha) ** 2)) * (
        (6 - 5 * alpha) * z[0] - 2 * (5 - 4 * alpha) * z[1] + (4 - 3 * alpha) * z[2]
    )
    c = (alpha / (1 - alpha)) ** 2 * (z[0] - 2 * z[1] + z[2])
    return a + b + 0.5 * c


def vectorized_ewma(data, param):
    # warning 'off'
    # ## CG Modification: Definition of the Outlier vector has changed!!! An
    #                     outlier flag is 1 and an accepted value flag is 0.
    #                     This is consistent with all other outlier detection
    #                     methods implemented.
    # INPUT:
    # DATA : Original data to filter.
    #           Column 1 = date of the observation in Matlab format.
    #           Column 2 = raw observation.
    # PARAM : Structure of parameters. Should be initialized by the function
    #         DefaultParam.m and the calibration from the function ModelCalib.m
    #         should be done to ensure that all parameters are properly
    #         initialized.
    # OUTPUT:
    # ACCEPTED_DATA : Data without outliers, but unfiltered.
    # SEC_RESULTS : A structure containing secondary results. Specifically:
    #   Orig :      Original dataset, for reference
    #   forecast_outlier : Forecast of the data based on the outlier filter.
    #   UpperLimit_outlier : limit above which an observation becomes an
    #                        outlier.
    #   LowerLimit_outlier : Limit under which an observation becomes an
    #                        outlier.
    #   outlier     : Detected outliers. outlier(i) = 1 for detected outlier.
    #                                    outlier(i) = 0 for accepted data.
    #   out_of_control_outlier : Data was in or out of control.
    #                               = 1 for "Out of control"
    #                               = 0 for "In control"
    #   reini_outlier : Indicates points of reinitialization of the outlier
    #                   filter when out of control.
    # A third-order smoothing model is used to predict the forecast value at time
    # t+1.

    # A first-order smoothing model is used to predict the forecast mean absolute deviation at
    # time t+1.

    # name = newData.columns[0]
    n_dat = len(data) + 3
    RawData = np.append(np.array(data).flatten(), np.array([np.nan, np.nan, np.nan]))
    # i_0 = 0
    # i_F = n-1

    # DATA : Time serie with the time and its corresponding value
    # ALPHA_Z and ALPHA_MAD : Smoothing parameters
    K = param["outlier_detection"]["nb_s"]
    # Number of standard deviation units
    # used for the calculation of the prediction interval
    MAD_ini = param["outlier_detection"]["MAD_ini"]
    # Initial mean absolute deviation used
    # to start or reinitiate the outlier detection method
    nb_reject = param["outlier_detection"]["nb_reject"]
    # Number of consecutive rejected data needed to reinitiate
    # the outlier detection method.  When nb_reject data are rejected,
    # this is called an out-of-control.
    nb_backward = param["outlier_detection"]["nb_backward"]
    # Number of data before the last rejected data
    # (the last of nb_reject data) where the outlier detection method
    # is reinitialization for a forward application.
    alpha_z = param["outlier_detection"]["lambda_z"]
    alpha_MAD = param["outlier_detection"]["lambda_MAD"]
    min_MAD = param["outlier_detection"]["min_MAD"]

    # Creation of the matrices

    # Accepted data and the forecast values which replaced the outliers. The
    # vector is initialized to NaN to prevent zeros from appearing at the
    # beginning and the  of the vector.
    AcceptedData = np.full((n_dat,), np.nan)

    # smoothed values
    z = np.zeros([3, 1])
    z_previous = np.zeros([3, 1])

    # mean absolute deviations
    MAD = np.full((n_dat,), np.nan)

    # forecast error standard deviation values
    s = np.full((n_dat,), np.nan)

    # OUTLIER : quality indicator => Data accepted = 0, Data rejected = 1
    outlier = np.zeros((n_dat,))

    # Limits of the prediction interval
    LowerLimit = np.full((n_dat,), np.nan)
    UpperLimit = np.full((n_dat,), np.nan)

    # Matrices used to recover data lost after a backward application of the
    # outlier detection method. Used in the Backward_Method.m script.
    # #ok<*NASGU>
    back_AcceptedData = np.full(
        (n_dat,), np.nan
    )  # Accepted data and the forecast values which replaced the outliers
    back_LowerLimit = np.full(
        (n_dat,), np.nan
    )  # Lower limit values of the prediction interval
    back_UpperLimit = np.full(
        (n_dat,), np.nan
    )  # Upper limit values of the prediction interval
    back_outlier = np.full(
        (n_dat,), 0.0
    )  # Quality indicator => Data rejected=0, Data accepted=1
    forecast = np.full((n_dat,), np.nan)  # Forecast values
    out_of_control = np.full(
        (n_dat,), 0.0
    )  # If out_of_control(i) = 1, there is an out of control situation.
    reini = np.full((n_dat,), 1.0)  # It indicates where the reinitialisation occurs

    # smoothed value
    z_previous[0] = RawData[0]
    z_previous[1] = RawData[0]
    z_previous[2] = RawData[0]

    z = calc_z(RawData[1], alpha_z, z_previous)

    # forecast value
    forecast = calc_forecast(alpha_z, z)

    MAD[2] = MAD_ini  # Initial MAD value
    s[2] = 1.25 * MAD[2]  # Initial forecast error standard deviation value
    LowerLimit[2] = forecast - K * s[2]  # Lower limit value
    UpperLimit[2] = forecast + K * s[2]  # Upper limit value

    # Application of the outlier detection method
    for i in range(2, n_dat - 3):  # Without reinitialization
        if (RawData[i] < UpperLimit[i]) and (RawData[i] > LowerLimit[i]):
            # The data is inside the prediction interval
            AcceptedData[i] = RawData[i]  # Accepted data

            outlier[i] = 0  # Quality indicator- Accepted=1

            # Calculation of the smoothed value
            z_previous = z
            z = calc_z(AcceptedData[i], alpha_z, z_previous)

            # Calculation of the MAD
            MAD[i + 1] = (
                abs(alpha_MAD * (AcceptedData[i] - forecast)) + (1 - alpha_MAD) * MAD[i]
            )
            MAD[i + 1] = max([MAD[i + 1], min_MAD])

            # Calculation of the forecast error standard deviation
            s[i + 1] = 1.25 * MAD[i + 1]

            # Calculation of the model parameter value
            forecast = calc_forecast(alpha_z, z)

            # Calculation of the prediction interval for the next data
            LowerLimit[i + 1] = forecast - K * s[i + 1]
            UpperLimit[i + 1] = forecast + K * s[i + 1]

        else:
            # The data is out of the prediction interval = Outlier

            # Outlier are replaced by the forecast value
            AcceptedData[i] = forecast
            outlier[i] = 1

            # Last MAD value kept
            MAD[i + 1] = MAD[i]
            MAD[i + 1] = max([MAD[i + 1], min_MAD])

            # Last forecast error standard deviation value
            s[i + 1] = s[i]

            # Last value kept for the next limit
            LowerLimit[i + 1] = forecast - K * s[i + 1]
            UpperLimit[i + 1] = forecast + K * s[i + 1]

    # Detection of outliers with reinitialization if an "out-of-control"
    # situation is encountered.
    for i in range(nb_reject, n_dat - 3):
        if (RawData[i] < UpperLimit[i]) and (RawData[i] > LowerLimit[i]):
            # The data is inside the prediction interval
            AcceptedData[i] = RawData[i]  # Accepted data

            outlier[i] = 0  # Quality indicator: Accepted = 1

            # Calculation of the smoothed value
            z_previous = z
            z = calc_z(AcceptedData[i], alpha_z, z_previous)

            # Calculation of the mean absolute deviation
            MAD[i + 1] = (
                abs(alpha_MAD * (AcceptedData[i] - forecast)) + (1 - alpha_MAD) * MAD[i]
            )
            MAD[i + 1] = max([MAD[i + 1], min_MAD])

            # Calculation of the forecast error standard deviation value
            s[i + 1] = 1.25 * MAD[i + 1]

            # Calculation of the forecast value
            forecast = calc_forecast(alpha_z, z)

            # Calculation of the prediction interval for the next data
            LowerLimit[i + 1] = forecast - K * s[i + 1]
            UpperLimit[i + 1] = forecast + K * s[i + 1]
        else:
            # The data is out of the prediction interval = Outlier

            AcceptedData[i] = forecast  # Outlier are replaced by the forecast value
            # outlier(i) = 0  # Quality indicator- outlier=0
            outlier[i] = 1  # Quality indicator- outlier=0

            MAD[i + 1] = MAD[i]  # Last MAD kept
            MAD[i + 1] = max([MAD[i + 1], min_MAD])

            s[i + 1] = s[i]  # Last forecast error standard deviation value kept

            # Last value kept for the next limit
            LowerLimit[i + 1] = forecast - K * s[i + 1]
            UpperLimit[i + 1] = forecast + K * s[i + 1]

        # Reinitialization in case of an out of control, which caused by nb_reject data detected as outlier

        # The outlier detection method is reinitiated nb_reject data later and
        # is applied backward to recovered the lost data. After that,
        # it is applied forward again nb_backward data before the last data
        # rejected(the last of nb_reject data).

        # Backward application of the outlier detection method
        if out_of_control[i - nb_reject : i].sum() == 0:
            # if outlier((i-nb_reject):i) == 0
            if outlier[i - nb_reject : i].sum() == nb_reject:
                # ###### BACKWARD METHOD ########
                # If nb_reject data are detected as outlier, there is
                # a reinitialization of the outlier detection method

                # In_control = 0, out_of_control = 1. It marks where the out of
                # control starts and where the reinitialization is done
                out_of_control[i - nb_reject : i] = 1
                reini[i] = 1

                # Backward reinitialization
                # smoothed value
                z_previous = np.array([RawData[i + 3], RawData[i + 3], RawData[i + 3]])

                z = calc_z(RawData[i + 2], alpha_z, z_previous)

                z_previous = z
                z = calc_z(RawData[i + 1], alpha_z, z_previous)

                forecast = calc_forecast(alpha_z, z)

                # Initial MAD value
                MAD[i] = MAD_ini
                # Initial forecast error standard deviation value
                s[i] = 1.25 * MAD[i]

                back_LowerLimit[i] = forecast - K * s[i]  # Lower limit value
                back_UpperLimit[i] = forecast + K * s[i]  # Upper limit value

                # Backward application of the outlier detection method
                for q in range(nb_reject):
                    f = (i - q) + 1  # Used for backward

                    if (RawData[f] > back_UpperLimit[f]) or (
                        RawData[f] < back_LowerLimit[f]
                    ):
                        # The data is out of the prediction interval and marked as "Outlier"

                        back_AcceptedData[
                            f
                        ] = forecast  # Outlier is replaced by the forecast value
                        # back_outlier(f) = 0    # Quality indicator - outlier=0
                        back_outlier[f] = 1  # Quality indicator - outlier=0

                        MAD[f - 1] = MAD[f]  # Last MAD kept
                        MAD[f - 1] = max([MAD[f - 1], min_MAD])

                        s[f - 1] = s[f]  # Forecast error standard deviation value kept

                        # Last value kept for the next limit
                        back_LowerLimit[f - 1] = forecast - K * s[f - 1]
                        back_UpperLimit[f - 1] = forecast + K * s[f - 1]
                    else:
                        # If the data is inside the prediction interval
                        back_AcceptedData[f] = RawData[f]  # Accepted data
                        # back_outlier[f] = 1    # Quality indicator -Accepted=1
                        back_outlier[f] = 0  # Quality indicator -Accepted=1

                        # Calculation of the smoothed value
                        z_previous = z
                        z = calc_z(RawData[f], alpha_z, z_previous)

                        # Calculation of the MAD
                        MAD[f - 1] = (
                            abs(alpha_MAD * (back_AcceptedData[f] - forecast))
                            + (1 - alpha_MAD) * MAD[f]
                        )
                        MAD[f - 1] = max([MAD[f - 1], min_MAD])

                        # Calculation of the  forecast error standard deviation value
                        s[f - 1] = 1.25 * MAD[f - 1]

                        # Calculation of the forecast value

                        forecast = calc_forecast(alpha_z, z)

                        # Calculation of the prediction interval for the next data
                        back_LowerLimit[f - 1] = forecast - K * s[f - 1]
                        back_UpperLimit[f - 1] = forecast + K * s[f - 1]

                # Forward application of the outlier detection method
                # The forward reinitialization is done nb_backward data before the last
                # rejected data which has caused an out of control situation

                # Forward reinitialization
                # Smoothed value
                prev_raw = RawData[i - nb_backward - 4]
                z_previous = np.array([prev_raw, prev_raw, prev_raw])

                prev_raw2 = RawData[i - nb_backward - 3]
                z = calc_z(prev_raw2, alpha_z, z_previous)

                prev_raw3 = RawData[i - nb_backward - 2]
                z_previous = z
                z = calc_z(prev_raw3, alpha_z, z_previous)

                prev_raw4 = RawData[i - nb_backward - 1]
                z_previous = z
                z = calc_z(prev_raw4, alpha_z, z_previous)

                # Model parameter value
                forecast = calc_forecast(alpha_z, z)

                MAD[i - nb_backward] = MAD_ini  # Initial MAD value
                s[i - nb_backward] = (
                    1.25 * MAD[i - nb_backward]
                )  # Initial forecast error standard deviation value

                LowerLimit[i - nb_backward] = (
                    forecast - K * s[i - nb_backward]
                )  # Lower limit value
                UpperLimit[i - nb_backward] = (
                    forecast + K * s[i - nb_backward]
                )  # Upper limit value

                # Forward application of the outlier detection method
                for k in range(i - nb_backward - 1, i):
                    if (RawData[k] < UpperLimit[k]) or (RawData[k] > LowerLimit[k]):
                        # The data is out of the prediction interval = Outlier
                        AcceptedData[
                            k
                        ] = forecast  # Outlier is replaced by the forecast
                        # outlier(k) = 0     # Quality indicator- outlier=0
                        outlier[k] = 1  # Quality indicator- outlier=0

                        MAD[k + 1] = MAD[k]  # Last MAD kept
                        MAD[k + 1] = max([MAD[k + 1], min_MAD])
                        s[k + 1] = s[
                            k
                        ]  # Last forecast error standard deviation value kept
                        # Last value kept for the next limit
                        LowerLimit[k + 1] = forecast - K * s[k + 1]
                        UpperLimit[k + 1] = forecast + K * s[k + 1]

                    else:
                        # The data is inside the prediction interval
                        AcceptedData[k] = RawData[k]  # Accepted data

                        # outlier(k) = 1 # Quality indicator -Accepted=1
                        outlier[k] = 0  # Quality indicator -Accepted=1

                        # Calculation of the smoothed value
                        z_previous = z
                        z = calc_z(AcceptedData[k], alpha_z, z_previous)

                        MAD[k + 1] = (
                            abs(alpha_MAD * (AcceptedData[k] - forecast))
                            + (1 - alpha_MAD) * MAD[k]
                        )
                        # Calculation of the MAD
                        MAD[k + 1] = max([MAD[k + 1], min_MAD])

                        s[k + 1] = 1.25 * MAD[k + 1]
                        # Calculation of the forecast error standard deviation value

                        # Calculation of the model parameter value
                        forecast = calc_forecast(alpha_z, z)

                        # Calculation of the prediction interval for the next data
                        LowerLimit[k + 1] = forecast - K * s[k + 1]
                        UpperLimit[k + 1] = forecast + K * s[k + 1]

                # After the backward application of the outlier detection method, only the first
                # half of the data between where the forward and

                # Backward applications are reinitialize is kept.
                # After the forward application of the outlier detection method,
                # only the second half of the data between where the forward and
                # backward applications are reinitialize is kept. In this way, the prediction
                # interval is adapted to the data serie before the application pf the outlier
                # detection method.
                strt = i - nb_reject + 1
                mid = int(i - np.floor(nb_backward / 2))
                AcceptedData[strt:mid] = back_AcceptedData[strt:mid]
                UpperLimit[strt:mid] = back_UpperLimit[strt:mid]
                LowerLimit[strt:mid] = back_LowerLimit[strt:mid]
                outlier[strt:mid] = back_outlier[strt:mid]

    # Filter last
    # Record the position of the next data to filter at next iteration.
    # index0 = len(RawData) - 3
    # Generation of outputs
    Sec_data = {
        "forecast": forecast,
        "UpperLimit_outlier": UpperLimit[:-3],
        "LowerLimit_outlier": LowerLimit[:-3],
        "outlier": [x == 1 for x in outlier][:-3],
        "out_of_control_outlier": out_of_control[:-3],
        "reini_outlier": reini[:-3],
    }
    Sec_Results = pd.DataFrame(index=data.index, data=Sec_data)

    data["Accepted"] = AcceptedData[:-3]

    newData = data.join(Sec_Results, how="left")

    return newData
