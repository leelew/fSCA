import xarray as xr
from scipy.optimize import minimize
import numpy as np
import pandas as pd



# aug
PATH = '/tera05/lilu/fSCA/SRfSCA/data/'

# construct train data
train_period = slice('1999-10-01', '2013-09-30')

ds = xr.open_dataset(PATH+'SD.nc')
sd = ds.SD_Post.load().sel(time=train_period).values
ds = xr.open_dataset(PATH+'SWE_SCA.nc')
swe = ds.SWE_Post.load().sel(time=train_period).values
sca = ds.SCA_Post.load().sel(time=train_period).values
ds = xr.open_dataset(PATH+'MERITHydro/MERITHydro_0p1_TP.nc')
elv_std = ds.elv_std.load().values


def NY07(sd, z0, rho_snow, rho_new, m, a):
    return np.tanh(sd/(a*z0*(rho_snow/rho_new)**m))

def cal_rmse(x, y):
    a = np.sqrt(np.nanmean((x - y)**2))
    return a

def f_1(params):
    m = params
    pred = NY07(sd, 0.01, swe/sd*1000, 100, m, 2.5)
    return cal_rmse(sca, pred)

def f_4(params):
    z0, rho_new, m, a = params
    pred = NY07(sd, z0, swe/sd*1000, rho_new, m, a)
    return cal_rmse(sca, pred)

# optimize NY07 by train data 
# tune 1 param, melting factor
initial_guess = [1.6]
result = minimize(f_1, initial_guess, method='nelder-mead', options={'disp': True})
if result.success:
    fitted_params = result.x
    print(fitted_params)

# tune 4 param 
initial_guess = [0.01, 100, 1.6, 2.5]
result = minimize(f_4, initial_guess, method='nelder-mead', options={'disp': True})
if result.success:
    fitted_params = result.x
    print(fitted_params)



# predict on test period 
test_period = slice('2013-10-01', '2017-09-30')
ds = xr.open_dataset(PATH+'SD.nc')
sd = ds.SD_Post.sel(time=test_period).values
ds = xr.open_dataset(PATH+'SWE_SCA.nc')
swe = ds.SWE_Post.sel(time=test_period).values
ds = xr.open_dataset(PATH+'MERITHydro/MERITHydro_0p1_TP.nc')
elv_std = ds.elv_std.values
lat, lon = ds.lat, ds.lon

sca_NY07 = NY07(sd, 0.01, swe/sd*1000, 100, 1.6, 2.5)
sca_NY07_tuned_1 = NY07(sd, 0.01, swe/sd*1000, 100, 2.70953125, 2.5)
sca_NY07_tuned_4 = NY07(sd, 0.00719659212, swe/sd*1000, 67.8581467, 2.18390713, 2.51452541)


# save 
ds = xr.Dataset(
    {"fSCA": (("time", "x", "y"), sca_NY07)},
        coords={
            "x": lat.values,
            "y": lon.values,
            "time": pd.date_range(start='2013-10-01', periods=sd.shape[0], freq='D'),
        },
)
ds.to_netcdf("fSCA_NY07.nc")

ds = xr.Dataset(
    {"fSCA": (("time", "x", "y"), sca_NY07_tuned_1)},
        coords={
            "x": lat.values,
            "y": lon.values,
            "time": pd.date_range(start='2013-10-01', periods=sd.shape[0], freq='D'),
        },
)
ds.to_netcdf("fSCA_NY07_tuned_1.nc")


ds = xr.Dataset(
    {"fSCA": (("time", "x", "y"), sca_NY07_tuned_4)},
        coords={
            "x": lat.values,
            "y": lon.values,
            "time": pd.date_range(start='2013-10-01', periods=sd.shape[0], freq='D'),
        },
)
ds.to_netcdf("fSCA_NY07_tuned_4.nc")