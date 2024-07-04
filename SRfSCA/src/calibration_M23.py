import xarray as xr
from scipy.optimize import minimize
import numpy as np
import pandas as pd



# construct train data
train_period = slice('1999-10-01', '2013-09-30')

ds = xr.open_dataset('SD.nc')
sd = ds.SD_Post.load().sel(time=train_period).values
ds = xr.open_dataset('SWE_SCA.nc')
swe = ds.SWE_Post.load().sel(time=train_period).values
sca = ds.SCA_Post.load().sel(time=train_period).values
ds = xr.open_dataset('MERITHydro/MERITHydro_0p1_TP.nc')
elv_std = ds.elv_std.load().values

def M23(sd, z0, rho_snow, rho_new, m, topo_std, a):
    scf = np.tanh(sd/(2.5*z0*(rho_snow/rho_new)**m))*((sd/(sd+0.0002*topo_std))**a)
    return scf

def cal_rmse(x, y):
    a = np.sqrt(np.nanmean((x - y)**2))
    print(a)
    return a

def f(params):
    m, a = params
    pred = M23(sd, 0.01, swe/sd*1000, 100, m, elv_std, a)
    rmse = cal_rmse(sca, pred)
    return rmse

# optimize
initial_guess = [1.6, 1]
result = minimize(f, initial_guess, method='nelder-mead', options={'disp': True})
if result.success:
    fitted_params = result.x
    print(fitted_params)
    
# predict on test period 
test_period = slice('2013-10-01', '2017-09-30')
ds = xr.open_dataset('SD.nc')
sd = ds.SD_Post.sel(time=test_period).values
ds = xr.open_dataset('SWE_SCA.nc')
swe = ds.SWE_Post.sel(time=test_period).values
ds = xr.open_dataset('MERITHydro/MERITHydro_0p1_TP.nc')
elv_std = ds.elv_std.values
lat, lon = ds.lat, ds.lon

sca_M23 = M23(sd, 0.01, swe/sd*1000, 100, 1.6, elv_std, 0.5)
sca_M23_tuned = M23(sd, 0.01, swe/sd*1000, 100, 2.53821066, elv_std, 0.34894026)

# save 
ds = xr.Dataset(
    {"fSCA": (("time", "x", "y"), sca_M23)},
        coords={
            "x": lat.values,
            "y": lon.values,
            "time": pd.date_range(start='2013-10-01', periods=sd.shape[0], freq='D'),
        },
)
ds.to_netcdf("fSCA_M23.nc")

ds = xr.Dataset(
    {"fSCA": (("time", "x", "y"), sca_M23_tuned)},
        coords={
            "x": lat.values,
            "y": lon.values,
            "time": pd.date_range(start='2013-10-01', periods=sd.shape[0], freq='D'),
        },
)
ds.to_netcdf("fSCA_M23_tuned.nc")

