import numpy as np
import xarray as xr
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score



def cand_eqs(revrho, swe, ta, q, revstd, eqs_num):
    if eqs_num == 1:
        return (-q + revstd + 5.9512205)*(q + 16.553125*swe + 6.9710293)/(0.016569955**revrho + revrho + swe + ta**2/(0.8273294 - ta))
    elif eqs_num == 2:
        return -(0.3462239*q + np.tanh(swe))*(13.7060802364902*q - swe + 1.3196727) + (swe + 0.3462239)**0.4002787*(143.8155*revrho + revstd + 59.989)
    elif eqs_num == 3:
        return -3.929743*0.6691585**swe*q*(q + 3.59086928122647*swe + 0.28061634) - (swe + 0.34623227)**0.4040043*(-144.55663*revrho - revstd - 60.487194)
    elif eqs_num == 4:
        return (150.0753 - revrho)*(revrho + 0.3991598)*(swe + 0.34623227)**0.40179837 - (0.37222725*q + np.tanh(swe))*(12.3218962313973*q - revrho - swe)


def loss(x, y):
    y[y>100] = 100
    y[y<0] = 0
    y = np.delete(y, np.where(np.isnan(x)), axis=0)
    x = np.delete(x, np.where(np.isnan(x)), axis=0)
    x = np.delete(x, np.where(np.isnan(y)), axis=0)
    y = np.delete(y, np.where(np.isnan(y)), axis=0)
    return mean_squared_error(x, y), r2_score(x, y)



def test_mse():
    X_train, y_train = np.load('/tera05/lilu/fSCA/SRfSCA/data/design_feat_sd/z0/x_train.npy'), np.load('/tera05/lilu/fSCA/SRfSCA/data/design_feat_sd/z0/y_train.npy')
    x, y = np.load('/tera05/lilu/fSCA/SRfSCA/data/design_feat_sd/z0/x_test.npy'), np.load('/tera05/lilu/fSCA/SRfSCA/data/design_feat_sd/z0/y_test.npy')
    x_mean, x_std = np.nanmean(X_train, axis=0, keepdims=True), np.nanstd(X_train, axis=0, keepdims=True)
    x = (x - x_mean) / x_std

    for i in range(1, 5):
        y_pred = cand_eqs(x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], eqs_num=i)
        mse, r2 = loss(y, y_pred)
        print(mse, r2)



def main(X_train):
    """train and inference of 12 features neural network"""
    # define train and test period
    test_period = slice('2013-10-01', '2017-09-30') # ~20%

    # read snow depth
    ds = xr.open_dataset('/tera05/lilu/fSCA/SRfSCA/data/SD.nc')
    sd_test = ds.SD_Post.sel(time=test_period).values
    nt_test = sd_test.shape[0]
    lat, lon = ds.lat, ds.lon

    # read z0
    ds = xr.open_dataset('/tera05/lilu/fSCA/SRfSCA/data/z0.nc')
    z0 = ds.forest_height.values*0.1

    # read snow water equivalent and snow cover
    ds = xr.open_dataset('/tera05/lilu/fSCA/SRfSCA/data/SWE_SCA.nc')
    swe_test = ds.SWE_Post.sel(time=test_period).values
    sca_test = ds.SCA_Post.sel(time=test_period).values
    swe, sca = ds.SWE_Post.values, ds.SCA_Post.values
    swe_max = np.nanmax(swe)

    # read topo-related vars
    ds = xr.open_dataset('/tera05/lilu/fSCA/SRfSCA/data/MERITHydro/MERITHydro_0p1_TP.nc')
    elv_std = ds.elv_std.values[np.newaxis]   

    # read forcing
    ds = xr.open_dataset('/tera05/lilu/fSCA/SRfSCA/data/FORCING.nc')
    ta_test = ds.Ta_Post.sel(time=test_period).values
    q_test = ds.q_Post.sel(time=test_period).values

    # calculate own designed factors
    rho_test = swe_test/sd_test
    rho_test[np.where((rho_test>5000) | (rho_test<0))] = np.nan

    # construct test data
    x_test = np.stack([sd_test/z0, 1/rho_test, swe_test/swe_max, ta_test/273.15, q_test, \
        200/np.tile(elv_std, (nt_test,1,1))], axis=-1)
    nt, nx, ny, nf = x_test.shape

    # reshape and normalize test data  
    x_mean, x_std = np.nanmean(X_train, axis=0, keepdims=True), \
        np.nanstd(X_train, axis=0, keepdims=True)
    x_test = x_test.reshape(-1, x_test.shape[-1])
    x_test = (x_test-x_mean)/x_std
    idx = np.unique((np.where(np.isnan(x_test)) or (np.where(np.isinf(x_test))))[0])
    all_idx = np.arange(x_test.shape[0])
    rest_idx = np.delete(all_idx, idx, axis=0)
    x = np.delete(x_test, idx, axis=0)
    
    # predict
    for i in range(1,5):
        y_pred = cand_eqs(x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], eqs_num=i)
        y = np.full((x_test.shape[0],), np.nan)
        y[rest_idx] = y_pred    
        y[y>100] = 100
        y[y<0] = 0
        y = y/100
        y = y.reshape(nt, nx, ny)
        
        # save
        ds = xr.Dataset(
            {"fSCA": (("time", "x", "y"), y)},
                coords={
                    "x": lat.values,
                    "y": lon.values,
                    "time": pd.date_range(start='2013-10-01', periods=nt_test, freq='D'),
                },
        )
        ds.to_netcdf("fSCA_cand_eqs_{num}.nc".format(num=i))


if __name__ == '__main__':
    X_train = np.load('/tera05/lilu/fSCA/SRfSCA/data/design_feat_sd/z0/x_train.npy')
    main(X_train)
    
    #test_mse()