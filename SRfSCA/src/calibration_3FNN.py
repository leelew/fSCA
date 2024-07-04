import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd


def model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((3)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # (0,1)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model
    
    
def main():
    """train and inference of 12 features neural network"""
    # define train and test period
    train_period = slice('1999-10-01', '2013-09-30') # ~80%
    test_period = slice('2013-10-01', '2017-09-30') # ~20%

    # read snow depth
    ds = xr.open_dataset('SD.nc')
    sd_train = ds.SD_Post.sel(time=train_period).values
    sd_test = ds.SD_Post.sel(time=test_period).values
    nt_train, nt_test = sd_train.shape[0], sd_test.shape[0]
    lat, lon = ds.lat, ds.lon
    
    # read snow water equivalent and snow cover
    ds = xr.open_dataset('SWE_SCA.nc')
    swe_train = ds.SWE_Post.sel(time=train_period).values
    swe_test = ds.SWE_Post.sel(time=test_period).values
    sca_train = ds.SCA_Post.sel(time=train_period).values
    
    # read topo-related vars
    ds = xr.open_dataset('MERITHydro/MERITHydro_0p1_TP.nc')
    elv_std = ds.elv_std.values[np.newaxis]   
    
    # construct train data
    x_train = np.stack([sd_train, swe_train,  \
        np.tile(elv_std, (nt_train,1,1))], axis=-1).reshape(-1, 3)
    y_train = sca_train.reshape(-1,1)
    
    # remove NaN in train data
    y_train = np.delete(y_train, np.where(np.isnan(x_train))[0], axis=0)
    x_train = np.delete(x_train, np.where(np.isnan(x_train))[0], axis=0)
    x_train = np.delete(x_train, np.where(np.isnan(y_train))[0], axis=0)
    y_train = np.delete(y_train, np.where(np.isnan(y_train))[0], axis=0)

    # normalize train data
    x_mean, x_std = np.nanmean(x_train, axis=0, keepdims=True), np.nanstd(x_train, axis=0, keepdims=True)
    x_train = (x_train - x_mean) / x_std

    # train model
    mdl = model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    mdl.fit(x_train, y_train, epochs=50, batch_size=256, validation_split=0.2, callbacks=[callback])
    
    # save model
    mdl.save('3FNN.h5')

    # construct test data
    x_test = np.stack([sd_test, swe_test, \
        np.tile(elv_std, (nt_test,1,1))], axis=-1).reshape(-1, 3)
    nt, nx, ny, nf = x_test.shape  
    
    # reshape and normalize test data  
    x_test = x_test.reshape(-1, nf)
    x_test = (x_test-x_mean)/x_std
    idx = np.unique(np.where(np.isnan(x_test))[0])
    all_idx = np.arange(nt)
    rest_idx = np.delete(all_idx, idx, axis=0)
    x = np.delete(x_test, idx, axis=0)
    y = np.full((nt,1), np.nan)
    
    # predict
    y_pred = mdl.predict(x)
    y[rest_idx] = y_pred
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
    ds.to_netcdf("fSCA_3FNN.nc")



if __name__ == '__main__':
    main()
