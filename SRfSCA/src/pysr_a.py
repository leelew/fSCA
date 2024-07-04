import os
import argparse

import numpy as np
import xarray as xr
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error, r2_score


def loss(x, y):
    y[y>100]=100
    y[y<0]=0
    y = np.delete(y, np.where(np.isnan(x)), axis=0)
    x = np.delete(x, np.where(np.isnan(x)), axis=0)
    x = np.delete(x, np.where(np.isnan(y)), axis=0)
    y = np.delete(y, np.where(np.isnan(y)), axis=0)
    return mean_squared_error(x, y), r2_score(x, y)


def main(X_train, #(N, feat) 
         y_train, #(N, 1)
         X_test, #(N, feat)
         y_test, #(N, 1)
         num_feat, 
         loss_exp, 
         subset_size, 
         verylow_complexity, 
         unit_fit=False,
         n_hour=1, 
         normalize=True, 
         batching=False):
    print(num_feat, loss_exp, subset_size, verylow_complexity)

    # define train set
    if num_feat == 3: # (SD, SWE, std) - LA23
        X_train = X_train[:, [0, 1, 8]]
        X_test = X_test[:, [0, 1, 8]]
        feat_name = ['SD', 'SWE', 'STD']
        x_units = ['m', 'm', 'm']
    elif num_feat == 5: # (SD, SWE, Ta, q, std)
        X_train = X_train[:, [0, 1, 2, 7, 8]]
        X_test = X_test[:, [0, 1, 2, 7, 8]]
        feat_name = ['SD', 'SWE', 'T', 'q', 'STD']
        x_units = ['m', 'm', 'K', 'kg/kg', 'm']
    elif num_feat == 8: # (SD, SWE, Ta, ppt, q, std, slp, asp)
        X_train = X_train[:, [0, 1, 2, 6, 7, 8, 9, 10]]
        X_test = X_test[:, [0, 1, 2, 6, 7, 8, 9, 10]]
        feat_name = ['SD', 'SWE', 'T', 'P', 'q', 'STD', 'SINSLP', 'COSASP']
        x_units = ['m', 'm', 'K', 'mm', 'kg/kg', 'm', '', '']
    
    # normalize
    if normalize:
        x_mean, x_std = np.nanmean(X_train, axis=0, keepdims=True), np.nanstd(X_train, axis=0, keepdims=True)
        X_train = (X_train - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std
        print(x_mean, x_std)

    """
    # random choice samples
    idx = np.random.choice(X_train.shape[0], subset_size, replace=False)
    X, y = X_train[idx], y_train[idx]
    assert(np.isnan(X).any() == False and np.isnan(y).any() == False)
    assert(np.isnan(X_test).any() == False and np.isnan(y_test).any() == False)
    print("Data prepared!")

    # create tmp file
    tempdir = '/tera06/lilu/fSCA/tmp/'+'case_{num_feat}_{loss_exp}_{subset_size}_{verylow_complexity}/'.format(
        num_feat=num_feat, loss_exp=loss_exp, subset_size=subset_size, verylow_complexity=verylow_complexity)
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    np.save("x_test_1.npy", X_test)
    np.save("y_test_1.npy", y_test)

    # define binary & unary operators and complexsity of operators
    unary_operators = ["exp","tanh","sin","cos","abs","cube","neg","square","inv(x)=1/x"]
    binary_operators = ["div","mult","plus","sub","pow"]

    # Very low-complexity operators (x)
    very_low_complex_ops = ["mult", "plus", "sub", "neg"] 
    # Low-complexity operators (2x)
    low_complex_ops = ["div", "abs", "cube", "square"]
    # Medium-complexity operators (3x)
    medium_complex_ops = ["exp", "tanh", "inv", "sin", "cos"]
    # High-complexity operators (9x)
    high_complex_ops = ["pow"]

    # train
    model = PySRRegressor(
        #procs=10,
        populations=20,
        #ncycles_per_iteration=100000,
        niterations=1000000000,  # Run forever
        timeout_in_seconds=int(3600*n_hour), 
        maxsize=100,
        maxdepth=5,
        complexity_of_operators = {**{key: 8*verylow_complexity for key in high_complex_ops},
                                **{key: 3*verylow_complexity for key in medium_complex_ops}, 
                                **{key: 2*verylow_complexity for key in low_complex_ops},
                                **{key: 1*verylow_complexity for key in very_low_complex_ops}},
        complexity_of_variables=1,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        extra_sympy_mappings={'inv': lambda x: 1/x},
        model_selection="best",
        constraints={"pow": (9, 1)}, 
        nested_constraints={"tanh": {"sin":0, "cos":0, "tanh":0},
                            "sin": {"sin":0, "cos":0, "tanh":0},
                            "cos": {"sin":0, "cos":0, "tanh":0}},
        loss="loss(x, y) = (abs(max(min(x,100),0) - y))^%d"%loss_exp, #(julia syntax)
        tempdir=tempdir,
        temp_equation_file=True,
        delete_tempfiles=False,
        #cluster_manager="slurm",
        #parsimony=10,
        bumper=True,
        weight_optimize=0.001,)
        #adaptive_parsimony_scaling=100)
        #batching=batching)
    #if unit_fit:
    #    model.fit(X, y, variable_names=feat_name, X_units=x_units)
    #else:
    model.fit(X, y, variable_names=feat_name)
    
    # save 
    with open(os.path.join(model.tempdir_, 'out.txt') , 'a') as file:
        file.write('Number of features: {num}\n'.format(num=num_feat))
        file.write('Size of the subset: %d\n'%subset_size)
        file.write('Loss {n}th power\n'.format(n=loss_exp))
        file.write('Complexity of very low complexity ops: %d\n'%verylow_complexity)
        file.write('Unit fit: %s\n'%unit_fit)
        file.write('Normalize: %s\n'%normalize)
        file.write('Batching: %s\n'%batching)
        file.write(model.latex_table(precision=3))
        file.write('\n')

        # predict
        for i in range(len(model.equations_)):
            pred = model.predict(X_test, i)
            mse, r2 = loss(y_test, pred)
            condition1 = (mse < 460) & (r2 > 0.50)
            condition2 = (mse < 170) & (r2 > 0.75)
            condition3 = (mse < 91) & (r2 > 0.79)

            file.write("\nEquation: {eq}".format(eq=model.sympy(i)))
            file.write("\ntest MSE {mse}".format(i=i, mse=np.round(mse, 2)))
            file.write("\ntest R2 {r2}".format(i=i, r2=np.round(r2, 2)))
            file.write("\nbetter than NY07? {cond}".format(cond=condition1))
            file.write("\nbetter than M23? {cond}".format(cond=condition2))
            file.write("\nbetter than 3FNN? {cond}\n".format(cond=condition3))

    """

if __name__ == '__main__':

    X_train, y_train = np.load('x_train.npy'), np.load('y_train.npy')
    X_test, y_test = np.load('x_test.npy'), np.load('y_test.npy')

    config = argparse.ArgumentParser()
    config.add_argument('--num_feat', type=int, default=8)
    config.add_argument('--loss_exp', type=int, default=2)
    config.add_argument('--subset_size', type=int, default=5000)
    config.add_argument('--verylow_complexity', type=int, default=2)
    config.add_argument('--unit_fit', type=bool, default=False)
    config.add_argument('--n_hour', type=int, default=2)
    config.add_argument('--normalize', type=bool, default=False)
    config.add_argument('--batching', type=bool, default=False)
    args = config.parse_args()

    model = main(
         X_train, y_train, 
         X_test, y_test,
         args.num_feat, 
         args.loss_exp, 
         args.subset_size, 
         args.verylow_complexity)
    
