    def param_hyperopt(self, x_train, x_valid, y_train, y_valid, w_train=None, w_valid=None, n_iter = 500, space = None):
        """Finds optimal hyperparameters based on cross validation AUC
        
        Arguments:
            x_train {pandas.DataFrame} -- training set
            x_valid {pandas.DataFrame} -- valid set
            y_train {pandas.DataFrame} -- target of training set
            y_valid {pandas.DataFrame} -- target of valid set
        
        Keyword Arguments:
            w_train {pandas.DataFrame} -- training set weights (default: None)
            w_valid {pandas.DataFrame} -- valid set weights (default: None)
            n_iter {int} -- number of iteration (default: {500})
            space {dict} -- hyperparameter space to be searched. if None, a default space is used (deafult: None)
        
        Returns:
            dictionary -- optimal hyperparameters
        """

        merged_ds = pd.concat([x_train[self.cols_pred], x_valid[self.cols_pred]]).reset_index(drop=True)
        merged_target = pd.concat([y_train, y_valid]).reset_index(drop=True)
        if (w_train is not None) and (w_valid is not None):
            merged_weight = pd.concat([w_train, w_valid]).reset_index(drop=True)
        else:
            merged_weight = None

        train_data = lgb.Dataset(merged_ds, label=merged_target, weight=merged_weight)

        def objective (params):

            print(params)
            cv_results = lgb.cv(params, train_data, stratified = True, nfold = 3)

            best_score = -max(cv_results['auc-mean'])
            print('Actual gini:')
            print(2*abs(best_score)-1)
            print('----------------------------------------------------------------------------------')

            return {'loss': best_score, 'params': params, 'status': STATUS_OK}

        if space is None:
            space = {
                'learning_rate': hp.choice('learning_rate', np.arange(0.02, 0.1, 0.02, dtype=float)),
                'num_leaves': hp.choice('num_leaves', np.arange(2, 64, 2, dtype=int)),
                'max_depth': hp.choice('max_depth', np.arange(2, 5, 1, dtype=int)),
                'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.5, 0.9, 0.05, dtype=float)),
                'subsample': hp.choice('subsample', np.arange(0.5, 0.9, 0.05, dtype=float)),
                'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(50, 500, 50, dtype=int)),
                'min_child_weight': hp.choice('min_child_weight', np.arange(10, 100, 10, dtype=int)),
                'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
                'verbose':1,
                'metric':'auc',
                'objective':'binary',
                'early_stopping_rounds':50,
                'num_boost_round':100000,
                'seed':1234
                }


        tpe_algorithm = tpe.suggest
        bayes_trials = Trials()

        best = fmin(fn = objective, space = space, algo = tpe_algorithm, 
            max_evals = n_iter, trials = bayes_trials)

        best_values = space_eval(space, best)

        print('Best combination of parameters is:')
        print(best_values)
        print('')
        return best_values
