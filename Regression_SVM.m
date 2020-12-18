function RMSE = Regression_SVM(path,kernel)
    data = readtable(path);
    training_data = data(1:1000, :);
    testing_data = data(1000:end, :);
    testing_labels = table2array(testing_data);
    fold_size = 100; 
    if strcmp(kernel,'linear')
        % Box constraint
        param_list = [0.2,0.4,0.6,0.8,1];
    elseif strcmp(kernel,'rbf')
        % sigma
        param_list = [0.001,0.01,0.1,1.0,100];
    else
        % polynomial order
        param_list = [1,2,3,4,5];
    end
    outer_RMSE = 0;
    outer_param = 0;
    outer_epsilon = 0;
    outer_SV = 0;
    for inner_fold = 0:9
        best_RMSE = 0;
        best_param = 0;
        best_epsilon = 0;
        best_SV = 0;
        fprintf("fold %d\n",inner_fold);
        lower_fold = (inner_fold*fold_size)+1;
        upper_fold = ((inner_fold+1)*fold_size);

        lower_training_data = training_data(1:lower_fold,:);
        upper_training_data = training_data(upper_fold:end,:);
        inner_training_data = [lower_training_data;upper_training_data];
        inner_testing_data = training_data(lower_fold:upper_fold,:);

        inner_testing_labels = table2array(inner_testing_data);
        for i=1:5
            param = param_list(i);
            for epsilon=0:0.2:1
                if strcmp(kernel,'linear')
                    model = fitrsvm(inner_training_data(:,1:5),inner_training_data(:,end),'KernelFunction','linear', 'BoxConstraint',param,'Epsilon',epsilon);
                elseif strcmp(kernel,'rbf')
                    model = fitrsvm(inner_training_data(:,1:5),inner_training_data(:,end), 'KernelFunction','rbf', 'BoxConstraint',1,'Epsilon',epsilon, 'KernelScale', 1/sqrt(param));
                else
                    model = fitrsvm(inner_training_data(:,1:5),inner_training_data(:,end), 'KernelFunction','polynomial', 'PolynomialOrder', param, 'BoxConstraint', 1,'Epsilon',epsilon);
                end

                [SV, a] = size(model.SupportVectors);
                svmLabel = predict(model, inner_testing_data);
                MSE = mean((svmLabel - inner_testing_labels(:,end)).^2);
                RMSE = sqrt(MSE);

                %fprintf("Param: %f, Epsilon: %f, SV: %d, RMSE: %f \n",param, epsilon, SV, RMSE);
                
                if RMSE < best_RMSE || best_RMSE == 0
                    if SV ~= 0
                        best_RMSE = RMSE;
                        best_model = model;
                        best_param = param;
                        best_epsilon = epsilon;
                        best_SV = SV;
                    end
                end
            end
        end
        fprintf("Best Param: %f, Best Epsilon: %f, Best SV: %d, Best RMSE: %f \n",best_param, best_epsilon, best_SV, best_RMSE);
        svmLabel = predict(best_model, testing_data);
        MSE = mean((svmLabel - testing_labels(:,end)).^2);
        RMSE = sqrt(MSE);
        fprintf("TESTING RMSE: %f\n",RMSE);
        if outer_RMSE == 0 || RMSE < outer_RMSE
            outer_RMSE = RMSE;
            outer_param = best_param;
            outer_epsilon = best_epsilon;
            outer_SV = best_SV;
        end
    end
    fprintf("==================================FINAL BEST=========================================\n");
    fprintf("Best testing RMSE: %d\n", outer_RMSE);
    fprintf("Best param: %d\n", outer_param);
    fprintf("Best epsilon: %d\n", outer_epsilon);
    fprintf("SV selected: %d\n", outer_SV);
end


