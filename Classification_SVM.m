function [retrained_model, f1_score] = Classification_SVM(file_path, kernel)

data = readtable(file_path, 'PreserveVariableNames', true); 
data{60: end,14} = 0;
data = data(randperm(size(data,1)), :);

training_x = data(1:100, 1:13);
training_y = data(1:100, 14);
 
testing_x = data(101:end, 1:13);
testing_y = table2array(data(101:end, 14));

if strcmp(kernel, 'linear')
    
    retrained_model = fitcsvm(training_x, training_y,'KernelFunction','linear', 'BoxConstraint', 1);
    model_prediction = predict(retrained_model, testing_x);
    fprintf('\n =================================================================\n');
    fprintf('RESULT of LINEAR MODEL');
    fprintf('\n =================================================================\n');
    f1_score = F1_Score_Test(model_prediction, testing_y);
    [sv_num, sv_col] = size(retrained_model.SupportVectors);
    [train_r, train_c] = size(training_x);
    fprintf('Number of Support Vectors: %d\n', sv_num);
    fprintf('Percentage of Support Vectors out of Training Data: %.2f percent\n', ((sv_num/train_r) * 100))
    fprintf('\n');

else
    
    [all_accuracies, kernel_flag] = Ten_Fold_CV(data, kernel);

    global_best = 0;
    best_param = 's';

    for idx = 1:width(all_accuracies)
        mean_accuracy = mean(all_accuracies{1,idx});
        fprintf('The mean accuracy of %s is %.2f \n', string(all_accuracies.Properties.VariableNames(idx)), mean_accuracy);

        if mean_accuracy > global_best
            global_best = mean_accuracy;
            best_param =  all_accuracies.Properties.VariableNames(idx);
        end
    end

    fprintf('The best parameter is %s with the accuracy of %.2f\n', string(best_param), global_best);

    if kernel_flag == 1
        kernel_param = sscanf(string(best_param),'box%fpoly%f');
        box_val = kernel_param(1);
        poly_val = kernel_param(2);

        fprintf('\nRetraining a new model with the best hyperparameter combination selected...\n');
        retrained_model = fitcsvm(training_x, training_y,'KernelFunction','polynomial', 'PolynomialOrder', poly_val, 'BoxConstraint', box_val);
        model_prediction = predict(retrained_model, testing_x);
        fprintf('\n =================================================================\n');
        fprintf('RESULT of RETRAINED MODEL');
        fprintf('\n =================================================================\n');
        f1_score = F1_Score_Test(model_prediction, testing_y);
        fprintf('Box Constraint: %.2f\n', box_val);
        fprintf('Polynomial Order: %.2f\n', poly_val);
        [sv_num, sv_col] = size(retrained_model.SupportVectors);
        [train_r, train_c] = size(training_x);
        fprintf('Number of Support Vectors: %d\n', sv_num);
        fprintf('Percentage of Support Vectors out of Training Data: %.2f percent\n', ((sv_num/train_r) * 100))
        fprintf('\n');

    elseif kernel_flag == 2
        kernel_param = sscanf(string(best_param),'box%fsigma%f');
        box_val = kernel_param(1);
        sigma_val = kernel_param(2);

        fprintf('\nRetraining a new model with the best hyperparameter combination selected...\n');
        retrained_model = fitcsvm(training_x, training_y,'KernelFunction','rbf', 'KernelScale', sigma_val, 'BoxConstraint', box_val);
        model_prediction = predict(retrained_model, testing_x);
        fprintf('\n =================================================================\n');
        fprintf('RESULT of RETRAINED MODEL');
        fprintf('\n =================================================================\n');
        f1_score = F1_Score_Test(model_prediction, testing_y);
        fprintf('Box Constraint: %.2f\n', box_val);
        fprintf('Kernel Scale (Sigma): %.2f\n', sigma_val);
        [sv_num, sv_col] = size(retrained_model.SupportVectors);
        [train_r, train_c] = size(training_x);
        fprintf('Number of Support Vectors: %d\n', sv_num);
        fprintf('Percentage of Support Vectors out of Training Data: %.2f percent\n', ((sv_num/train_r) * 100))
        fprintf('\n');

    else
        fprintf('WHOOPS! Something is wrong');
    end
end
end

function f1_score = F1_Score_Test(predicted_value, actual_value)
    [m, n] = size(predicted_value);
    [r, c] = size(actual_value);
    
    if (m ~= r | n ~= c)
        fprintf('Please insert the same dimension for both predicted values and actual values\n');
    
    else
        true_positive = 0;
        false_positive = 0;

        true_negative = 0;
        false_negative = 0;

        for z = 1:m
            
        
            if (predicted_value(z) == actual_value(z) & actual_value(z) == 1)
                true_positive = true_positive + 1;

            elseif (predicted_value(z) ~= actual_value(z) & actual_value(z) == 1)
                false_negative = false_negative + 1;

            elseif (predicted_value(z) == actual_value(z) & actual_value(z) == 0)
                true_negative = true_negative + 1;

            elseif (predicted_value(z) ~= actual_value(z) & actual_value(z) == 0)
                false_positive = false_positive + 1;
            end
        end
        
        
       
        precision = true_positive / (true_positive + false_positive);
        recall = true_positive / (true_positive + false_negative);
        
        if isnan(precision)
            precision = 0;
        end
        
        if isnan(recall)
            recall = 0;
        end
        
        if precision + recall == 0
            f1_score = 0;
        else
            f1_score = 2 * ((precision * recall) / (precision + recall));
        end

        fprintf('Precision: %.2f\n', precision);
        fprintf('Recall: %.2f\n', recall);
        fprintf('F1 - Score: %.2f\n', f1_score);
    end
end


function [T, kernel_flag] = Ten_Fold_CV(dataset, kernel)
   
    T = table();
    global_best_acc = 0;
    polynomial_order_param = [1, 2, 3, 4];
    sigma_param = [5,15,25,35];
    box_constraint_param = [0.5, 1, 1.5, 2];
    [h, t] = size(polynomial_order_param);
    [m, n] = size(dataset);
    fold_size = ceil(m/10);
    best_acc = 0;
    num_node = 0;

if strcmp(kernel, 'polynomial')
    kernel_flag = 1;
    for i = 0:9
        fprintf('\n========== This is Fold %d ==========\n', i);
        data = dataset;
        
        l = ((i * fold_size)+1);
        j = ((i+1) * fold_size);
        

        if i == 9
            training_fold = data;
            testing_fold = data(l:end, :);
            training_fold(l:end, :) = [];
        else
            testing_fold = data(l:j, :);
            training_fold = data;
            training_fold(l:j, :) = [];
        end
        
        for k = 0:9
            [r, c] = size(training_fold);
            inner_fold_size = ceil(r/10);
            inner_data = training_fold;
          
            
            s = ((k * inner_fold_size)+1);
            e = ((k+1) * inner_fold_size);
           
                
            if k == 9
                inner_training_fold = inner_data;
                validation_fold = inner_data(s:end, :);
                inner_training_fold(s:end, :) = [];
            else
                validation_fold = inner_data(s:e, :);
                inner_training_fold = inner_data;
                inner_training_fold(s:e, :) = [];
            end
            
            for g = 1:4
                fprintf('================= Box Constraint: %d =================\n', box_constraint_param(g));
                for d = 1:4
                    fprintf('================= Polynomial Order: %d =================\n', polynomial_order_param(d));
                    
                    polynomial_model = fitcsvm(inner_training_fold(:, 1:13), inner_training_fold(:, 14),'KernelFunction','polynomial', 'PolynomialOrder', polynomial_order_param(d), 'BoxConstraint', box_constraint_param(g));

                    polynomial_preds = predict(polynomial_model, validation_fold(:,1:13));
                    validation_fold_y = table2array(validation_fold(:, 14));
                    acc = F1_Score_Test(polynomial_preds, validation_fold_y);

                    box_value = string(box_constraint_param(g));
                    poly_value = string(polynomial_order_param(d));
                    param_combo = strcat('box', box_value, 'poly', poly_value);
                    
                    all_names =  T.Properties.VariableNames;
                    
                    if width(T) == 0
                        T.param_combo = [acc];
                        T.Properties.VariableNames = param_combo;
                    else
                        for param_count = 1:width(T)
                            flag = 0;
                            if strcmp(param_combo , all_names{param_count})
                                flag = 1;
                                T.(param_combo) = [T.(param_combo), acc];
                                break;
                            end
                        end
                        if flag == 0
                            T.param_combo = acc;
                            T.Properties.VariableNames(end) = param_combo;
                        end
                    end
                    
                    if i == 0
                        best_acc = acc;
                        best_poly_param =  polynomial_order_param(d);
                        best_box_const =  box_constraint_param(g);
                        best_model = polynomial_model;
                    end

                    if acc > best_acc 
                        best_acc = acc;
                        best_poly_param =  polynomial_order_param(d);
                        best_box_const =  box_constraint_param(g);
                        best_model = polynomial_model;
                    end
                end
            end
            
            fprintf('============================ This is the best model against the test set ============================\n');
            polynomial_preds = predict(best_model, testing_fold(:,1:13));
            actual_val_y = table2array(testing_fold(:, 14));
            acc = F1_Score_Test(polynomial_preds, actual_val_y);

            if i == 0
                global_best_acc = acc;
                global_best_model = best_model;
            end

            if acc > global_best_acc 
                global_best_acc = acc;
                global_best_model = best_model;
            end
                      
        end
        
    end
    
elseif strcmp(kernel, 'rbf')
    
    kernel_flag = 2;
    for i = 0:9
        fprintf('\n========== This is Fold %d ==========\n', i);
        data = dataset;
        
        l = ((i * fold_size)+1);
        j = ((i+1) * fold_size);
        

        if i == 9
            training_fold = data;
            testing_fold = data(l:end, :);
            training_fold(l:end, :) = [];
        else
            testing_fold = data(l:j, :);
            training_fold = data;
            training_fold(l:j, :) = [];
        end
        
        for k = 0:9
            [r, c] = size(training_fold);
            inner_fold_size = ceil(r/10);
            inner_data = training_fold;
          
            
            s = ((k * inner_fold_size)+1);
            e = ((k+1) * inner_fold_size);
           
                
            if k == 9
                inner_training_fold = inner_data;
                validation_fold = inner_data(s:end, :);
                inner_training_fold(s:end, :) = [];
            else
                validation_fold = inner_data(s:e, :);
                inner_training_fold = inner_data;
                inner_training_fold(s:e, :) = [];
            end
            
            for g = 1:4
                fprintf('================= Box Constraint: %d =================\n', box_constraint_param(g));
                for d = 1:4
                    fprintf('================= Sigma: %d =================\n', sigma_param(d));
                    
                    rbf_model = fitcsvm(inner_training_fold(:, 1:13), inner_training_fold(:, 14),'KernelFunction','rbf', 'KernelScale', sigma_param(d), 'BoxConstraint', box_constraint_param(g));

                    rbf_preds = predict(rbf_model, validation_fold(:,1:13));
                    validation_fold_y = table2array(validation_fold(:, 14));
                    acc = F1_Score_Test(rbf_preds, validation_fold_y);

                    box_value = string(box_constraint_param(g));
                    sigma_value = string(sigma_param(d));
                    param_combo = strcat('box', box_value, 'sigma', sigma_value);
                    
                    all_names =  T.Properties.VariableNames;
                    
                    if width(T) == 0
                        T.param_combo = [acc];
                        T.Properties.VariableNames = param_combo;
                    else
                        for param_count = 1:width(T)
                            flag = 0;
                            if strcmp(param_combo , all_names{param_count})
                                flag = 1;
                                T.(param_combo) = [T.(param_combo), acc];
                                break;
                            end
                        end
                        if flag == 0
                            T.param_combo = acc;
                            T.Properties.VariableNames(end) = param_combo;
                        end
                    end
                    
                    if i == 0
                        best_acc = acc;
                        best_sigma_param =  sigma_param(d);
                        best_box_const =  box_constraint_param(g);
                        best_model = rbf_model;
                    end

                    if acc > best_acc 
                        best_acc = acc;
                        best_sigma_param =  sigma_param(d);
                        best_box_const =  box_constraint_param(g);
                        best_model = rbf_model;
                    end
                end
            end
            
            fprintf('============================ This is the best model against the test set ============================\n');
            rbf_preds = predict(best_model, testing_fold(:,1:13));
            actual_val_y = table2array(testing_fold(:, 14));
            acc = F1_Score_Test(rbf_preds, actual_val_y);

            if i == 0
                global_best_acc = acc;
                global_best_model = best_model;
            end

            if acc > global_best_acc 
                global_best_acc = acc;
                global_best_model = best_model;
            end
                      
        end
        
    end
else
    fprintf('Please insert a valid kernel value');
end
    
    fprintf('============================== THE ULTIMATE BEST MODEL SELECTED FROM CV ==============================\n')
    fprintf('ACCURACY: %d \n', global_best_acc);
    fprintf('Best Model\n');
    disp(global_best_model);
end