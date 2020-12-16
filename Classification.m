% Classification data
data = readtable('.\Data\Wine\refined_data.csv', 'PreserveVariableNames', true); 
data{60: end,14} = 0;
data = data(randperm(size(data,1)), :);

training_x = data(1:100, 1:13);
training_y = data(1:100, 14);
 
testing_x = data(101:end, 1:13);
testing_y = table2array(data(101:end, 14));
% polynomial_model = fitcsvm(training_x, training_y, 'Standardize',true,'KernelFunction','Polynomial','OptimizeHyperparameters','Polynomial');

% Binary Classification Model(s)
% linear_model = fitcsvm(training_x, training_y, 'KernelFunction','linear', 'BoxConstraint',1);
% linear_preds = predict(linear_model, testing_x);
% f1_score = F1_Score_Test(linear_preds, testing_y);


% rbf_model = fitcsvm(training_x, training_y, 'KernelFunction','rbf', 'BoxConstraint',1, 'KernelScale', 100);
% linear_preds = predict(linear_model, testing_x);
% f1_score = F1_Score_Test(linear_preds, testing_y);

% polynomial_model = fitcsvm(training_x, training_y, 'KernelFunction','polynomial', 'PolynomialOrder', 2, 'BoxConstraint', 1);
% linear_preds = predict(polynomial_model, testing_x);
% f1_score = F1_Score_Test(linear_preds, testing_y);

Ten_Fold_CV(data);

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
            
%             fprintf('=================================================\n');
%             fprintf('Predicted: %d\n', predicted_value(z));
%             fprintf('Actual: %d\n', actual_value(z));
        
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

        f1_score = 2 * ((precision * recall) / (precision + recall));

        fprintf('Precision: %.2f\n', precision);
        fprintf('Recall: %.2f\n', recall);
        fprintf('F1 - Score: %.2f\n', f1_score);
    end
end


function Ten_Fold_CV(dataset)
   
    polynomial_order_param = [1, 2, 3, 4];
    box_constraint_param = [0.5, 1, 1.5, 2];
    [h, t] = size(polynomial_order_param);
    [m, n] = size(dataset);
    fold_size = ceil(m/10);
    best_acc = 0;
    num_node = 0;
    
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
          
            
            s = ((k * inner_fold_size)+1)
            e = ((k+1) * inner_fold_size)
           
                
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
                    
                    polynomial_model = fitcsvm(inner_training_fold(:, 1:13), inner_training_fold(:, 14),'Standardize',true, 'KernelFunction','polynomial', 'PolynomialOrder', polynomial_order_param(d), 'BoxConstraint', box_constraint_param(g));

                    polynomial_preds = predict(polynomial_model, validation_fold(:,1:13));
                    validation_fold_y = table2array(validation_fold(:, 14));
                    acc = F1_Score_Test(polynomial_preds, validation_fold_y);
                    
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
    
    fprintf('============================== THE ULTIMATE BEST MODEL ==============================\n')
    fprintf('ACCURACY: %d \n', global_best_acc);
    fprintf('Best Model\n');
    disp(global_best_model);
end