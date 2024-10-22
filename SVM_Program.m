%This is the main program to run different kernel for classification and
%regression.

i = 1;
while i == 1
    fprintf('Please select classification or regression\n');
    fprintf('0 - Classification\n');
    fprintf('1 - Regression\n');
    type = input('>>>');
    if type == 0
        path = '.\Data\Wine\refined_data.csv';
    elseif type ==1
        path = '.\Data\Airfoil_self_noise\airfoil_self_noise.dat';
    end
    fprintf('Please select the SVM Kernel Type\n');
    fprintf('0 - Polynomial\n');
    fprintf('1 - RBF\n');
    fprintf('2 - Linear\n');
    fprintf('3 - Exit\n');
    user_ans = input('>>> ');
 
    if user_ans == 0
        if type == 0
            [retrained_model, f1_score] = Classification_SVM(path, 'polynomial');
        elseif type ==1
            RMSE = Regression_SVM(path,'polynomial');
        end
        
    elseif user_ans == 1
        if type == 0
            [retrained_model, f1_score] = Classification_SVM(path, 'rbf');
        elseif type ==1
            RMSE = Regression_SVM(path,'rbf');
        end
    elseif user_ans == 2
        if type == 0
            [retrained_model, f1_score] = Classification_SVM(path, 'linear');
        elseif type ==1
            RMSE = Regression_SVM(path,'linear');
        end
    elseif user_ans == 3
        i = 0;
        return;        
    else
        disp('Please provide a valid input')
        continue;
    end
end