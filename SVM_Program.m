% Selection between RBF or Polynomial Kernel
path = '.\Data\Wine\refined_data.csv';
i = 1;
while i == 1
    fprintf('Please select the SVM Kernel Type\n');
    fprintf('0 - Polynomial\n');
    fprintf('1 - RBF\n');
    fprintf('2 - Linear\n');
    fprintf('3 - Exit\n');
    user_ans = input('>>> ');
 
    if user_ans == 0
        [retrained_model, f1_score] = Classification_SVM(path, 'polynomial');
        continue;
        
    elseif user_ans == 1
        [retrained_model, f1_score] = Classification_SVM(path, 'rbf');
        continue;
    
    elseif user_ans == 2
        [retrained_model, f1_score] = Classification_SVM(path, 'linear');
    elseif user_ans == 3
        i = 0;
        return;
        
    else
        disp('Please provide a valid input')
        continue;
    
    end
end