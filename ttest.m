X = [];
Y = [];

fprintf('Please enter an array of accuracies for algorithm 1 used, separated by commas\n');
X = input('>>>');
fprintf('Please enter an array of accuracies for algorithm 2 used, separated by commas\n');
Y = input('>>>');

diff = ttest2(X,Y);
if diff == 0
    fprintf("Algorithm 1 is not statistically significancly different than algorithm 2\n ");
elseif diff ==1
    fprintf("Algorithm 1 is statistically significancly different than algorithm 2\n ");
end
        
