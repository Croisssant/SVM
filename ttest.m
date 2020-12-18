X = [];
Y = [];

fprintf('Please enter an array of accuracies for algorithm 1 used, in ["value1","value2"] format \n');
X = input('>>>');
fprintf('Please enter an array of accuracies for algorithm 2 used, in ["value1","value2"] format \n');
Y = input('>>>');

diff = ttest2(X,Y);
if diff == 1
    fprintf("Algorithm 1 is statistically significancly different than algorithm 2\n ");
else
    fprintf("Algorithm 1 is not statistically significancly different than algorithm 2\n ");
end
        
