function fitness = SOP_cec_evaluate(x, problem)
% Evaluate a CEC objective through the local wrapper interface.
%
% The official CEC MEX functions use D-by-N matrices. The local wrappers
% also accept N-by-D matrices, but CEC2014/CEC2017 dimensions can be
% ambiguous when the population size is itself a valid dimension (for
% example 50-by-30). Because our optimizers store populations as N-by-D and
% the problem dimension is known here, transpose such matrices explicitly.
if ~isvector(x) && size(x, 2) == problem.dimension
    x_eval = x';
else
    x_eval = x;
end
switch upper(problem.suite)
    case 'CEC2014'
        if ismember(problem.func_num, [17 18 19 20 21 22 29 30])
            fitness = CEC2014_second_evaluate(x_eval, problem.func_num);
        else
            fitness = CEC2014_evaluate(x_eval, problem.func_num);
        end
    case 'CEC2017'
        fitness = CEC2017_evaluate(x_eval, problem.func_num);
    case 'CEC2019'
        fitness = CEC2019_evaluate(x_eval, problem.func_num);
    otherwise
        error('SOP_cec_evaluate:InvalidSuite', 'Unsupported suite: %s.', problem.suite);
end
fitness = double(fitness(:));
end
