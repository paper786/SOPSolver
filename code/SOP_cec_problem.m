function problem = SOP_cec_problem(suite, func_num, dimension)
% Build a CEC problem descriptor without reading benchmark internals.
suite = upper(string(suite));
func_num = double(func_num);
dimension = double(dimension);

if suite == "CEC2014"
    assert(func_num >= 1 && func_num <= 30 && func_num == floor(func_num), ...
        'SOP_cec_problem:InvalidFunction', 'CEC2014 function number must be 1..30.');
    assert(any(dimension == [2 10 20 30 50 100]), ...
        'SOP_cec_problem:InvalidDimension', 'CEC2014 dimension must be 2, 10, 20, 30, 50, or 100.');
    lb = -100 * ones(1, dimension);
    ub = 100 * ones(1, dimension);
    record_metric = 'cec2014_error';
elseif suite == "CEC2017"
    assert(any(func_num == [1 3:30]), ...
        'SOP_cec_problem:InvalidFunction', 'CEC2017 has F1 and F3-F30; F2 is deleted.');
    assert(any(dimension == [2 10 20 30 50 100]), ...
        'SOP_cec_problem:InvalidDimension', 'CEC2017 dimension must be 2, 10, 20, 30, 50, or 100.');
    lb = -100 * ones(1, dimension);
    ub = 100 * ones(1, dimension);
    record_metric = 'raw';
elseif suite == "CEC2019"
    assert(func_num >= 1 && func_num <= 10 && func_num == floor(func_num), ...
        'SOP_cec_problem:InvalidFunction', 'CEC2019 function number must be 1..10.');
    expected_dim = SOP_cec2019_dimension(func_num);
    assert(dimension == expected_dim, ...
        'SOP_cec_problem:InvalidDimension', 'CEC2019 F%d dimension must be %d.', func_num, expected_dim);
    [lb, ub] = SOP_cec2019_bounds(func_num, dimension);
    record_metric = 'raw';
else
    error('SOP_cec_problem:InvalidSuite', 'Unsupported CEC suite: %s.', suite);
end

problem.suite = char(suite);
problem.func_num = func_num;
problem.dimension = dimension;
problem.lb = lb;
problem.ub = ub;
problem.function_name = sprintf('%s_F%d', char(suite), func_num);
problem.file_stem = sprintf('%s_fun%d_%dD', lower(char(suite)), func_num, dimension);
problem.record_metric = record_metric;
end

function dimension = SOP_cec2019_dimension(func_num)
if func_num == 1
    dimension = 9;
elseif func_num == 2
    dimension = 16;
elseif func_num == 3
    dimension = 18;
else
    dimension = 10;
end
end

function [lb, ub] = SOP_cec2019_bounds(func_num, dimension)
% CEC2019 uses special ranges for the first three functions and [-100,100]
% for F4-F10. These are search bounds, not benchmark solution data.
if func_num == 1
    lower = -8192;
    upper = 8192;
elseif func_num == 2
    lower = -16384;
    upper = 16384;
elseif func_num == 3
    lower = -4;
    upper = 4;
else
    lower = -100;
    upper = 100;
end
lb = lower * ones(1, dimension);
ub = upper * ones(1, dimension);
end
