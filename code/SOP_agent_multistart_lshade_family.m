function result = SOP_agent_multistart_lshade_family(problem, seed, options)
% Multi-start L-SHADE family runner.
%
% Several independent L-SHADE or L-SHADE-CMA runs share the total function
% evaluation budget. The best run is retained.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
num_starts = get_option(options, 'num_starts', 3);
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
algorithm = lower(string(get_option(options, 'base_algorithm', 'lshade')));
base_seed = seed;
if isempty(base_seed)
    base_seed = randi(1000000);
end

budget_per_start = max(1000, floor(max_fes / num_starts));
best = [];
total_evals = 0;
all_curve = [];
for k = 1:num_starts
    remaining_time = max_runtime_sec - toc(t_start);
    if remaining_time <= 0
        break;
    end
    local_options = options;
    local_options.max_fes = budget_per_start;
    local_options.max_runtime_sec = remaining_time;
    local_options.verbose = false;
    sub_seed = double(base_seed) + 9973 * (k - 1);
    switch algorithm
        case "lshade_cma"
            current = SOP_agent_lshade_cma(problem, sub_seed, local_options);
            family_name = 'L-SHADE with elite covariance sampling';
        case "lshade_jso"
            current = SOP_agent_lshade_jso(problem, sub_seed, local_options);
            family_name = 'jSO/L-SHADE staged parameter adaptation';
        otherwise
            current = SOP_agent_lshade(problem, sub_seed, local_options);
            family_name = 'L-SHADE success-history adaptation';
    end
    total_evals = total_evals + current.evaluation_count;
    all_curve = [all_curve; current.convergence_curve(:)]; %#ok<AGROW>
    if isempty(best) || current.record_value < best.record_value
        best = current;
    end
end
if isempty(best)
    fallback_options = options;
    fallback_options.max_fes = min(max_fes, max(1000, get_option(options, 'population_num', 180)));
    fallback_options.max_runtime_sec = max(1, max_runtime_sec);
    best = SOP_agent_lshade(problem, base_seed, fallback_options);
    total_evals = best.evaluation_count;
    all_curve = best.convergence_curve(:);
end

runtime = toc(t_start);
result = best;
result.runtime = runtime;
result.evaluation_count = total_evals;
result.convergence_curve = all_curve;
result.iteration = best.iteration * num_starts;
result.algorithm_combination = sprintf('Differential Evolution (DE)\nMulti-start %s', family_name);
result.combination_number = 4;
result.agent_id = 'Agent1';
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
