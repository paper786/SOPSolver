function result = SOP_agent_multistart_sade(problem, seed, options)
% Multi-start self-adaptive Differential Evolution.
% Several independent SaDE runs are launched with deterministic seed
% offsets, and the best run is retained. This is useful for highly
% multimodal cases where a single population can settle in a local basin.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
num_starts = get_option(options, 'num_starts', 3);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
base_seed = seed;
if isempty(base_seed)
    base_seed = randi(1000000);
end

best = [];
total_evals = 0;
all_curve = [];
for k = 1:num_starts
    remaining_time = max_runtime_sec - toc(t_start);
    if remaining_time <= 0
        break;
    end
    local_options = options;
    local_options.max_runtime_sec = remaining_time;
    local_options.verbose = false;
    sub_seed = double(base_seed) + 9973 * (k - 1);
    current = SOP_agent1_adaptive_de(problem, sub_seed, local_options);
    total_evals = total_evals + current.evaluation_count;
    all_curve = [all_curve; current.convergence_curve(:)]; %#ok<AGROW>
    if isempty(best) || current.record_value < best.record_value
        best = current;
    end
end
if isempty(best)
    fallback_options = options;
    fallback_options.max_iter = 1;
    fallback_options.max_runtime_sec = max(1, max_runtime_sec);
    best = SOP_agent1_adaptive_de(problem, base_seed, fallback_options);
    total_evals = best.evaluation_count;
    all_curve = best.convergence_curve(:);
end

runtime = toc(t_start);
result = best;
result.runtime = runtime;
result.evaluation_count = total_evals;
result.convergence_curve = all_curve;
result.iteration = best.iteration * num_starts;
result.algorithm_combination = sprintf('Self-Adaptive Differential Evolution (SaDE)\nMulti-start DE/current-to-pbest with archive');
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
