function result = SOP_agent_gsk_cmaes_relay(problem, seed, options)
% GSK basin discovery followed by an independent CMA-ES distribution relay.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if isempty(seed)
    seed = randi(1000000);
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
base_runtime_sec = min(max_runtime_sec, get_option(options, 'base_runtime_sec', 230));

gsk_options = options;
gsk_options.max_runtime_sec = base_runtime_sec;
gsk_options.max_fes = max_fes;
gsk_options.max_iter = max(1, floor((max_fes - get_option(options, 'population_num', 180)) / ...
    get_option(options, 'population_num', 180)));
gsk_options.knowledge_rate = get_option(options, 'knowledge_rate', 0.72);
gsk_options.knowledge_factor = get_option(options, 'knowledge_factor', 0.38);
gsk_options.verbose = false;
gsk = SOP_agent_gsk(problem, double(seed), gsk_options);

remaining_fes = max(0, max_fes - gsk.evaluation_count);
remaining_time = max(0, max_runtime_sec - toc(t_start));
cma = [];
if remaining_fes > 0 && remaining_time > 1
    cma_options = struct();
    cma_options.population_num = get_option(options, 'cma_population_num', 36);
    cma_options.max_fes = remaining_fes;
    cma_options.max_runtime_sec = remaining_time;
    cma_options.initial_point = gsk.best_position;
    cma_options.initial_population = gsk.final_population;
    cma_options.seed_covariance = true;
    cma_options.covariance_seed_count = min(size(gsk.final_population, 1), ...
        get_option(options, 'covariance_seed_count', 80));
    cma_options.seed_covariance_blend = get_option(options, 'seed_covariance_blend', 0.42);
    cma_options.seed_covariance_ridge = get_option(options, 'seed_covariance_ridge', 0.18);
    cma_options.sigma0 = get_option(options, 'cma_sigma0', 0.10);
    cma_options.restart_sigma = get_option(options, 'cma_restart_sigma', 0.045);
    cma_options.restart_limit = get_option(options, 'cma_restart_limit', 2);
    cma_options.preserve_covariance_on_restart = true;
    cma_options.eig_interval = get_option(options, 'cma_eig_interval', 8);
    cma_options.verbose = false;
    cma = SOP_agent_cma_es(problem, double(seed) + 104729, cma_options);
end

result = gsk;
if ~isempty(cma) && cma.record_value < result.record_value
    result = cma;
end
result.runtime = toc(t_start);
if isempty(cma)
    result.evaluation_count = gsk.evaluation_count;
    result.iteration = gsk.iteration;
    result.convergence_curve = gsk.convergence_curve(:);
    result.raw_convergence_curve = raw_curve_for(gsk);
else
    result.evaluation_count = gsk.evaluation_count + cma.evaluation_count;
    result.iteration = gsk.iteration + cma.iteration;
    result.convergence_curve = [gsk.convergence_curve(:); cma.convergence_curve(:)];
    result.raw_convergence_curve = [raw_curve_for(gsk); raw_curve_for(cma)];
end
result.algorithm_combination = sprintf(['Gaining-Sharing Knowledge (GSK) basin discovery\n' ...
    'Covariance Matrix Adaptation Evolution Strategy distribution relay']);
result.combination_number = 2;
result.agent_id = 'Agent1';
end

function curve = raw_curve_for(result)
if isfield(result, 'raw_convergence_curve') && ~isempty(result.raw_convergence_curve)
    curve = result.raw_convergence_curve(:);
else
    curve = result.convergence_curve(:);
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
