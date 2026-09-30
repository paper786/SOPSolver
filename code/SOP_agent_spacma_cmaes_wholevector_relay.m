function result = SOP_agent_spacma_cmaes_wholevector_relay(problem, seed, options)
% SPACMA basin search followed by covariance-seeded CMA-ES refinement.
if nargin < 2 || isempty(seed)
    seed = randi(1000000);
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 15000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', 295);
spacma_fes = min(max_fes - 1000, get_option(options, 'spacma_fes', 10000 * problem.dimension));

spacma_options = options;
spacma_options.population_num = get_option(options, 'population_num', 18 * problem.dimension);
spacma_options.min_population_num = 4;
spacma_options.max_fes = spacma_fes;
spacma_options.max_runtime_sec = max(1, min(0.64 * max_runtime_sec, max_runtime_sec - 2));
spacma_options.memory_size = 5;
spacma_options.p_rate = 0.11;
spacma_options.archive_rate = 1.4;
spacma_options.class_learning_rate = 0.8;
spacma_options.sigma_init = 0.5;
spacma_options.verbose = false;
spacma = SOP_agent_lshade_spacma_faithful(problem, double(seed), spacma_options);

remaining_fes = max(1000, max_fes - spacma.evaluation_count);
cma_options = struct();
cma_options.population_num = get_option(options, 'cma_population_num', 64);
cma_options.max_fes = remaining_fes;
cma_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
cma_options.initial_point = spacma.best_position;
cma_options.initial_population = spacma.final_population;
cma_options.seed_covariance = true;
cma_options.covariance_seed_count = min(size(spacma.final_population, 1), ...
    get_option(options, 'covariance_seed_count', 160));
cma_options.seed_covariance_blend = 0.68;
cma_options.seed_covariance_ridge = 0.06;
cma_options.seed_covariance_min_eig = 0.03;
cma_options.seed_covariance_max_eig = 14;
cma_options.sigma0 = get_option(options, 'cma_sigma0', 0.0080);
cma_options.min_sigma = 1e-9;
cma_options.restart_sigma = get_option(options, 'cma_restart_sigma', 0.0030);
cma_options.restart_limit = 2;
cma_options.eig_interval = 8;
cma_options.preserve_covariance_on_restart = true;
cma_options.verbose = false;
cma = SOP_agent_cma_es(problem, double(seed) + 104729, cma_options);

if cma.record_value < spacma.record_value
    result = cma;
else
    result = spacma;
end
result.runtime = toc(t_start);
result.evaluation_count = spacma.evaluation_count + cma.evaluation_count;
result.iteration = spacma.iteration + cma.iteration;
result.convergence_curve = [spacma.convergence_curve(:); cma.convergence_curve(:)];
result.raw_convergence_curve = [spacma.raw_convergence_curve(:); cma.raw_convergence_curve(:)];
result.population_num = spacma_options.population_num;
result.algorithm_combination = sprintf(['L-SHADE-SPACMA whole-vector basin search\n' ...
    'Elite-covariance-seeded CMA-ES refinement\n' ...
    'Restarted rank-based Gaussian distribution adaptation']);
result.combination_number = 4;
result.agent_id = 'Agent2';
result.problem = problem;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
