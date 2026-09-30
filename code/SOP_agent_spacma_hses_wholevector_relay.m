function result = SOP_agent_spacma_hses_wholevector_relay(problem, seed, options)
% SPACMA basin search followed by whole-vector HSES sampling refinement.
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
spacma_options.max_runtime_sec = max(1, min(0.62 * max_runtime_sec, max_runtime_sec - 2));
spacma_options.memory_size = 5;
spacma_options.p_rate = 0.11;
spacma_options.archive_rate = 1.4;
spacma_options.class_learning_rate = 0.8;
spacma_options.sigma_init = 0.5;
spacma_options.verbose = false;
spacma = SOP_agent_lshade_spacma_faithful(problem, double(seed), spacma_options);

remaining_fes = max(1000, max_fes - spacma.evaluation_count);
hses_options = struct();
hses_options.population_num = get_option(options, 'hses_population_num', 48);
hses_options.max_fes = remaining_fes;
hses_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
hses_options.initial_point = spacma.best_position;
hses_options.initial_population = spacma.final_population;
hses_options.initial_radius = get_option(options, 'hses_initial_radius', 0.025);
hses_options.preserve_initial_population_after_radius = true;
hses_options.elite_rate = 0.34;
hses_options.sigma0 = get_option(options, 'hses_sigma0', 0.0040);
hses_options.min_sigma = 1e-8;
hses_options.max_sigma = 0.045;
hses_options.covariance_sample_rate = 0.52;
hses_options.univariate_sample_rate = 0.36;
hses_options.hybrid_mask_rate = 0.28;
hses_options.cov_scale = 0.88;
hses_options.uni_scale = 0.96;
hses_options.best_blend = 0.20;
hses_options.elite_recomb_rate = 0.08;
hses_options.best_pull_rate = 0.18;
hses_options.reset_stall = 24;
hses_options.reset_sigma = 0.006;
hses_options.reset_rate = 0.25;
hses_options.verbose = false;
hses = SOP_agent_hses_sampling(problem, double(seed) + 104729, hses_options);

if hses.record_value < spacma.record_value
    result = hses;
else
    result = spacma;
end
result.runtime = toc(t_start);
result.evaluation_count = spacma.evaluation_count + hses.evaluation_count;
result.iteration = spacma.iteration + hses.iteration;
result.convergence_curve = [spacma.convergence_curve(:); hses.convergence_curve(:)];
result.raw_convergence_curve = [spacma.raw_convergence_curve(:); hses.raw_convergence_curve(:)];
result.population_num = spacma_options.population_num;
result.algorithm_combination = sprintf(['L-SHADE-SPACMA whole-vector basin search\n' ...
    'HSES covariance and univariate Gaussian sampling relay\n' ...
    'Elite population preservation without coordinate crossover']);
result.combination_number = 4;
result.agent_id = 'Agent1';
result.problem = problem;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
