function result = SOP_agent_metaheuristic_lshade_relay(problem, seed, options)
% Metaheuristic scout followed by basin-centered soft-restart L-SHADE.
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
max_fes = get_option(options, 'max_fes', 30000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', 295);
scout_fes = min(max_fes, get_option(options, 'scout_fes', 10000 * problem.dimension));
scout_key = lower(string(get_option(options, 'scout_algorithm', 'spacma')));

scout_options = options;
scout_options.population_num = get_option(options, 'scout_population_num', 18 * problem.dimension);
scout_options.max_fes = scout_fes;
scout_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'scout_runtime_sec', 0.42 * max_runtime_sec));
scout_options.memory_size = 5;
scout_options.min_population_num = 4;
scout_options.p_rate = 0.11;
scout_options.archive_rate = 1.4;
scout_options.verbose = false;
switch scout_key
    case "cnepsin"
        scout_options.eigen_rate = 0.4;
        scout_options.neighbor_rate = 0.5;
        scout_options.learning_period = 20;
        scout_options.initial_frequency = 0.5;
        scout = SOP_agent_lshade_cnepsin(problem, double(seed), scout_options);
        scout_label = 'Ensemble sinusoidal L-SHADE with neighborhood eigen crossover';
    case "cnepsin_faithful"
        scout_options.eigen_rate = 0.4;
        scout_options.neighbor_rate = 0.5;
        scout_options.learning_period = 20;
        scout_options.initial_frequency = 0.5;
        scout = SOP_agent_lshade_cnepsin_faithful(problem, double(seed), scout_options);
        scout_label = 'Conditioned neighborhood cnEpSin search';
    otherwise
        scout_options.class_learning_rate = 0.8;
        scout_options.sigma_init = 0.5;
        scout = SOP_agent_lshade_spacma_faithful(problem, double(seed), scout_options);
        scout_label = 'Semi-adaptive L-SHADE/SPACMA search';
end

remaining_fes = max(1000, max_fes - scout.evaluation_count);
remaining_time = max(1, max_runtime_sec - toc(t_start));
refine_options = struct();
refine_options.population_num = get_option(options, 'refine_population_num', 300);
refine_options.min_population_num = 4;
refine_options.max_fes = remaining_fes;
refine_options.max_runtime_sec = remaining_time;
refine_options.p_rate = get_option(options, 'refine_p_rate', 0.08);
refine_options.initial_point = scout.best_position;
refine_options.initial_radius = get_option(options, 'refine_radius', 0.018);
refine_options.initial_cauchy = true;
refine_options.inloop_soft_restart = true;
refine_options.soft_restart_stall_iter = get_option(options, 'soft_restart_stall_iter', 24);
refine_options.soft_restart_worst_rate = get_option(options, 'soft_restart_worst_rate', 0.16);
refine_options.soft_restart_radius = get_option(options, 'soft_restart_radius', 0.005);
refine_options.soft_restart_reset_radius = get_option(options, 'soft_restart_reset_radius', 0.014);
refine_options.soft_restart_min_radius = get_option(options, 'soft_restart_min_radius', 0.00010);
refine_options.soft_restart_cauchy_rate = get_option(options, 'soft_restart_cauchy_rate', 0.34);
refine_options.soft_restart_wide_rate = get_option(options, 'soft_restart_wide_rate', 0.18);
refine_options.verbose = false;
refine = SOP_agent_lshade(problem, double(seed) + 65537, refine_options);

if refine.record_value < scout.record_value
    result = refine;
else
    result = scout;
end
result.runtime = toc(t_start);
result.evaluation_count = scout.evaluation_count + refine.evaluation_count;
result.iteration = scout.iteration + refine.iteration;
result.convergence_curve = [scout.convergence_curve(:); refine.convergence_curve(:)];
result.raw_convergence_curve = [scout.raw_convergence_curve(:); refine.raw_convergence_curve(:)];
result.population_num = scout_options.population_num;
result.algorithm_combination = sprintf(['%s\n' ...
    'Basin-centered L-SHADE relay\n' ...
    'In-loop soft restart of the worst population fraction'], scout_label);
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
