function result = SOP_agent_mvo_spacma_wholevector_relay(problem, seed, options)
% Whole-vector MVO basin discovery followed by SPACMA refinement.
if nargin < 2 || isempty(seed)
    seed = randi(1000000);
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 3 * 10000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', 295);
NP = get_option(options, 'population_num', 420);
mvo_fraction = get_option(options, 'mvo_fraction', 0.34);
mvo_fes = max(NP, floor(mvo_fraction * max_fes));

mvo_options = options;
mvo_options.method = 'mvo';
mvo_options.population_num = NP;
mvo_options.max_fes = mvo_fes;
mvo_options.max_runtime_sec = max(1, min(0.38 * max_runtime_sec, max_runtime_sec - 2));
mvo_options.max_iter = max(1, floor((mvo_fes - NP) / NP));
mvo_options.verbose = false;
mvo = SOP_agent_literature_swarm(problem, double(seed), mvo_options);

remaining_fes = max(1000, max_fes - mvo.evaluation_count);
remaining_time = max(1, max_runtime_sec - toc(t_start));
spacma_options = options;
spacma_options.population_num = get_option(options, 'spacma_population_num', 18 * problem.dimension);
spacma_options.min_population_num = 4;
spacma_options.max_fes = remaining_fes;
spacma_options.max_runtime_sec = remaining_time;
spacma_options.memory_size = 5;
spacma_options.p_rate = get_option(options, 'spacma_p_rate', 0.11);
spacma_options.archive_rate = 1.4;
spacma_options.class_learning_rate = 0.8;
spacma_options.sigma_init = 0.5;
spacma_options.initial_point = mvo.best_position;
spacma_options.initial_radius = get_option(options, 'handoff_radius', 0.10);
spacma_options.initial_anchor_rate = get_option(options, 'handoff_anchor_rate', 0.42);
spacma_options.initial_cauchy = true;
if isfield(mvo, 'final_population') && ~isempty(mvo.final_population)
    spacma_options.initial_population = mvo.final_population;
    spacma_options.preserve_initial_population = true;
end
spacma_options.verbose = false;
spacma = SOP_agent_lshade_spacma_faithful(problem, double(seed) + 7919, spacma_options);

if spacma.record_value < mvo.record_value
    result = spacma;
else
    result = mvo;
end
result.runtime = toc(t_start);
result.evaluation_count = mvo.evaluation_count + spacma.evaluation_count;
result.iteration = mvo.iteration + spacma.iteration;
result.convergence_curve = [mvo.convergence_curve(:); spacma.convergence_curve(:)];
result.raw_convergence_curve = [raw_curve_for(mvo); raw_curve_for(spacma)];
result.population_num = NP + spacma_options.population_num;
result.algorithm_combination = sprintf(['Multi-Verse Optimizer whole-vector basin discovery\n' ...
    'Elite population and intact best-vector relay\n' ...
    'L-SHADE-SPACMA success-history DE/CMA refinement']);
result.combination_number = 4;
result.agent_id = 'Agent2';
result.problem = problem;
end

function curve = raw_curve_for(result)
if isfield(result, 'raw_convergence_curve') && ~isempty(result.raw_convergence_curve)
    curve = result.raw_convergence_curve(:);
elseif isfield(result, 'convergence_curve') && ~isempty(result.convergence_curve)
    curve = result.convergence_curve(:);
else
    curve = result.best_value;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
