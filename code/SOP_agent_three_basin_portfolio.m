function result = SOP_agent_three_basin_portfolio(problem, seed, options)
% Independent SPACMA, L-SHADE, and MVO whole-basin portfolio.
if nargin < 2 || isempty(seed)
    seed = randi(1000000);
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 30000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', 295);
stage_fes = floor(max_fes / 3);

spacma_options = struct();
spacma_options.population_num = 18 * problem.dimension;
spacma_options.min_population_num = 4;
spacma_options.max_fes = stage_fes;
spacma_options.max_runtime_sec = min(105, 0.38 * max_runtime_sec);
spacma_options.memory_size = 5;
spacma_options.p_rate = 0.11;
spacma_options.archive_rate = 1.4;
spacma_options.class_learning_rate = 0.8;
spacma_options.sigma_init = 0.5;
spacma_options.verbose = false;
spacma = SOP_agent_lshade_spacma_faithful(problem, double(seed), spacma_options);

lshade_options = struct();
lshade_options.population_num = get_option(options, 'lshade_population_num', 360);
lshade_options.min_population_num = 4;
lshade_options.max_fes = stage_fes;
lshade_options.max_runtime_sec = min(90, max(1, max_runtime_sec - toc(t_start) - 2));
lshade_options.p_rate = 0.10;
lshade_options.inloop_soft_restart = true;
lshade_options.soft_restart_stall_iter = 26;
lshade_options.soft_restart_worst_rate = 0.17;
lshade_options.soft_restart_radius = 0.0055;
lshade_options.soft_restart_reset_radius = 0.012;
lshade_options.soft_restart_min_radius = 0.00010;
lshade_options.soft_restart_cauchy_rate = 0.34;
lshade_options.soft_restart_wide_rate = 0.16;
lshade_options.verbose = false;
lshade = SOP_agent_lshade(problem, double(seed) + 65537, lshade_options);

remaining_fes = max(1000, max_fes - spacma.evaluation_count - lshade.evaluation_count);
mvo_options = struct();
mvo_options.method = 'mvo';
mvo_options.population_num = get_option(options, 'mvo_population_num', 260);
mvo_options.max_fes = remaining_fes;
mvo_options.max_runtime_sec = max(1, max_runtime_sec - toc(t_start));
mvo_options.max_iter = max(1, floor((remaining_fes - mvo_options.population_num) / mvo_options.population_num));
mvo_options.verbose = false;
mvo = SOP_agent_literature_swarm(problem, double(seed) + 104729, mvo_options);

members = {spacma, lshade, mvo};
values = [spacma.record_value, lshade.record_value, mvo.record_value];
labels = {'L-SHADE-SPACMA', 'L-SHADE soft-restart', 'Multi-Verse Optimizer'};
[~, selected] = min(values);
result = members{selected};
result.runtime = toc(t_start);
result.evaluation_count = spacma.evaluation_count + lshade.evaluation_count + mvo.evaluation_count;
result.iteration = spacma.iteration + lshade.iteration + mvo.iteration;
result.convergence_curve = [spacma.convergence_curve(:); lshade.convergence_curve(:); mvo.convergence_curve(:)];
result.raw_convergence_curve = [raw_curve_for(spacma); raw_curve_for(lshade); raw_curve_for(mvo)];
result.population_num = spacma_options.population_num + lshade_options.population_num + mvo_options.population_num;
result.algorithm_combination = sprintf(['Independent L-SHADE-SPACMA whole-basin search\n' ...
    'Independent L-SHADE soft-restart whole-basin search\n' ...
    'Independent Multi-Verse Optimizer whole-basin search\n' ...
    'Objective-based selector retained %s'], labels{selected});
result.combination_number = 4;
result.agent_id = 'EvaluateAgent';
result.problem = problem;
end

function curve = raw_curve_for(result)
if isfield(result, 'raw_convergence_curve') && ~isempty(result.raw_convergence_curve)
    curve = result.raw_convergence_curve(:);
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
