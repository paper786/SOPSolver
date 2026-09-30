function result = SOP_agent_spacma_three_island_selector(problem, seed, options)
% Three independent SPACMA islands for three-component composition basins.
if nargin < 2 || isempty(seed)
    seed = randi(1000000);
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 27000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', 295);
island_fes = floor(max_fes / 3);
seeds = double(seed) + [0, 3571, 7919];
p_rates = [0.11, 0.18, 0.075];
archive_rates = [1.4, 2.0, 1.2];
de_probabilities = [0.50, 0.68, 0.34];
labels = {'balanced', 'DE-diversity', 'covariance-dominant'};
islands = cell(1, 3);

for island = 1:3
    remaining_time = max(1, max_runtime_sec - toc(t_start));
    island_options = struct();
    island_options.population_num = 18 * problem.dimension;
    island_options.min_population_num = 4;
    island_options.max_fes = island_fes;
    island_options.max_runtime_sec = min(92, remaining_time);
    island_options.memory_size = 5;
    island_options.p_rate = p_rates(island);
    island_options.archive_rate = archive_rates(island);
    island_options.class_learning_rate = 0.8;
    island_options.initial_de_probability = de_probabilities(island);
    island_options.sigma_init = 0.5;
    island_options.verbose = false;
    islands{island} = SOP_agent_lshade_spacma_faithful(problem, seeds(island), island_options);
end

values = cellfun(@(item) item.record_value, islands);
[~, selected] = min(values);
result = islands{selected};
result.runtime = toc(t_start);
result.evaluation_count = sum(cellfun(@(item) item.evaluation_count, islands));
result.iteration = sum(cellfun(@(item) item.iteration, islands));
result.convergence_curve = vertcat(islands{1}.convergence_curve(:), ...
    islands{2}.convergence_curve(:), islands{3}.convergence_curve(:));
result.raw_convergence_curve = vertcat(islands{1}.raw_convergence_curve(:), ...
    islands{2}.raw_convergence_curve(:), islands{3}.raw_convergence_curve(:));
result.population_num = 3 * 18 * problem.dimension;
result.algorithm_combination = sprintf(['Three independent L-SHADE-SPACMA composition islands\n' ...
    'Balanced, DE-diversity, and covariance-dominant search profiles\n' ...
    'Whole-vector objective selector retained the %s island'], labels{selected});
result.combination_number = 4;
result.agent_id = 'EvaluateAgent';
result.problem = problem;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
