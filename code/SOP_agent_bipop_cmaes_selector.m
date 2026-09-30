function result = SOP_agent_bipop_cmaes_selector(problem, seed, options)
% Small/medium/large population CMA-ES whole-basin selector.
if nargin < 2 || isempty(seed)
    seed = randi(1000000);
end
if nargin < 3 || isempty(options)
    options = struct();
end

t_start = tic;
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', 295);
island_fes = floor(max_fes / 3);
populations = [32, 64, 128];
sigma_rates = [0.12, 0.28, 0.48];
seeds = double(seed) + [0, 3571, 7919];
labels = {'small-population', 'medium-population', 'large-population'};
islands = cell(1, 3);

for island = 1:3
    remaining_time = max(1, max_runtime_sec - toc(t_start));
    cma_options = struct();
    cma_options.population_num = populations(island);
    cma_options.max_fes = island_fes;
    cma_options.max_runtime_sec = min(92, remaining_time);
    cma_options.sigma0 = sigma_rates(island);
    cma_options.restart_sigma = 0.10 * sigma_rates(island);
    cma_options.restart_limit = 2;
    cma_options.eig_interval = 8;
    cma_options.verbose = false;
    islands{island} = SOP_agent_cma_es(problem, seeds(island), cma_options);
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
result.population_num = sum(populations);
result.algorithm_combination = sprintf(['BIPOP-style independent CMA-ES basin search\n' ...
    'Small, medium, and large population/step-size regimes\n' ...
    'Whole-vector objective selector retained the %s regime'], labels{selected});
result.combination_number = 1;
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
