function result = SOP_agent_lshade_spacma_archive(problem, seed, options)
% Staged L-SHADE / CMA-ES / L-SHADE-CMA cooperative hybrid.
%
% Literature basis: L-SHADE success-history DE, CMA-ES covariance adaptation,
% and SPACMA-style cooperation between DE and covariance-based sampling.
% This implementation only uses public objective evaluations.
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
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP = get_option(options, 'population_num', max(120, 4 * D));
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
p_rate = get_option(options, 'p_rate', 0.10);
explore_fraction = get_option(options, 'explore_fraction', 0.55);
cma_fraction = get_option(options, 'cma_fraction', 0.18);
local_radius = get_option(options, 'local_radius', 0.006);
cma_sigma = get_option(options, 'cma_sigma', 0.004);

explore_fes = max(NP, floor(explore_fraction * max_fes));
explore_options = options;
explore_options.population_num = NP;
explore_options.max_fes = explore_fes;
explore_options.max_runtime_sec = max(1, explore_fraction * max_runtime_sec);
explore_options.p_rate = p_rate;
explore_options.verbose = false;
explore = SOP_agent_lshade(problem, double(seed), explore_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - explore.evaluation_count);
cma_fes = max(0, min(remaining_fes, floor(cma_fraction * max_fes)));
if cma_fes >= 1000 && remaining_time > 2
    cma_options = struct();
    cma_options.population_num = max(24, round(0.18 * NP));
    cma_options.max_fes = cma_fes;
    cma_options.max_runtime_sec = max(1, min(remaining_time, cma_fraction * max_runtime_sec));
    cma_options.initial_point = explore.best_position;
    cma_options.sigma0 = cma_sigma;
    cma_options.restart_sigma = 0.5 * cma_sigma;
    cma_options.restart_limit = 2;
    cma_options.verbose = false;
    cma = SOP_agent_cma_es(problem, double(seed) + 8191, cma_options);
else
    cma = empty_stage(problem, D);
end

best_so_far = choose_best(explore, cma);
remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - explore.evaluation_count - cma.evaluation_count);
if remaining_fes >= NP && remaining_time > 2
    exploit_options = options;
    exploit_options.population_num = max(80, round(0.70 * NP));
    exploit_options.max_fes = remaining_fes;
    exploit_options.max_runtime_sec = remaining_time;
    exploit_options.p_rate = p_rate;
    exploit_options.cma_rate = get_option(options, 'tail_cma_rate', 0.18);
    exploit_options.elite_rate = get_option(options, 'tail_elite_rate', 0.24);
    exploit_options.cma_interval = get_option(options, 'tail_cma_interval', 12);
    exploit_options.initial_population = make_seed_population(best_so_far, explore, cma, exploit_options.population_num, lb, ub, span, local_radius);
    exploit_options.verbose = false;
    exploit = SOP_agent_lshade_cma(problem, double(seed) + 16381, exploit_options);
else
    exploit = empty_stage(problem, D);
end

result = choose_best(choose_best(explore, cma), exploit);
result.runtime = toc(t_start);
result.evaluation_count = explore.evaluation_count + cma.evaluation_count + exploit.evaluation_count;
result.iteration = explore.iteration + cma.iteration + exploit.iteration;
result.convergence_curve = [explore.convergence_curve(:); cma.convergence_curve(:); exploit.convergence_curve(:)];
result.raw_convergence_curve = [explore.raw_convergence_curve(:); cma.raw_convergence_curve(:); exploit.raw_convergence_curve(:)];
result.population_num = NP;
result.algorithm_combination = sprintf('Differential Evolution (DE)\nL-SHADE success-history adaptation\nSPACMA-style CMA-ES covariance cooperation\nL-SHADE-CMA elite archive refinement');
result.combination_number = 5;
result.agent_id = 'Agent1';
result.problem = problem;
end

function population = make_seed_population(best_stage, explore, cma, NP, lb, ub, span, radius_rate)
D = numel(lb);
population = lb + rand(NP, D) .* span;
row = 1;
population(row, :) = best_stage.best_position;
row = row + 1;
if isfield(explore, 'final_population') && ~isempty(explore.final_population)
    take = min(size(explore.final_population, 1), max(0, floor(0.35 * NP)));
    if take > 0
        population(row:row + take - 1, :) = explore.final_population(1:take, :);
        row = row + take;
    end
end
if isfield(cma, 'best_position') && ~isempty(cma.best_position) && all(isfinite(cma.best_position))
    population(row, :) = cma.best_position;
    row = row + 1;
end
radius = radius_rate .* span;
while row <= NP
    if rand() < 0.25
        noise = tan(pi * (rand(1, D) - 0.5));
        noise = min(max(noise, -8), 8) .* radius;
    else
        noise = randn(1, D) .* radius;
    end
    population(row, :) = min(max(best_stage.best_position + noise, lb), ub);
    row = row + 1;
end
end

function stage = choose_best(a, b)
if isempty(b.record_value) || b.record_value < a.record_value
    stage = b;
else
    stage = a;
end
end

function result = empty_stage(problem, D)
result = struct();
result.best_value = inf;
result.record_value = inf;
result.best_position = nan(1, D);
result.convergence_curve = [];
result.raw_convergence_curve = [];
result.runtime = 0;
result.iteration = 0;
result.population_num = 0;
result.evaluation_count = 0;
result.algorithm_combination = '';
result.combination_number = 0;
result.agent_id = 'Agent1';
result.problem = problem;
result.final_population = zeros(0, D);
result.final_fitness = [];
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
