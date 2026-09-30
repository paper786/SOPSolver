function result = SOP_agent_lshade_cc_refine(problem, seed, options)
% L-SHADE-CMA local restart followed by block cooperative refinement.
%
% The first phase uses the retained DE family (L-SHADE/L-SHADE-CMA) to find
% a basin. The second phase refines only small coordinate groups at a time
% using DE-style elite differences plus stochastic local perturbations.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if isempty(seed)
    seed = randi(1000000);
end
rng(double(seed), 'twister');

t_start = tic;
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);

base_options = options;
base_options.max_runtime_sec = min(max_runtime_sec, get_option(options, 'pre_runtime_sec', 0.76 * max_runtime_sec));
base_options.max_fes = min(max_fes, get_option(options, 'pre_max_fes', max_fes));
base_options.base_algorithm = get_option(options, 'base_algorithm', 'lshade_cma');
base_options.local_algorithm = get_option(options, 'local_algorithm', 'lshade_cma');
base_options.base_max_fes = get_option(options, 'base_max_fes', 1000000);
base_options.local_radius = get_option(options, 'local_radius', 0.006);
base_options.local_population_num = get_option(options, 'local_population_num', max(80, round(0.55 * get_option(options, 'population_num', 180))));
base_options.use_base_elites = get_option(options, 'use_base_elites', true);
base_options.pattern_after = false;
base_options.verbose = false;
base = SOP_agent_lshade_local_restart(problem, double(seed), base_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
remaining_fes = max(0, max_fes - base.evaluation_count);
cc = cc_refine(problem, base, remaining_fes, remaining_time, options);

result = base;
if cc.record_value < result.record_value
    result.best_value = cc.best_value;
    result.record_value = cc.record_value;
    result.best_position = cc.best_position;
end
result.runtime = toc(t_start);
result.evaluation_count = base.evaluation_count + cc.evaluation_count;
result.iteration = base.iteration + cc.iteration;
result.raw_convergence_curve = [base.raw_convergence_curve(:); cc.raw_convergence_curve(:)];
result.convergence_curve = SOP_cec_record_value(result.raw_convergence_curve, problem);
result.algorithm_combination = sprintf('Differential Evolution (DE)\nL-SHADE-CMA local restart\nBlock cooperative DE/stochastic refinement');
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function cc = cc_refine(problem, base, max_fes, max_runtime_sec, options)
t_start = tic;
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
best_x = base.best_position;
best_raw = base.best_value;
eval_count = 0;
iter = 0;
curve = zeros(max(1, ceil(max_fes / max(1, get_option(options, 'cc_batch', 80)))), 1);

if max_fes <= 0 || max_runtime_sec <= 0
    cc = pack_cc(best_raw, best_x, curve([]), eval_count, iter, problem);
    return;
end

group_size = get_option(options, 'cc_group_size', 8);
group_sizes = get_option(options, 'cc_group_sizes', group_size);
batch = get_option(options, 'cc_batch', 80);
partitions = get_option(options, 'cc_partitions', 3);
sigma = get_option(options, 'cc_sigma', 0.0018) .* span;
min_sigma = get_option(options, 'cc_min_sigma', 2e-7) .* max(1, span);
reset_sigma = get_option(options, 'cc_reset_sigma', 0.0035) .* span;
elite_count = get_option(options, 'cc_elite_count', 24);
stall_limit = get_option(options, 'cc_stall_limit', 18);

elite_pool = [];
if isfield(base, 'final_population') && ~isempty(base.final_population)
    elite_pool = base.final_population(1:min(elite_count, size(base.final_population, 1)), :);
end
if isempty(elite_pool)
    elite_pool = repmat(best_x, max(4, elite_count), 1);
end
if size(elite_pool, 1) < 4
    elite_pool = [elite_pool; repmat(best_x, 4 - size(elite_pool, 1), 1)];
end

groups = build_groups(D, group_sizes, partitions);
stall = zeros(numel(groups), 1);
g = 1;
while eval_count < max_fes && toc(t_start) < max_runtime_sec
    group = groups{g};
    count = min(batch, max_fes - eval_count);
    candidates = repmat(best_x, count, 1);
    for i = 1:count
        candidates(i, group) = make_block_candidate(best_x, elite_pool, group, sigma, lb, ub);
    end
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        elite_pool = update_elites(elite_pool, best_x);
        sigma(group) = max(0.992 .* sigma(group), min_sigma(group));
        stall(g) = 0;
    else
        sigma(group) = max(0.93 .* sigma(group), min_sigma(group));
        stall(g) = stall(g) + 1;
        if stall(g) >= stall_limit
            sigma(group) = max(sigma(group), reset_sigma(group) .* (0.65 + 0.70 * rand(1, numel(group))));
            stall(g) = 0;
        end
    end
    iter = iter + 1;
    if iter > numel(curve)
        curve(end + 256, 1) = 0; %#ok<AGROW>
    end
    curve(iter) = best_raw;
    g = g + 1;
    if g > numel(groups)
        groups = build_groups(D, group_sizes, partitions);
        g = 1;
        sigma = max(0.985 .* sigma, min_sigma);
    end
end
curve = curve(1:iter);
cc = pack_cc(best_raw, best_x, curve, eval_count, iter, problem);
end

function candidate_block = make_block_candidate(best_x, elites, group, sigma, lb, ub)
block_len = numel(group);
mode = rand();
if mode < 0.45
    noise = randn(1, block_len);
    if rand() < 0.35
        cauchy_noise = tan(pi * (rand(1, block_len) - 0.5));
        noise = min(max(cauchy_noise, -8), 8);
    end
    candidate_block = best_x(group) + noise .* sigma(group);
elseif mode < 0.78
    n = size(elites, 1);
    a = randi(n);
    b = randi(n);
    while b == a
        b = randi(n);
    end
    F = min(0.90, max(0.15, 0.42 + 0.18 * randn()));
    candidate_block = best_x(group) + F .* (elites(a, group) - elites(b, group));
    if rand() < 0.45
        candidate_block = candidate_block + randn(1, block_len) .* (0.35 .* sigma(group));
    end
else
    candidate_block = best_x(group);
    touch = false(1, block_len);
    touch(randperm(block_len, max(1, min(block_len, ceil(0.35 * block_len))))) = true;
    candidate_block(touch) = candidate_block(touch) + sign(randn(1, sum(touch))) .* sigma(group(touch)) .* (0.4 + rand(1, sum(touch)));
end
candidate_block = min(max(candidate_block, lb(group)), ub(group));
end

function groups = build_groups(D, group_sizes, partitions)
group_sizes = unique(max(1, round(group_sizes(:)')));
groups = {};
for p = 1:partitions
    for s = 1:numel(group_sizes)
        group_size = group_sizes(s);
        order = randperm(D);
        for start_idx = 1:group_size:D
            groups{end + 1, 1} = order(start_idx:min(D, start_idx + group_size - 1)); %#ok<AGROW>
        end
    end
end
groups = groups(randperm(numel(groups)));
end

function elites = update_elites(elites, best_x)
elites = [best_x; elites]; %#ok<AGROW>
if size(elites, 1) > 48
    elites = elites(1:48, :);
end
end

function cc = pack_cc(best_raw, best_x, curve, eval_count, iter, problem)
cc = struct();
cc.best_value = best_raw;
cc.record_value = SOP_cec_record_value(best_raw, problem);
cc.best_position = best_x;
cc.raw_convergence_curve = curve(:);
cc.convergence_curve = SOP_cec_record_value(curve(:), problem);
cc.evaluation_count = eval_count;
cc.iteration = iter;
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
