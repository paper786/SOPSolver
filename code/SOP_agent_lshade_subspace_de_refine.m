function result = SOP_agent_lshade_subspace_de_refine(problem, seed, options)
% L-SHADE/L-SHADE-CMA followed by local subspace Differential Evolution.
%
% The base stage finds a promising basin. The refinement stage keeps a
% compact local population around the best point and applies DE-style
% mutation/crossover only on random coordinate subsets, which is a
% cooperative-coevolution style metaheuristic for high-dimensional cases.
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
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
base_fraction = get_option(options, 'base_fraction', 0.72);
base_algorithm = lower(string(get_option(options, 'base_algorithm', 'lshade')));

base_options = options;
base_options.max_runtime_sec = base_fraction * max_runtime_sec;
base_options.max_fes = max(1000, floor(base_fraction * max_fes));
base_options.verbose = false;
switch base_algorithm
    case "lshade_cma"
        base = SOP_agent_lshade_cma(problem, double(seed), base_options);
        base_label = 'L-SHADE with elite covariance sampling';
    case "lshade_jso"
        base = SOP_agent_lshade_jso(problem, double(seed), base_options);
        base_label = 'jSO/L-SHADE success-history adaptation';
    otherwise
        base = SOP_agent_lshade(problem, double(seed), base_options);
        base_label = 'L-SHADE success-history adaptation';
end

D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
best_x = base.best_position;
best_raw = base.best_value;
eval_count = base.evaluation_count;
curve = base.raw_convergence_curve(:);
local_np = get_option(options, 'local_population_num', max(32, round(0.35 * get_option(options, 'population_num', 180))));
radius = get_option(options, 'local_radius', 0.010) .* span;
min_radius = get_option(options, 'min_radius', 1e-7) .* span;
F0 = get_option(options, 'local_F', 0.55);
CR0 = get_option(options, 'local_CR', 0.35);
block_rate = get_option(options, 'block_rate', min(0.25, max(0.06, 10 / D)));
block_min = get_option(options, 'block_min', 4);
stall = 0;
refine_iter = 0;
subspace_bandit = get_option(options, 'subspace_bandit', false);
bandit_values = zeros(1, 4);
bandit_counts = ones(1, 4);
anchor_population = [];
if isfield(base, 'final_population') && ~isempty(base.final_population)
    anchor_population = base.final_population;
end

population = repmat(best_x, local_np, 1) + randn(local_np, D) .* repmat(radius, local_np, 1);
population = min(max(population, lb), ub);
population(1, :) = best_x;
fitness = SOP_cec_evaluate(population, problem);
eval_count = eval_count + numel(fitness);
[trial_best, best_idx] = min(fitness);
if trial_best < best_raw
    best_raw = trial_best;
    best_x = population(best_idx, :);
end

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    refine_iter = refine_iter + 1;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if fitness(1) < best_raw
        best_raw = fitness(1);
        best_x = population(1, :);
    end
    trial_pop = population;
    selected_ops = ones(local_np, 1);
    for i = 1:local_np
        if subspace_bandit
            op = select_bandit_operator(bandit_values, bandit_counts, get_option(options, 'bandit_epsilon', 0.18));
            selected_ops(i) = op;
            subset = bandit_subset(op, population, anchor_population, D, block_rate, block_min, options);
        else
            subset = rand(1, D) < block_rate;
            if sum(subset) < block_min
                cols = randperm(D, min(D, block_min));
                subset(cols) = true;
            end
        end
        r1 = random_index_except(local_np, i);
        r2 = random_index_except(local_np, r1);
        F = min(0.95, max(0.12, F0 + 0.12 * randn()));
        CR = min(1, max(0.02, CR0 + 0.15 * randn()));
        direction = population(r1, :) - population(r2, :);
        if subspace_bandit && selected_ops(i) == 2
            elite_count = max(2, min(local_np, round(get_option(options, 'bandit_elite_rate', 0.24) * local_np)));
            a = randi(elite_count);
            b = randi(elite_count);
            while b == a && elite_count > 1
                b = randi(elite_count);
            end
            direction = population(a, :) - population(b, :);
        elseif subspace_bandit && selected_ops(i) == 3 && size(anchor_population, 1) >= 2
            a = randi(size(anchor_population, 1));
            b = randi(size(anchor_population, 1));
            while b == a && size(anchor_population, 1) > 1
                b = randi(size(anchor_population, 1));
            end
            direction = anchor_population(a, :) - anchor_population(b, :);
        end
        mutant = population(i, :);
        mutant(subset) = population(i, subset) ...
            + F .* (best_x(subset) - population(i, subset)) ...
            + F .* direction(subset);
        mask = (rand(1, D) < CR) & subset;
        if ~any(mask)
            subset_idx = find(subset);
            mask(subset_idx(randi(numel(subset_idx)))) = true;
        end
        candidate = population(i, :);
        candidate(mask) = mutant(mask);
        if rand() < 0.18
            candidate(subset) = candidate(subset) + randn(1, sum(subset)) .* radius(subset);
        end
        trial_pop(i, :) = min(max(candidate, lb), ub);
    end

    remaining = max_fes - eval_count;
    if remaining < local_np
        trial_pop = trial_pop(1:remaining, :);
        selected_ops = selected_ops(1:remaining);
    end
    values = SOP_cec_evaluate(trial_pop, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    rows = numel(values);
    improved = values <= fitness(1:rows);
    if subspace_bandit
        deltas = max(0, fitness(1:rows) - values(:));
        for k = 1:rows
            op = selected_ops(k);
            bandit_counts(op) = bandit_counts(op) + 1;
            reward = double(improved(k)) + log1p(max(0, deltas(k)));
            bandit_values(op) = bandit_values(op) + reward;
        end
    end
    population(improved, :) = trial_pop(improved, :);
    fitness(improved) = values(improved);
    [iter_best, iter_idx] = min(fitness);
    if iter_best < best_raw
        best_raw = iter_best;
        best_x = population(iter_idx, :);
        radius = max(0.985 .* radius, min_radius);
        stall = 0;
    else
        radius = max(0.94 .* radius, min_radius);
        stall = stall + 1;
    end
    if stall >= 40
        radius = max(radius, get_option(options, 'reset_radius', 0.006) .* span .* (0.7 + 0.6 * rand(1, D)));
        stall = 0;
    end
    if mod(refine_iter, 10) == 0
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
    end
end

result = base;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.raw_convergence_curve = curve;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = toc(t_start);
result.iteration = base.iteration + refine_iter;
result.evaluation_count = eval_count;
if subspace_bandit
    refine_label = 'Local bandit-selected random/elite/archive/block subspace DE refinement';
else
    refine_label = 'Local random-subspace DE refinement';
end
result.algorithm_combination = sprintf('Differential Evolution (DE)\n%s\n%s', base_label, refine_label);
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function op = select_bandit_operator(values, counts, epsilon)
if rand() < epsilon
    op = randi(numel(values));
    return;
end
means = values ./ max(1, counts);
bonus = sqrt(log(sum(counts) + 1) ./ max(1, counts));
[~, op] = max(means + 0.25 .* bonus + 1e-9 .* rand(size(values)));
end

function subset = bandit_subset(op, population, anchor_population, D, block_rate, block_min, options)
subset = false(1, D);
count = max(block_min, round(block_rate * D));
count = min(D, max(1, count));
if op == 2
    elite_count = max(2, min(size(population, 1), round(get_option(options, 'bandit_elite_rate', 0.24) * size(population, 1))));
    a = randi(elite_count);
    b = randi(elite_count);
    while b == a && elite_count > 1
        b = randi(elite_count);
    end
    score = abs(population(a, :) - population(b, :)) + 1e-12 .* rand(1, D);
    [~, order] = sort(score, 'descend');
    subset(order(1:count)) = true;
elseif op == 3 && size(anchor_population, 1) >= 2
    a = randi(size(anchor_population, 1));
    b = randi(size(anchor_population, 1));
    while b == a && size(anchor_population, 1) > 1
        b = randi(size(anchor_population, 1));
    end
    score = abs(anchor_population(a, :) - anchor_population(b, :)) + 1e-12 .* rand(1, D);
    [~, order] = sort(score, 'descend');
    subset(order(1:count)) = true;
elseif op == 4
    block_len = max(block_min, min(D, round(get_option(options, 'bandit_block_len_rate', 0.075) * D)));
    start_dim = randi(D);
    dims = mod((start_dim - 1):(start_dim + block_len - 2), D) + 1;
    subset(dims) = true;
else
    subset = rand(1, D) < block_rate;
end
if sum(subset) < block_min
    cols = randperm(D, min(D, block_min));
    subset(cols) = true;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
