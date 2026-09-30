function result = SOP_agent_abchs_bbo_seeded_jso(problem, seed, options)
% ABC/HS/BBO seeded population followed by jSO/L-SHADE-RSP exploitation.
if nargin < 2
    seed = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end
if ~isempty(seed)
    rng(double(seed), 'twister');
else
    rng('shuffle');
end

t_start = tic;
D = problem.dimension;
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
NP = get_option(options, 'population_num', max(180, 10 * D));
scout_fraction = get_option(options, 'seed_scout_fraction', 0.12);
scout_fes = max(NP, floor(scout_fraction * max_fes));
scout_options = options;
scout_options.max_fes = scout_fes;
scout_options.population_num = get_option(options, 'seed_scout_population_num', max(80, round(0.34 * NP)));
scout_options.max_runtime_sec = max(1, min(max_runtime_sec, scout_fraction * max_runtime_sec));
scout = abchs_bbo_scout(problem, double(seed), scout_options);

remaining_time = max(1, max_runtime_sec - toc(t_start));
jso_options = options;
jso_options.max_fes = max(1000, max_fes - scout.evaluation_count);
jso_options.max_runtime_sec = remaining_time;
jso_options.population_num = NP;
jso_options.initial_population = seeded_population_from_scout(scout, NP, problem.lb, problem.ub, options);
jso_options.verbose = false;
jso_options.include_center = get_option(options, 'include_center', true);
jso_options.ranked_r1 = get_option(options, 'ranked_r1', true);
jso_options.rank_pressure = get_option(options, 'rank_pressure', 1.7);
jso_options.mu_F_init = get_option(options, 'mu_F_init', 0.34);
jso_options.mu_CR_init = get_option(options, 'mu_CR_init', 0.86);
jso_options.p_rate_start = get_option(options, 'p_rate_start', 0.20);
jso_options.p_rate_end = get_option(options, 'p_rate_end', 0.040);
jso_options.archive_factor_start = get_option(options, 'archive_factor_start', 1.8);
jso_options.archive_factor_end = get_option(options, 'archive_factor_end', 3.2);
jso = SOP_agent_lshade_jso(problem, double(seed) + 7919, jso_options);

if jso.record_value < scout.record_value
    result = jso;
else
    result = scout;
end
result.runtime = toc(t_start);
result.evaluation_count = scout.evaluation_count + jso.evaluation_count;
result.iteration = scout.iteration + jso.iteration;
result.convergence_curve = [scout.convergence_curve(:); jso.convergence_curve(:)];
result.raw_convergence_curve = [scout.raw_convergence_curve(:); jso.raw_convergence_curve(:)];
result.algorithm_combination = sprintf('Artificial Bee Colony (ABC), Harmony Search (HS), and BBO seeded scout\njSO/L-SHADE ranked-r1 exploitation');
result.combination_number = 5;
result.agent_id = 'Agent2';
end

function scout = abchs_bbo_scout(problem, seed, options)
rng(double(seed), 'twister');
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP = get_option(options, 'population_num', 120);
max_fes = get_option(options, 'max_fes', 100000);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
t_start = tic;
population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', true)
    population(1, :) = 0.5 .* (lb + ub);
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, idx] = min(fitness);
best_x = population(idx, :);
curve = zeros(max(1, ceil(max_fes / NP)), 1);
iter = 0;
while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    best_x = population(1, :);
    best_raw = fitness(1);
    count = min(NP, max_fes - eval_count);
    trial = population(1:count, :);
    inv_fit = max(fitness) - fitness + eps;
    habitat_prob = inv_fit ./ sum(inv_fit);
    cum_prob = cumsum(habitat_prob);
    progress = eval_count / max(1, max_fes);
    for i = 1:count
        op = mod(i + iter, 3);
        x = population(i, :);
        if op == 0
            neighbor = population(random_index_except(NP, i), :);
            elite = population(randi(max(2, round(0.20 * NP))), :);
            phi = -1 + 2 * rand(1, D);
            mask = rand(1, D) < get_option(options, 'abc_block_rate', max(0.05, 6 / D));
            if ~any(mask)
                mask(randi(D)) = true;
            end
            x(mask) = x(mask) + get_option(options, 'abc_neighbor_scale', 0.55) .* phi(mask) .* (x(mask) - neighbor(mask)) + ...
                get_option(options, 'abc_best_pull', 0.12) .* rand(1, nnz(mask)) .* (best_x(mask) - elite(mask));
        elseif op == 1
            hmcr = get_option(options, 'hs_memory_rate', 0.90);
            par = get_option(options, 'hs_pitch_rate', 0.22) * (1 - 0.45 * progress);
            bw = get_option(options, 'hs_bandwidth', 0.010) .* (1 - 0.70 * progress) .* span;
            for d = 1:D
                if rand() < hmcr
                    donor = randi(NP);
                    x(d) = population(donor, d);
                    if rand() < par
                        x(d) = x(d) + randn() .* bw(d);
                    end
                elseif rand() < 0.18
                    x(d) = lb(d) + rand() .* span(d);
                end
            end
            x = 0.72 .* x + 0.28 .* best_x;
        else
            immigration = 0.18 + 0.64 * (i - 1) / max(1, NP - 1);
            mutation = 0.04 + 0.18 * (i - 1) / max(1, NP - 1);
            for d = 1:D
                if rand() < immigration
                    donor = find(cum_prob >= rand(), 1, 'first');
                    x(d) = population(donor, d);
                end
                if rand() < mutation
                    x(d) = x(d) + randn() .* get_option(options, 'bbo_mutation_scale', 0.006) .* span(d);
                end
            end
            x = x + get_option(options, 'bbo_best_pull', 0.10) .* rand(1, D) .* (best_x - x);
        end
        trial(i, :) = min(max(x, lb), ub);
    end
    values = SOP_cec_evaluate(trial, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    improved = values(:) <= fitness(1:count);
    population(improved, :) = trial(improved, :);
    fitness(improved) = values(improved);
    [current_best, idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_x = population(idx, :);
    end
    curve(iter) = best_raw;
end
curve = curve(1:iter);
[final_fitness, order] = sort(fitness);
scout = struct();
scout.best_value = best_raw;
scout.record_value = SOP_cec_record_value(best_raw, problem);
scout.best_position = best_x;
scout.convergence_curve = SOP_cec_record_value(curve, problem);
scout.raw_convergence_curve = curve;
scout.runtime = toc(t_start);
scout.iteration = iter;
scout.population_num = NP;
scout.evaluation_count = eval_count;
scout.final_population = population(order, :);
scout.final_fitness = final_fitness;
end

function population = seeded_population_from_scout(scout, NP, lb, ub, options)
elite = scout.final_population;
fitness = scout.final_fitness;
[fitness, order] = sort(fitness);
elite = elite(order, :);
D = size(elite, 2);
span = ub - lb;
elite_keep = min(size(elite, 1), max(6, round(get_option(options, 'seed_elite_keep_rate', 0.30) * NP)));
population = zeros(NP, D);
population(1, :) = scout.best_position;
population(2:elite_keep + 1, :) = elite(1:elite_keep, :);
filled = elite_keep + 1;
candidate_limit = min(size(elite, 1), max(elite_keep, round(get_option(options, 'seed_diversity_pool_rate', 1.8) * NP)));
while filled < NP
    filled = filled + 1;
    if rand() < get_option(options, 'seed_diversity_rate', 0.55) && candidate_limit > elite_keep
        selected = population(1:filled - 1, :);
        distances = zeros(candidate_limit, 1);
        for k = 1:candidate_limit
            distances(k) = min(sqrt(mean(((selected - elite(k, :)) ./ max(span, eps)) .^ 2, 2)));
        end
        distances(1:elite_keep) = -inf;
        [~, idx] = max(distances);
        child = elite(idx, :);
    else
        a = randi(elite_keep);
        b = randi(elite_keep);
        child = elite(a, :);
        mask = rand(1, D) < get_option(options, 'seed_mix_rate', 0.34);
        if ~any(mask)
            mask(randi(D)) = true;
        end
        child(mask) = elite(b, mask);
        child = child + randn(1, D) .* (get_option(options, 'seed_jitter', 0.0022) .* span);
    end
    population(filled, :) = min(max(child, lb), ub);
end
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
