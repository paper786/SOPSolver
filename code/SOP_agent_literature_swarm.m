function result = SOP_agent_literature_swarm(problem, seed, options)
% Literature-inspired swarm/metaheuristic family for SOP screening.
%
% Supported methods include WSO, MPA, MVO, AVOA, RIME, AO, HHO, GWO, WOA,
% EO, SMA, HGS, and TLBO. The implementations
% are original MATLAB implementations of the public update ideas from the
% corresponding papers and use only objective feedback.
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
method = lower(string(get_option(options, 'method', 'wso')));
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP = get_option(options, 'population_num', 120);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_iter = get_option(options, 'max_iter', max(1, floor((max_fes - NP) / NP)));
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
initial_point = get_option(options, 'initial_point', []);
if ~isempty(initial_point)
    center = min(max(initial_point(:)', lb), ub);
    initial_radius = get_option(options, 'initial_radius', []);
    if ~isempty(initial_radius)
        radius = make_radius(initial_radius, span, D);
        population = repmat(center, NP, 1) + randn(NP, D) .* repmat(radius, NP, 1);
        if get_option(options, 'initial_cauchy', false)
            cauchy_noise = tan(pi * (rand(NP, D) - 0.5));
            cauchy_noise = min(max(cauchy_noise, -8), 8);
            cauchy_mask = rand(NP, D) < 0.30;
            center_matrix = repmat(center, NP, 1);
            radius_matrix = repmat(radius, NP, 1);
            population(cauchy_mask) = center_matrix(cauchy_mask) + cauchy_noise(cauchy_mask) .* radius_matrix(cauchy_mask);
        end
        population = min(max(population, lb), ub);
    end
    population(1, :) = center;
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
velocity = 0.02 .* (2 * rand(NP, D) - 1) .* span;
curve = zeros(max_iter, 1);
actual_iter = 0;
mvo_operator_probability = get_option(options, 'mvo_operator_probability', [0.68, 0.32]);
mvo_operator_probability = max(mvo_operator_probability(:)', 0);
if numel(mvo_operator_probability) ~= 2 || sum(mvo_operator_probability) <= 0
    mvo_operator_probability = [0.68, 0.32];
end
mvo_operator_probability = mvo_operator_probability ./ sum(mvo_operator_probability);
mvo_operator_quality = ones(1, 2);
mvo_operator_learning_rate = get_option(options, 'mvo_operator_learning_rate', 0.20);
mvo_operator_min_probability = get_option(options, 'mvo_operator_min_probability', 0.10);

for iter = 1:max_iter
    if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
        break;
    end
    actual_iter = iter;
    progress = iter / max_iter;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if size(velocity, 1) == NP
        velocity = velocity(order, :);
    end
    best_position = population(1, :);
    best_raw = fitness(1);

    switch method
        case "wso"
            [trial, velocity] = wso_step(population, fitness, velocity, best_position, lb, ub, span, progress);
        case "mpa"
            trial = mpa_step(population, best_position, lb, ub, span, progress);
        case "mvo"
            trial = mvo_step(population, fitness, best_position, lb, ub, progress);
        case "mvo_adaptive"
            [trial, mvo_operator_used] = mvo_adaptive_step(population, fitness, best_position, ...
                lb, ub, progress, mvo_operator_probability);
        case "avoa"
            trial = avoa_step(population, fitness, best_position, lb, ub, progress);
        case "rime"
            trial = rime_step(population, fitness, best_position, lb, ub, progress);
        case "ao"
            trial = ao_step(population, fitness, best_position, lb, ub, span, progress);
        case "hho"
            trial = hho_step(population, best_position, lb, ub, progress);
        case "gwo"
            trial = gwo_step(population, lb, ub, progress);
        case "woa"
            trial = woa_step(population, best_position, lb, ub, progress);
        case "eo"
            trial = eo_step(population, lb, ub, progress);
        case "sma"
            trial = sma_step(population, fitness, best_position, lb, ub, progress);
        case "hgs"
            trial = hgs_step(population, fitness, best_position, lb, ub, span, progress);
        case "tlbo"
            trial = tlbo_step(population, fitness, lb, ub);
        case "clpso"
            [trial, velocity] = clpso_step(population, fitness, velocity, best_position, lb, ub, span, progress);
        otherwise
            error('SOP_agent_literature_swarm:BadMethod', 'Unknown literature swarm method: %s.', method);
    end

    remaining = max_fes - eval_count;
    if remaining < NP
        trial = trial(1:remaining, :);
        if method == "mvo_adaptive"
            mvo_operator_used = mvo_operator_used(1:remaining);
        end
    end
    trial_fitness = SOP_cec_evaluate(trial, problem);
    eval_count = eval_count + numel(trial_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    rows = numel(trial_fitness);
    improved = trial_fitness <= fitness(1:rows);
    if method == "mvo_adaptive"
        for operator_idx = 1:2
            used_mask = mvo_operator_used(1:rows) == operator_idx;
            if any(used_mask)
                success_rate = sum(improved(used_mask)) / sum(used_mask);
                mvo_operator_quality(operator_idx) = ...
                    (1 - mvo_operator_learning_rate) * mvo_operator_quality(operator_idx) + ...
                    mvo_operator_learning_rate * max(success_rate, 1e-4);
            end
        end
        adaptive_mass = max(0, 1 - 2 * mvo_operator_min_probability);
        mvo_operator_probability = mvo_operator_min_probability + ...
            adaptive_mass .* mvo_operator_quality ./ sum(mvo_operator_quality);
    end
    population(improved, :) = trial(improved, :);
    fitness(improved) = trial_fitness(improved);
    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        best_raw = current_best;
        best_position = population(current_idx, :);
    end
    curve(iter) = best_raw;
end
curve = curve(1:actual_iter);

runtime = toc(t_start);
[final_fitness, final_order] = sort(fitness);
final_population = population(final_order, :);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = runtime;
result.iteration = actual_iter;
result.population_num = NP;
result.evaluation_count = eval_count;
result.algorithm_combination = method_label(method);
result.combination_number = 1;
result.agent_id = 'Agent2';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;

if verbose
    fprintf('%s finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        upper(method), problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function [trial, velocity] = wso_step(population, fitness, velocity, best, lb, ub, span, progress)
% White Shark Optimizer style: smell-driven movement to the best and random
% fish-school interaction with velocity memory.
[NP, D] = size(population);
trial = population;
wf = 0.75 - 0.45 * progress;
visual = 0.18 * (1 - progress) + 0.015;
for i = 1:NP
    r1 = random_index_except(NP, i);
    r2 = random_index_except(NP, r1);
    smell = (best - population(i, :)) .* (0.3 + rand(1, D));
    school = population(r1, :) - population(r2, :);
    pressure = exp(-rank_index(fitness, i) / max(1, NP));
    velocity(i, :) = wf .* velocity(i, :) + pressure .* rand(1, D) .* smell + 0.35 .* rand(1, D) .* school;
    if rand() < 0.45 + 0.35 * progress
        candidate = population(i, :) + velocity(i, :);
    else
        candidate = best + randn(1, D) .* visual .* span;
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = mpa_step(population, best, lb, ub, span, progress)
% Marine Predators Algorithm style three-phase Brownian/Levy foraging.
[NP, D] = size(population);
elite = repmat(best, NP, 1);
trial = population;
if progress < 1/3
    step = randn(NP, D) .* (elite - rand(NP, D) .* population);
    trial = population + 0.5 .* rand(NP, D) .* step;
elseif progress < 2/3
    half = floor(NP / 2);
    levy = levy_flight(NP, D);
    brown = randn(NP, D);
    trial(1:half, :) = elite(1:half, :) + 0.5 .* levy(1:half, :) .* (elite(1:half, :) - population(1:half, :));
    trial(half + 1:end, :) = population(half + 1:end, :) + 0.5 .* brown(half + 1:end, :) .* (elite(half + 1:end, :) - population(half + 1:end, :));
else
    levy = levy_flight(NP, D);
    trial = elite + 0.5 .* levy .* (elite - population);
end
if rand() < 0.2
    mask = rand(NP, D) < 0.12;
    random_points = lb + rand(NP, D) .* span;
    trial(mask) = random_points(mask);
end
trial = min(max(trial, lb), ub);
end

function trial = mvo_step(population, fitness, best, lb, ub, progress)
% Multi-Verse Optimizer style white-hole exchange and wormhole travel.
[NP, D] = size(population);
trial = population;
inv_fit = max(fitness) - fitness + eps;
prob = inv_fit ./ sum(inv_fit);
cum_prob = cumsum(prob);
wep = 0.2 + 0.8 * progress;
tdr = 1 - progress ^ (1 / 6);
for i = 1:NP
    candidate = population(i, :);
    for d = 1:D
        if rand() < prob(i)
            donor = find(cum_prob >= rand(), 1, 'first');
            candidate(d) = population(donor, d);
        end
        if rand() < wep
            if rand() < 0.5
                candidate(d) = best(d) + tdr * rand() * (ub(d) - lb(d));
            else
                candidate(d) = best(d) - tdr * rand() * (ub(d) - lb(d));
            end
        end
    end
    trial(i, :) = candidate;
end
trial = min(max(trial, lb), ub);
end

function [trial, operator_used] = mvo_adaptive_step(population, fitness, best, lb, ub, progress, operator_probability)
% Success-adaptive MVO: choose between the original universe exchange and a
% rank-difference current-to-pbest wormhole operator.
[NP, D] = size(population);
classic_trial = mvo_step(population, fitness, best, lb, ub, progress);
trial = classic_trial;
operator_used = zeros(NP, 1);
p_rate = 0.24 - 0.14 * progress;
p_count = max(2, min(NP, ceil(p_rate * NP)));
rank_pressure = 1.35 + 1.15 * progress;
wep = 0.12 + 0.48 * progress;
tdr = max(0.01, 1 - progress ^ (1 / 6));
span = ub - lb;

for i = 1:NP
    operator_used(i) = sample_discrete(operator_probability);
    if operator_used(i) == 1
        continue;
    end

    pbest_idx = rank_weighted_index(p_count, rank_pressure);
    r1 = random_index_except(NP, i);
    r2 = random_index_except(NP, i);
    while r2 == r1
        r2 = random_index_except(NP, i);
    end
    F = min(0.92, max(0.24, 0.42 + 0.22 * randn()));
    candidate = population(i, :) + ...
        F .* (population(pbest_idx, :) - population(i, :)) + ...
        F .* (population(r1, :) - population(r2, :));

    wormhole_mask = rand(1, D) < wep;
    if any(wormhole_mask)
        wormhole_scale = (0.04 + 0.18 * rand(1, sum(wormhole_mask))) .* tdr;
        candidate(wormhole_mask) = best(wormhole_mask) + ...
            randn(1, sum(wormhole_mask)) .* wormhole_scale .* span(wormhole_mask);
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = avoa_step(population, fitness, best, lb, ub, progress)
% African Vultures Optimization style exploration/exploitation around top
% vultures, using starvation pressure to choose behavior.
[NP, D] = size(population);
trial = population;
second = population(min(2, NP), :);
F = (2 * rand(NP, 1) - 1) .* (1 - progress) + randn(NP, 1) .* 0.08;
for i = 1:NP
    leader = best;
    if rand() < 0.35
        leader = second;
    end
    if abs(F(i)) >= 1
        rand_idx = randi(NP);
        candidate = population(rand_idx, :) - rand(1, D) .* abs(2 * rand(1, D) .* population(rand_idx, :) - population(i, :));
    else
        if rand() < 0.5
            candidate = leader - F(i) .* abs(leader - population(i, :));
        else
            spiral = abs(leader - population(i, :)) .* exp(rand()) .* cos(2 * pi * rand(1, D));
            candidate = leader + F(i) .* spiral;
        end
    end
    if rank_index(fitness, i) > 0.7 * NP && rand() < 0.25
        candidate = candidate + 0.05 * randn(1, D) .* (ub - lb);
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = rime_step(population, fitness, best, lb, ub, progress)
% RIME style soft-rime global drift and hard-rime puncture to the best.
[NP, D] = size(population);
trial = population;
soft = (1 - progress) * cos(pi * progress / 2);
norm_fit = (max(fitness) - fitness + eps) ./ (max(fitness) - min(fitness) + eps);
for i = 1:NP
    candidate = population(i, :);
    soft_mask = rand(1, D) < (0.25 + 0.45 * (1 - progress));
    candidate(soft_mask) = best(soft_mask) + soft .* randn(1, sum(soft_mask)) .* (ub(soft_mask) - lb(soft_mask));
    hard_mask = rand(1, D) < norm_fit(i);
    candidate(hard_mask) = best(hard_mask);
    if rand() < 0.1 * (1 - progress)
        candidate = lb + rand(1, D) .* (ub - lb);
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = ao_step(population, fitness, best, lb, ub, span, progress)
% Aquila Optimizer style expanded/narrowed exploration and exploitation.
[NP, D] = size(population);
trial = population;
mean_x = mean(population, 1);
G1 = 2 * rand() - 1;
G2 = 2 * (1 - progress);
alpha = 0.1;
delta = 0.1;
QF = progress ^ ((2 * rand() - 1) / (1 - progress + eps));
for i = 1:NP
    if progress < 2/3
        if rand() < 0.5
            candidate = best .* (1 - progress) + (mean_x - best .* rand(1, D));
        else
            theta = 2 * pi * rand();
            r = 0.1 + 0.9 * rand();
            spiral = r .* [cos(theta * (1:D)); sin(theta * (1:D))];
            spiral = mean(spiral, 1);
            random_x = population(randi(NP), :);
            candidate = best .* levy_flight(1, D) + random_x + spiral .* rand(1, D) .* span;
        end
    else
        if rand() < 0.5
            candidate = (best - mean_x) .* alpha - rand(1, D) + (lb + rand(1, D) .* span) .* delta;
        else
            candidate = QF .* best - G1 .* population(i, :) .* rand(1, D) - G2 .* levy_flight(1, D) .* span;
        end
    end
    if rank_index(fitness, i) > 0.65 * NP && rand() < 0.15
        candidate = lb + rand(1, D) .* span;
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = hho_step(population, best, lb, ub, progress)
% Harris Hawks Optimizer standalone phase switching.
[NP, D] = size(population);
trial = population;
center = mean(population, 1);
for i = 1:NP
    E = 2 * (1 - progress) * (2 * rand() - 1);
    q = rand();
    J = 2 * (1 - rand(1, D));
    if abs(E) >= 1
        random_hawk = population(randi(NP), :);
        if q < 0.5
            candidate = random_hawk - rand(1, D) .* abs(random_hawk - 2 * rand(1, D) .* population(i, :));
        else
            candidate = (best - center) - rand(1, D) .* (lb + rand(1, D) .* (ub - lb));
        end
    else
        if q >= 0.5 && abs(E) >= 0.5
            candidate = best - E .* abs(J .* best - population(i, :));
        elseif q >= 0.5
            candidate = best - E .* abs(best - population(i, :));
        else
            candidate = best - E .* abs(J .* best - population(i, :)) + randn(1, D) .* 0.01 .* (ub - lb) .* levy_flight(1, D);
        end
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = gwo_step(population, lb, ub, progress)
% Grey Wolf Optimizer standalone leadership hierarchy.
[NP, D] = size(population);
alpha = population(1, :);
beta = population(min(2, NP), :);
delta = population(min(3, NP), :);
a = 2 * (1 - progress);
trial = population;
for i = 1:NP
    A1 = 2 * a * rand(1, D) - a; C1 = 2 * rand(1, D);
    A2 = 2 * a * rand(1, D) - a; C2 = 2 * rand(1, D);
    A3 = 2 * a * rand(1, D) - a; C3 = 2 * rand(1, D);
    X1 = alpha - A1 .* abs(C1 .* alpha - population(i, :));
    X2 = beta - A2 .* abs(C2 .* beta - population(i, :));
    X3 = delta - A3 .* abs(C3 .* delta - population(i, :));
    trial(i, :) = min(max((X1 + X2 + X3) / 3, lb), ub);
end
end

function trial = woa_step(population, best, lb, ub, progress)
% Whale Optimization Algorithm standalone encircling and spiral update.
[NP, D] = size(population);
a = 2 * (1 - progress);
trial = population;
for i = 1:NP
    p = rand();
    A = 2 * a * rand(1, D) - a;
    C = 2 * rand(1, D);
    if p < 0.5
        if norm(A) / sqrt(D) < 1
            candidate = best - A .* abs(C .* best - population(i, :));
        else
            random_x = population(randi(NP), :);
            candidate = random_x - A .* abs(C .* random_x - population(i, :));
        end
    else
        ell = -1 + 2 * rand(1, D);
        candidate = abs(best - population(i, :)) .* exp(ell) .* cos(2 * pi .* ell) + best;
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = eo_step(population, lb, ub, progress)
% Equilibrium Optimizer style equilibrium-pool update.
[NP, D] = size(population);
pool_count = min(4, NP);
eq_pool = population(1:pool_count, :);
eq_pool(end + 1, :) = mean(eq_pool, 1);
t = (1 - progress) ^ (2 * progress + eps);
a1 = 2;
trial = population;
for i = 1:NP
    ceq = eq_pool(randi(size(eq_pool, 1)), :);
    lambda = rand(1, D);
    r = rand(1, D);
    F = a1 * sign(r - 0.5) .* (exp(-lambda .* t) - 1);
    gcp = 0.5 * rand(1, D) .* (rand(1, D) < 0.5);
    G = gcp .* (ceq - lambda .* population(i, :));
    candidate = ceq + (population(i, :) - ceq) .* F + (G ./ (lambda + eps)) .* (1 - F);
    if rand() < 0.08 * (1 - progress)
        candidate = lb + rand(1, D) .* (ub - lb);
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = sma_step(population, fitness, best, lb, ub, progress)
% Slime Mould Algorithm style adaptive oscillation weights.
[NP, D] = size(population);
best_fit = min(fitness);
worst_fit = max(fitness);
weights = ones(NP, D);
half = max(1, floor(NP / 2));
for i = 1:NP
    rel = (best_fit - fitness(i)) / (best_fit - worst_fit + eps) + 1;
    rel = max(rel, 1e-12);
    if i <= half
        weights(i, :) = 1 + rand(1, D) .* log10(rel);
    else
        weights(i, :) = 1 - rand(1, D) .* log10(rel);
    end
end
a = atanh(max(1e-6, 1 - progress));
vc = 1 - progress;
trial = population;
for i = 1:NP
    p = tanh(abs(fitness(i) - best_fit));
    r1 = randi(NP);
    r2 = randi(NP);
    if rand() < 0.03
        candidate = lb + rand(1, D) .* (ub - lb);
    elseif rand() < p
        vb = -a + 2 * a * rand(1, D);
        candidate = best + vb .* (weights(i, :) .* population(r1, :) - population(r2, :));
    else
        candidate = vc .* population(i, :);
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = hgs_step(population, fitness, best, lb, ub, span, progress)
% Hunger Games Search style hunger-weighted attraction/roaming.
[NP, D] = size(population);
best_fit = min(fitness);
worst_fit = max(fitness);
quality = (worst_fit - fitness + eps) ./ (worst_fit - best_fit + eps);
hunger = 1 + (1 - quality) .* (1 + 2 * rand(NP, 1));
shrink = 1 - progress;
trial = population;
for i = 1:NP
    r1 = randi(NP);
    r2 = randi(NP);
    if rand() < 0.5 + 0.35 * progress
        candidate = population(i, :) + randn(1, D) .* shrink .* (best - abs(population(i, :))) ./ hunger(i);
    else
        candidate = best + rand(1, D) .* (population(r1, :) - population(r2, :)) ./ hunger(i);
    end
    if rand() < 0.12 * shrink
        candidate = candidate + 0.06 * randn(1, D) .* span;
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function trial = tlbo_step(population, fitness, lb, ub)
% Teaching-Learning-Based Optimization teacher and learner phases.
[NP, D] = size(population);
teacher = population(1, :);
mean_x = mean(population, 1);
trial = population;
for i = 1:NP
    TF = randi(2);
    candidate = population(i, :) + rand(1, D) .* (teacher - TF .* mean_x);
    j = random_index_except(NP, i);
    if fitness(i) < fitness(j)
        candidate = candidate + rand(1, D) .* (population(i, :) - population(j, :));
    else
        candidate = candidate + rand(1, D) .* (population(j, :) - population(i, :));
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function [trial, velocity] = clpso_step(population, fitness, velocity, best, lb, ub, span, progress)
% Comprehensive Learning PSO style dimension-wise exemplar learning.
[NP, D] = size(population);
trial = population;
if size(velocity, 1) ~= NP
    velocity = 0.02 .* (2 * rand(NP, D) - 1) .* span;
end
[~, order] = sort(fitness);
elite_count = max(3, min(NP, round(0.35 * NP)));
elites = population(order(1:elite_count), :);
inertia = 0.78 - 0.42 * progress;
learning = 1.55 + 0.35 * (1 - progress);
global_pull = 0.08 + 0.18 * progress;
for i = 1:NP
    exemplar = zeros(1, D);
    if rand() < 0.22 + 0.35 * progress
        exemplar(:) = best;
    else
        for d = 1:D
            a = randi(elite_count);
            b = randi(elite_count);
            if fitness(order(a)) <= fitness(order(b))
                exemplar(d) = elites(a, d);
            else
                exemplar(d) = elites(b, d);
            end
        end
    end
    velocity(i, :) = inertia .* velocity(i, :) + ...
        learning .* rand(1, D) .* (exemplar - population(i, :)) + ...
        global_pull .* rand(1, D) .* (best - population(i, :));
    vmax = (0.18 - 0.10 * progress) .* span;
    velocity(i, :) = min(max(velocity(i, :), -vmax), vmax);
    candidate = population(i, :) + velocity(i, :);
    if rand() < 0.06 * (1 - progress)
        mask = rand(1, D) < max(0.04, 5 / D);
        if ~any(mask)
            mask(randi(D)) = true;
        end
        candidate(mask) = lb(mask) + rand(1, sum(mask)) .* (ub(mask) - lb(mask));
    end
    trial(i, :) = min(max(candidate, lb), ub);
end
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function rank = rank_index(fitness, i)
[~, order] = sort(fitness);
rank_positions = zeros(size(fitness));
rank_positions(order) = 1:numel(fitness);
rank = rank_positions(i);
end

function idx = rank_weighted_index(limit, pressure)
weights = (limit:-1:1) .^ pressure;
idx = sample_discrete(weights ./ sum(weights));
end

function idx = sample_discrete(probability)
cumulative = cumsum(probability(:) ./ sum(probability));
idx = find(cumulative >= rand(), 1, 'first');
if isempty(idx)
    idx = numel(probability);
end
end

function L = levy_flight(NP, D)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
    (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
u = randn(NP, D) * sigma;
v = randn(NP, D);
L = u ./ (abs(v) .^ (1 / beta) + eps);
L = min(max(L, -10), 10);
end

function label = method_label(method)
switch method
    case "wso"
        label = sprintf('White Shark Optimizer (WSO)\nSmell-guided velocity and fish-school movement');
    case "mpa"
        label = sprintf('Marine Predators Algorithm (MPA)\nBrownian/Levy predator-prey foraging');
    case "mvo"
        label = sprintf('Multi-Verse Optimizer (MVO)\nWhite-hole exchange and wormhole exploitation');
    case "mvo_adaptive"
        label = sprintf(['Success-Adaptive Multi-Verse Optimizer (SA-MVO)\n' ...
            'Adaptive classic universe exchange and rank-difference wormhole search']);
    case "avoa"
        label = sprintf('African Vultures Optimization Algorithm (AVOA)\nStarvation-driven vulture exploration/exploitation');
    case "rime"
        label = sprintf('RIME Optimization Algorithm\nSoft-rime drift and hard-rime puncture');
    case "ao"
        label = sprintf('Aquila Optimizer (AO)\nExpanded/narrowed Aquila exploration and exploitation');
    case "hho"
        label = sprintf('Harris Hawks Optimizer (HHO)\nEscape-energy exploration/exploitation');
    case "gwo"
        label = sprintf('Grey Wolf Optimizer (GWO)\nAlpha-beta-delta leadership hierarchy');
    case "woa"
        label = sprintf('Whale Optimization Algorithm (WOA)\nEncircling and spiral bubble-net search');
    case "eo"
        label = sprintf('Equilibrium Optimizer (EO)\nEquilibrium-pool concentration update');
    case "sma"
        label = sprintf('Slime Mould Algorithm (SMA)\nAdaptive oscillation-weighted foraging');
    case "hgs"
        label = sprintf('Hunger Games Search (HGS)\nHunger-weighted attraction and roaming');
    case "tlbo"
        label = sprintf('Teaching-Learning-Based Optimization (TLBO)\nTeacher and learner population phases');
    case "clpso"
        label = sprintf('Comprehensive Learning Particle Swarm Optimization (CLPSO)\nDimension-wise exemplar learning scout');
    otherwise
        label = char(method);
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end

function radius = make_radius(initial_radius, span, D)
radius = initial_radius;
if isscalar(radius)
    radius = radius .* span;
end
radius = radius(:)';
if numel(radius) ~= D
    radius = repmat(radius(1), 1, D);
end
radius = max(radius, 1e-12);
end
