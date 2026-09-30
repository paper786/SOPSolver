function result = SOP_agent_gsk(problem, seed, options)
% Gaining-Sharing Knowledge inspired optimizer.
%
% Literature basis: Gaining-sharing knowledge based algorithm (GSK). The
% implementation uses junior and senior knowledge sharing phases with
% greedy selection and bound repair.
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
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
NP = get_option(options, 'population_num', 160);
max_iter = get_option(options, 'max_iter', 4000);
KR = get_option(options, 'knowledge_rate', 0.9);
KF0 = get_option(options, 'knowledge_factor', 0.55);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
verbose = get_option(options, 'verbose', false);
top_rate = get_option(options, 'top_rate', 0.18);
mid_rate = get_option(options, 'mid_rate', 0.45);
bottom_rate = get_option(options, 'bottom_rate', 0.75);
junior_power = get_option(options, 'junior_power', 1.0);
juniors_min_probability = get_option(options, 'junior_min_probability', 0.0);
junior_rand_weight = get_option(options, 'junior_rand_weight', 0.5);
senior_best_weight = get_option(options, 'senior_best_weight', 1.0);
senior_top_weight = get_option(options, 'senior_top_weight', 1.0);
senior_mid_weight = get_option(options, 'senior_mid_weight', 0.25);
kf_decay = get_option(options, 'kf_decay', 0.35);
kf_noise = get_option(options, 'kf_noise', 0.08);
kf_min = get_option(options, 'kf_min', 0.15);
kf_max = get_option(options, 'kf_max', 0.95);
rime_puncture_rate = get_option(options, 'rime_puncture_rate', 0);
rime_start_progress = get_option(options, 'rime_start_progress', 0.45);
rime_block_rate = get_option(options, 'rime_block_rate', 0.05);
rime_radius = get_option(options, 'rime_radius', 0.0025);
rime_hard_probability = get_option(options, 'rime_hard_probability', 0.55);
rime_success_gate = get_option(options, 'rime_success_gate', false);
adaptive_rime_rate = rime_puncture_rate;
rime_rate_min = get_option(options, 'rime_rate_min', 0.004);
rime_rate_max = get_option(options, 'rime_rate_max', max(rime_puncture_rate, 0.08));
linewell_rate = get_option(options, 'linewell_rate', 0);
linewell_start_progress = get_option(options, 'linewell_start_progress', 0.48);
linewell_block_rate = get_option(options, 'linewell_block_rate', max(0.035, 5 / D));
linewell_step_scales = get_option(options, 'linewell_step_scales', [0.0015, 0.0030, 0.0055]);
linewell_blend = get_option(options, 'linewell_blend', 0.55);
adaptive_knowledge_sources = get_option(options, 'adaptive_knowledge_sources', false);
source_probability = get_option(options, 'source_probability_init', [0.42, 0.42, 0.16]);
source_probability = source_probability(:)' ./ sum(source_probability);
source_learning_rate = get_option(options, 'source_learning_rate', 0.18);
source_min_probability = get_option(options, 'source_min_probability', 0.08);
cross_rank_weight = get_option(options, 'cross_rank_weight', 0.62);
cross_archive = zeros(0, D);
cross_archive_factor = get_option(options, 'cross_archive_factor', 1.2);
eig_success_gate = get_option(options, 'eig_success_gate', false);
eig_gate_rate = get_option(options, 'eig_gate_rate', 0.045);
eig_rate_min = get_option(options, 'eig_rate_min', 0.004);
eig_rate_max = get_option(options, 'eig_rate_max', max(0.07, eig_gate_rate));
eig_interval = get_option(options, 'eig_interval', 16);
eig_elite_rate = get_option(options, 'eig_elite_rate', 0.20);
eig_basis = eye(D);
exemplar_success_gate = get_option(options, 'exemplar_success_gate', false);
exemplar_rate = get_option(options, 'exemplar_rate', 0.022);
exemplar_rate_min = get_option(options, 'exemplar_rate_min', 0.002);
exemplar_rate_max = get_option(options, 'exemplar_rate_max', max(0.06, exemplar_rate));
exemplar_start_progress = get_option(options, 'exemplar_start_progress', 0.42);
exemplar_block_rate = get_option(options, 'exemplar_block_rate', max(0.04, 5 / D));
exemplar_elite_rate = get_option(options, 'exemplar_elite_rate', 0.28);
exemplar_blend = get_option(options, 'exemplar_blend', 0.34);
evolution_path_enabled = get_option(options, 'evolution_path_enabled', false);
evolution_path_rate = get_option(options, 'evolution_path_rate', 0.028);
evolution_path_rate_min = get_option(options, 'evolution_path_rate_min', 0.002);
evolution_path_rate_max = get_option(options, 'evolution_path_rate_max', max(0.07, evolution_path_rate));
evolution_path_start_progress = get_option(options, 'evolution_path_start_progress', 0.40);
evolution_path_learning_rate = get_option(options, 'evolution_path_learning_rate', 0.18);
evolution_path_step_scales = get_option(options, 'evolution_path_step_scales', [0.5, 1.0, 2.0, 4.0]);
evolution_path_blend = get_option(options, 'evolution_path_blend', 0.72);
evolution_path = zeros(1, D);
evolution_path_scale = 0;

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
        center_matrix = repmat(center, NP, 1);
        radius_matrix = repmat(radius, NP, 1);
        population = center_matrix + randn(NP, D) .* radius_matrix;
        if get_option(options, 'initial_cauchy', false)
            cauchy_noise = tan(pi * (rand(NP, D) - 0.5));
            cauchy_noise = min(max(cauchy_noise, -8), 8);
            cauchy_mask = rand(NP, D) < 0.30;
            population(cauchy_mask) = center_matrix(cauchy_mask) + cauchy_noise(cauchy_mask) .* radius_matrix(cauchy_mask);
        end
        population = min(max(population, lb), ub);
    end
    population(1, :) = center;
end
eig_center = mean(population, 1);
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
curve = zeros(max_iter, 1);
actual_iter = 0;

for iter = 1:max_iter
    if toc(t_start) >= max_runtime_sec
        break;
    end
    actual_iter = iter;
    progress = iter / max_iter;
    [fitness, order] = sort(fitness);
    population = population(order, :);
    if eig_success_gate && (iter == 1 || mod(iter, eig_interval) == 0)
        [eig_basis, eig_center] = elite_eigen_basis(population, span, eig_elite_rate);
    end
    top_count = max(2, round(top_rate * NP));
    mid_start = max(top_count + 1, round(mid_rate * NP));
    bottom_start = max(mid_start + 1, round(bottom_rate * NP));
    KF = KF0 * (1 - kf_decay * progress) + kf_noise * randn(NP, 1);
    KF = min(kf_max, max(kf_min, KF));
    trial_population = population;
    used_rime = false(NP, 1);
    used_source = zeros(NP, 1);
    used_eig = false(NP, 1);
    used_exemplar = false(NP, 1);
    used_evolution_path = false(NP, 1);
    junior_probability = max(juniors_min_probability, (1 - progress) ^ junior_power);

    for i = 1:NP
        if adaptive_knowledge_sources
            active_probability = source_probability;
            active_probability(1) = active_probability(1) * (0.55 + junior_probability);
            active_probability(2) = active_probability(2) * (1.45 - junior_probability);
            active_probability = active_probability ./ sum(active_probability);
            source = sample_discrete(active_probability);
        elseif rand < junior_probability
            source = 1;
        else
            source = 2;
        end
        used_source(i) = source;
        if source == 1
            better = randi(top_count);
            worse = bottom_start - 1 + randi(NP - bottom_start + 1);
            r1 = random_index_except(NP, i);
            r2 = random_index_except(NP, r1);
            mutant = population(i, :) ...
                + KF(i) .* (population(better, :) - population(worse, :)) ...
                + junior_rand_weight * KF(i) .* (population(r1, :) - population(r2, :));
        elseif source == 2
            senior_top = randi(top_count);
            senior_mid = mid_start - 1 + randi(max(1, bottom_start - mid_start));
            senior_bottom = bottom_start - 1 + randi(NP - bottom_start + 1);
            mutant = population(i, :) ...
                + senior_best_weight * KF(i) .* (population(1, :) - population(i, :)) ...
                + senior_top_weight * KF(i) .* (population(senior_top, :) - population(senior_bottom, :)) ...
                + senior_mid_weight * KF(i) .* (population(senior_mid, :) - population(i, :));
        else
            p_count = max(2, round((0.08 + 0.12 * (1 - progress)) * NP));
            pbest = randi(p_count);
            r1 = random_index_except(NP, i);
            donor_pool = [population; cross_archive];
            r2 = random_pool_index(size(donor_pool, 1), [i, r1]);
            mutant = population(i, :) ...
                + cross_rank_weight * KF(i) .* (population(pbest, :) - population(i, :)) ...
                + KF(i) .* (population(r1, :) - donor_pool(r2, :));
        end
        mask = rand(1, D) <= KR;
        mask(randi(D)) = true;
        if eig_success_gate && rand() < eig_gate_rate
            used_eig(i) = true;
            trial = eigen_crossover(population(i, :), mutant, KR, eig_basis, eig_center);
        else
            trial = population(i, :);
            trial(mask) = mutant(mask);
        end
        active_rime_rate = rime_puncture_rate;
        if rime_success_gate
            active_rime_rate = adaptive_rime_rate;
        end
        if active_rime_rate > 0 && progress >= rime_start_progress && rand() < active_rime_rate
            used_rime(i) = true;
            block = rand(1, D) < rime_block_rate;
            if ~any(block)
                block(randi(D)) = true;
            end
            local_radius = (rime_radius * (1 - 0.65 * progress) + 2e-5) .* span;
            trial(block) = best_position(block) + randn(1, nnz(block)) .* local_radius(block);
            hard = block & (rand(1, D) < rime_hard_probability);
            trial(hard) = best_position(hard);
        end
        if linewell_rate > 0 && progress >= linewell_start_progress && rand() < linewell_rate
            donor_a = population(randi(top_count), :);
            donor_b = population(randi(max(2, round(0.55 * NP))), :);
            direction = donor_a - donor_b;
            if norm(direction) < eps || rand() < 0.52
                block = rand(1, D) < linewell_block_rate;
                if ~any(block)
                    block(randi(D)) = true;
                end
                direction(~block) = 0;
                if norm(direction) < eps
                    direction(block) = randn(1, nnz(block)) .* span(block);
                end
            end
            direction = direction ./ max(norm(direction), eps);
            jump = linewell_step_scales(randi(numel(linewell_step_scales))) .* norm(span);
            if rand() < 0.5
                jump = -jump;
            end
            well_trial = best_position + jump .* direction;
            trial = (1 - linewell_blend) .* trial + linewell_blend .* well_trial;
        end
        if exemplar_success_gate && progress >= exemplar_start_progress && rand() < exemplar_rate
            used_exemplar(i) = true;
            exemplar_count = max(4, min(NP, round(exemplar_elite_rate * NP)));
            block = rand(1, D) < exemplar_block_rate;
            if ~any(block)
                block(randi(D)) = true;
            end
            block_dims = find(block);
            exemplar = trial;
            for dim_idx = 1:numel(block_dims)
                donor_a = randi(exemplar_count);
                donor_b = randi(exemplar_count);
                donor = min(donor_a, donor_b);
                d = block_dims(dim_idx);
                exemplar(d) = population(donor, d);
            end
            trial(block) = trial(block) + exemplar_blend .* (exemplar(block) - trial(block));
        end
        if evolution_path_enabled && progress >= evolution_path_start_progress && ...
                norm(evolution_path) > eps && rand() < evolution_path_rate
            used_evolution_path(i) = true;
            direction = evolution_path ./ max(norm(evolution_path), eps);
            scale = evolution_path_step_scales(randi(numel(evolution_path_step_scales))) .* ...
                max(evolution_path_scale, 1e-6 * norm(span));
            path_trial = best_position + scale .* direction;
            trial = (1 - evolution_path_blend) .* trial + evolution_path_blend .* path_trial;
        end
        trial_population(i, :) = min(max(trial, lb), ub);
    end

    trial_fitness = SOP_cec_evaluate(trial_population, problem);
    eval_count = eval_count + numel(trial_fitness);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    improved = trial_fitness <= fitness;
    if eig_success_gate && any(used_eig)
        eig_success = nnz(improved & used_eig);
        eig_trials = nnz(used_eig);
        target_rate = eig_gate_rate * (0.35 + 1.65 * eig_success / max(1, eig_trials));
        if eig_success == 0
            target_rate = 0.58 * eig_gate_rate;
        end
        eig_gate_rate = 0.78 * eig_gate_rate + 0.22 * target_rate;
        eig_gate_rate = min(eig_rate_max, max(eig_rate_min, eig_gate_rate));
    end
    if exemplar_success_gate && any(used_exemplar)
        exemplar_trials = nnz(used_exemplar);
        exemplar_success = nnz(improved & used_exemplar);
        target_rate = exemplar_rate * (0.35 + 1.75 * exemplar_success / max(1, exemplar_trials));
        if exemplar_success == 0
            target_rate = 0.56 * exemplar_rate;
        end
        exemplar_rate = 0.80 * exemplar_rate + 0.20 * target_rate;
        exemplar_rate = min(exemplar_rate_max, max(exemplar_rate_min, exemplar_rate));
    end
    if evolution_path_enabled && any(used_evolution_path)
        path_trials = nnz(used_evolution_path);
        path_success = nnz(improved & used_evolution_path);
        target_rate = evolution_path_rate * (0.35 + 1.85 * path_success / max(1, path_trials));
        if path_success == 0
            target_rate = 0.55 * evolution_path_rate;
        end
        evolution_path_rate = 0.80 * evolution_path_rate + 0.20 * target_rate;
        evolution_path_rate = min(evolution_path_rate_max, max(evolution_path_rate_min, evolution_path_rate));
    end
    if adaptive_knowledge_sources
        old_population = population;
        source_score = zeros(1, 3);
        source_count = zeros(1, 3);
        gain = max(0, fitness - trial_fitness);
        for source = 1:3
            source_rows = used_source == source;
            source_count(source) = nnz(source_rows);
            if source_count(source) > 0
                source_score(source) = (nnz(improved & source_rows) + ...
                    sum(gain(source_rows)) / max(eps, sum(abs(fitness(source_rows))))) ...
                    / source_count(source);
            end
        end
        target_probability = source_score + source_min_probability;
        target_probability = target_probability ./ sum(target_probability);
        source_probability = (1 - source_learning_rate) .* source_probability + ...
            source_learning_rate .* target_probability;
        source_probability = max(source_min_probability, source_probability);
        source_probability = source_probability ./ sum(source_probability);
        if any(improved)
            cross_archive = [cross_archive; old_population(improved, :)]; %#ok<AGROW>
            archive_limit = max(NP, round(cross_archive_factor * NP));
            if size(cross_archive, 1) > archive_limit
                cross_archive = cross_archive(randperm(size(cross_archive, 1), archive_limit), :);
            end
        end
    end
    if rime_success_gate && any(used_rime)
        rime_trials = nnz(used_rime);
        rime_success = nnz(improved & used_rime);
        rime_success_rate = rime_success / max(1, rime_trials);
        target_rate = rime_puncture_rate * (0.35 + 1.65 * rime_success_rate);
        if rime_success == 0
            target_rate = 0.62 * adaptive_rime_rate;
        end
        adaptive_rime_rate = 0.76 * adaptive_rime_rate + 0.24 * target_rate;
        adaptive_rime_rate = min(rime_rate_max, max(rime_rate_min, adaptive_rime_rate));
    end
    population(improved, :) = trial_population(improved, :);
    fitness(improved) = trial_fitness(improved);
    [current_best, current_idx] = min(fitness);
    if current_best < best_raw
        displacement = population(current_idx, :) - best_position;
        displacement_norm = norm(displacement);
        if evolution_path_enabled && displacement_norm > eps
            unit_displacement = displacement ./ displacement_norm;
            evolution_path = (1 - evolution_path_learning_rate) .* evolution_path + ...
                evolution_path_learning_rate .* unit_displacement;
            evolution_path_scale = (1 - evolution_path_learning_rate) .* evolution_path_scale + ...
                evolution_path_learning_rate .* displacement_norm;
        end
        best_raw = current_best;
        best_position = population(current_idx, :);
    end
    curve(iter) = best_raw;
end

runtime = toc(t_start);
curve = curve(1:actual_iter);
[final_fitness, final_order] = sort(fitness);
final_population = population(final_order, :);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.runtime = runtime;
result.iteration = actual_iter;
result.population_num = NP;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Gaining-Sharing Knowledge (GSK)\nJunior-senior knowledge sharing');
if adaptive_knowledge_sources
    result.algorithm_combination = sprintf('%s\nSuccess-adaptive junior, senior, and rank-difference knowledge sources', ...
        result.algorithm_combination);
end
if eig_success_gate
    result.algorithm_combination = sprintf('%s\nSuccess-gated elite eigen-coordinate learning', ...
        result.algorithm_combination);
end
if exemplar_success_gate
    result.algorithm_combination = sprintf('%s\nSuccess-gated dimension-wise elite exemplar learning', ...
        result.algorithm_combination);
end
if evolution_path_enabled
    result.algorithm_combination = sprintf('%s\nSuccess-adaptive evolution-path extrapolation', ...
        result.algorithm_combination);
end
result.combination_number = 1;
result.agent_id = 'Agent2';
result.problem = problem;
result.final_population = final_population;
result.final_fitness = final_fitness;

if verbose
    fprintf('GSK finished %s %dD F%d: best %.12g, report %.12g, runtime %.4f s.\n', ...
        problem.suite, D, problem.func_num, result.best_value, result.record_value, runtime);
end
end

function idx = random_index_except(NP, banned)
idx = randi(NP - 1);
if idx >= banned
    idx = idx + 1;
end
end

function idx = random_pool_index(pool_size, banned)
idx = randi(pool_size);
tries = 0;
while any(idx == banned) && tries < 30
    idx = randi(pool_size);
    tries = tries + 1;
end
end

function idx = sample_discrete(probability)
edge = cumsum(probability);
idx = find(edge >= rand(), 1, 'first');
if isempty(idx)
    idx = numel(probability);
end
end

function [basis, center] = elite_eigen_basis(population, span, elite_rate)
elite_count = max(4, min(size(population, 1), round(elite_rate * size(population, 1))));
elites = population(1:elite_count, :);
weights = log(elite_count + 0.5) - log(1:elite_count);
weights = weights ./ sum(weights);
center = weights * elites;
centered = elites - center;
cov_matrix = centered' * (centered .* weights') + diag((0.002 * span) .^ 2 + 1e-14);
cov_matrix = 0.5 * (cov_matrix + cov_matrix');
[basis, values] = eig(cov_matrix, 'vector');
if ~isvector(values)
    values = diag(values);
end
[~, order] = sort(values, 'descend');
basis = real(basis(:, order));
if any(~isfinite(basis(:)))
    basis = eye(size(population, 2));
end
end

function trial = eigen_crossover(parent, mutant, CR, basis, center)
parent_z = (parent - center) * basis;
mutant_z = (mutant - center) * basis;
mask = rand(1, numel(parent)) <= CR;
mask(randi(numel(parent))) = true;
trial_z = parent_z;
trial_z(mask) = mutant_z(mask);
trial = trial_z * basis' + center;
end

function radius = make_radius(initial_radius, span, D)
radius = initial_radius;
if isscalar(radius)
    radius = radius .* span;
else
    radius = reshape(radius, 1, []);
    if numel(radius) ~= D
        radius = median(radius(:)) .* ones(1, D);
    end
end
radius = max(radius, eps);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
