function result = SOP_agent_lshade_subspace_refine(problem, seed, options)
% L-SHADE followed by stochastic subspace refinement.
%
% Literature basis: Differential Evolution/L-SHADE combined with simulated
% annealing style random local search. The refinement perturbs random
% coordinate subsets and accepts only objective improvements.
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
max_fes = get_option(options, 'max_fes', 10000 * problem.dimension);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
base_fraction = get_option(options, 'base_fraction', 0.78);
base_algorithm = lower(string(get_option(options, 'base_algorithm', 'lshade')));
base_options = options;
base_options.max_fes = max(1000, floor(base_fraction * max_fes));
base_options.max_runtime_sec = 0.78 * max_runtime_sec;
base_options.verbose = false;

switch base_algorithm
    case "lshade_cma"
        base = SOP_agent_lshade_cma(problem, seed, base_options);
        base_name = 'L-SHADE with elite covariance sampling';
    case "lshade_jso"
        base = SOP_agent_lshade_jso(problem, seed, base_options);
        base_name = 'jSO/L-SHADE success-history adaptation';
    otherwise
        base = SOP_agent_lshade(problem, seed, base_options);
        base_name = 'L-SHADE success-history adaptation';
end

D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
best_x = base.best_position;
best_raw = base.best_value;
eval_count = base.evaluation_count;
curve = base.raw_convergence_curve(:);

samples = get_option(options, 'refine_samples', max(24, round(0.35 * get_option(options, 'population_num', 180))));
subspace_rate = get_option(options, 'subspace_rate', min(0.20, max(0.04, 8 / D)));
adaptive_subspace = get_option(options, 'adaptive_subspace_rate', false);
subspace_rate_min = get_option(options, 'subspace_rate_min', max(1 / D, 0.70 * subspace_rate));
subspace_rate_max = get_option(options, 'subspace_rate_max', min(0.20, 1.60 * subspace_rate));
subspace_success_shrink = get_option(options, 'subspace_success_shrink', 0.94);
subspace_stall_expand = get_option(options, 'subspace_stall_expand', 1.05);
sigma = get_option(options, 'refine_sigma', 0.018) .* span;
min_sigma = get_option(options, 'min_sigma', 1e-7) .* span;
reset_sigma = get_option(options, 'reset_sigma', 0.035) .* span;
stall = 0;
refine_iter = 0;
refine_mode = lower(string(get_option(options, 'refine_mode', 'sa_subspace')));
refine_label = 'Stochastic subspace simulated-annealing refinement';

if refine_mode == "tlbo"
    [best_x, best_raw, eval_count, curve, refine_iter] = tlbo_subspace_refine(problem, base, best_x, best_raw, ...
        eval_count, curve, max_fes, max_runtime_sec, t_start, options);
    refine_label = 'Teaching-Learning-Based Optimization subspace refinement';
elseif refine_mode == "snap_tlbo"
    [best_x, best_raw, eval_count, curve, refine_iter] = snap_tlbo_refine(problem, base, best_x, best_raw, ...
        eval_count, curve, max_fes, max_runtime_sec, t_start, options);
    refine_label = 'Elite differential snap-step and TLBO subspace refinement';
elseif refine_mode == "eda_block_tlbo_gsk"
    [best_x, best_raw, eval_count, curve, refine_iter] = eda_block_tlbo_gsk_refine(problem, base, best_x, best_raw, ...
        eval_count, curve, max_fes, max_runtime_sec, t_start, options);
    refine_label = 'Elite EDA block learning with TLBO/GSK/RIME block refinement';
else
while eval_count < max_fes && toc(t_start) < max_runtime_sec
    refine_iter = refine_iter + 1;
    count = min(samples, max_fes - eval_count);
    masks = rand(count, D) < subspace_rate;
    empty = ~any(masks, 2);
    if any(empty)
        cols = randi(D, sum(empty), 1);
        rows = find(empty);
        for k = 1:numel(rows)
            masks(rows(k), cols(k)) = true;
        end
    end
    noise = randn(count, D);
    cauchy_mask = rand(count, D) < 0.35;
    cauchy_noise = tan(pi * (rand(count, D) - 0.5));
    cauchy_noise = min(max(cauchy_noise, -8), 8);
    noise(cauchy_mask) = cauchy_noise(cauchy_mask);
    candidates = repmat(best_x, count, 1) + masks .* noise .* repmat(sigma, count, 1);
    candidates = min(max(candidates, lb), ub);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(idx, :);
        sigma = max(0.985 .* sigma, min_sigma);
        if adaptive_subspace
            subspace_rate = max(subspace_rate_min, subspace_success_shrink * subspace_rate);
        end
        stall = 0;
    else
        sigma = max(0.94 .* sigma, min_sigma);
        stall = stall + 1;
        if adaptive_subspace && mod(stall, 8) == 0
            subspace_rate = min(subspace_rate_max, subspace_stall_expand * subspace_rate);
        end
    end
    if stall >= 35
        sigma = max(sigma, reset_sigma .* (0.70 + 0.60 * rand(1, D)));
        stall = 0;
    end
    if mod(refine_iter, 20) == 0
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
    end
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
result.algorithm_combination = sprintf('Differential Evolution (DE)\n%s\n%s', base_name, refine_label);
result.combination_number = 5;
result.agent_id = 'Agent1';
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end

function [best_x, best_raw, eval_count, curve, refine_iter] = tlbo_subspace_refine(problem, base, best_x, best_raw, ...
    eval_count, curve, max_fes, max_runtime_sec, t_start, options)
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
learner_count = get_option(options, 'tlbo_learners', max(24, round(0.22 * get_option(options, 'population_num', 180))));
learner_count = max(8, learner_count);
if isfield(base, 'final_population') && ~isempty(base.final_population)
    rows = min(learner_count, size(base.final_population, 1));
    learners = base.final_population(1:rows, :);
    if isfield(base, 'final_fitness') && numel(base.final_fitness) >= rows
        learner_fit = base.final_fitness(1:rows);
    else
        learner_fit = SOP_cec_evaluate(learners, problem);
        eval_count = eval_count + numel(learner_fit);
    end
    if rows < learner_count
        extra = repmat(best_x, learner_count - rows, 1) + randn(learner_count - rows, D) .* ...
            repmat(get_option(options, 'tlbo_init_sigma', 0.003) .* span, learner_count - rows, 1);
        extra = min(max(extra, lb), ub);
        extra_fit = SOP_cec_evaluate(extra, problem);
        eval_count = eval_count + numel(extra_fit);
        learners = [learners; extra]; %#ok<AGROW>
        learner_fit = [learner_fit(:); extra_fit(:)]; %#ok<AGROW>
    end
else
    learners = repmat(best_x, learner_count, 1) + randn(learner_count, D) .* ...
        repmat(get_option(options, 'tlbo_init_sigma', 0.003) .* span, learner_count, 1);
    learners(1, :) = best_x;
    learners = min(max(learners, lb), ub);
    learner_fit = SOP_cec_evaluate(learners, problem);
    eval_count = eval_count + numel(learner_fit);
end

subspace_rate = get_option(options, 'subspace_rate', min(0.16, max(0.04, 6 / D)));
teacher_scale = get_option(options, 'tlbo_teacher_scale', 1.0);
learner_scale = get_option(options, 'tlbo_learner_scale', 0.72);
noise_sigma = get_option(options, 'tlbo_noise_sigma', 0.0015) .* span;
min_noise = get_option(options, 'min_sigma', 1e-7) .* span;
stall = 0;
refine_iter = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    refine_iter = refine_iter + 1;
    [learner_fit, order] = sort(learner_fit);
    learners = learners(order, :);
    if learner_fit(1) < best_raw
        best_raw = learner_fit(1);
        best_x = learners(1, :);
    end
    teacher = learners(1, :);
    mean_x = mean(learners, 1);
    count = min(size(learners, 1), max_fes - eval_count);
    if count <= 0
        break;
    end
    candidates = learners(1:count, :);
    for i = 1:count
        mask = rand(1, D) < subspace_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        teaching_factor = 1 + double(rand() < 0.5);
        step = teacher_scale .* rand(1, D) .* (teacher - teaching_factor .* mean_x);
        noise = randn(1, D) .* noise_sigma;
        if rand() < get_option(options, 'tlbo_cauchy_rate', 0.25)
            cauchy_noise = tan(pi * (rand(1, D) - 0.5));
            cauchy_noise = min(max(cauchy_noise, -7), 7);
            noise = cauchy_noise .* noise_sigma;
        end
        candidates(i, mask) = candidates(i, mask) + step(mask) + noise(mask);
    end
    candidates = min(max(candidates, lb), ub);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    improved = values(:) < learner_fit(1:count);
    learners(improved, :) = candidates(improved, :);
    learner_fit(improved) = values(improved);

    remaining = max_fes - eval_count;
    count = min(size(learners, 1), remaining);
    if count <= 0
        break;
    end
    pair_candidates = learners(1:count, :);
    for i = 1:count
        partner = randi(size(learners, 1));
        while partner == i && size(learners, 1) > 1
            partner = randi(size(learners, 1));
        end
        if learner_fit(i) < learner_fit(partner)
            direction = learners(i, :) - learners(partner, :);
        else
            direction = learners(partner, :) - learners(i, :);
        end
        mask = rand(1, D) < subspace_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        pair_candidates(i, mask) = pair_candidates(i, mask) + learner_scale .* rand(1, sum(mask)) .* direction(mask);
    end
    pair_candidates = min(max(pair_candidates, lb), ub);
    pair_values = SOP_cec_evaluate(pair_candidates, problem);
    eval_count = eval_count + numel(pair_values);
    improved = pair_values(:) < learner_fit(1:count);
    learners(improved, :) = pair_candidates(improved, :);
    learner_fit(improved) = pair_values(improved);

    [trial_raw, idx] = min(learner_fit);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = learners(idx, :);
        noise_sigma = max(0.985 .* noise_sigma, min_noise);
        stall = 0;
    else
        noise_sigma = max(0.94 .* noise_sigma, min_noise);
        stall = stall + 1;
    end
    if stall >= get_option(options, 'tlbo_reset_stall', 18)
        replace_count = max(2, round(get_option(options, 'tlbo_reset_rate', 0.18) * size(learners, 1)));
        rows = size(learners, 1) - replace_count + 1:size(learners, 1);
        learners(rows, :) = repmat(best_x, replace_count, 1) + randn(replace_count, D) .* ...
            repmat(get_option(options, 'reset_sigma', 0.010) .* span, replace_count, 1);
        learners(rows, :) = min(max(learners(rows, :), lb), ub);
        learner_fit(rows) = SOP_cec_evaluate(learners(rows, :), problem);
        eval_count = eval_count + numel(rows);
        stall = 0;
    end
    if mod(refine_iter, 10) == 0
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
    end
end
end

function [best_x, best_raw, eval_count, curve, refine_iter] = snap_tlbo_refine(problem, base, best_x, best_raw, ...
    eval_count, curve, max_fes, max_runtime_sec, t_start, options)
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
learner_count = get_option(options, 'snap_learners', max(48, round(0.22 * get_option(options, 'population_num', 180))));
learner_count = max(12, learner_count);
if isfield(base, 'final_population') && ~isempty(base.final_population)
    rows = min(learner_count, size(base.final_population, 1));
    learners = base.final_population(1:rows, :);
    if isfield(base, 'final_fitness') && numel(base.final_fitness) >= rows
        learner_fit = base.final_fitness(1:rows);
    else
        learner_fit = SOP_cec_evaluate(learners, problem);
        eval_count = eval_count + numel(learner_fit);
    end
else
    rows = 0;
    learners = zeros(0, D);
    learner_fit = zeros(0, 1);
end
if size(learners, 1) < learner_count && eval_count < max_fes
    extra_count = min(learner_count - size(learners, 1), max_fes - eval_count);
    sigma0 = get_option(options, 'snap_init_sigma', 0.0025) .* span;
    extra = repmat(best_x, extra_count, 1) + randn(extra_count, D) .* repmat(sigma0, extra_count, 1);
    extra = min(max(extra, lb), ub);
    extra_fit = SOP_cec_evaluate(extra, problem);
    eval_count = eval_count + numel(extra_fit);
    learners = [learners; extra]; %#ok<AGROW>
    learner_fit = [learner_fit(:); extra_fit(:)]; %#ok<AGROW>
end
if isempty(learners)
    learners = best_x;
    learner_fit = best_raw;
end

subspace_rate = get_option(options, 'subspace_rate', max(0.030, 5 / D));
sigma = get_option(options, 'snap_noise_sigma', 0.0016) .* span;
min_sigma = get_option(options, 'min_sigma', 1e-7) .* span;
reset_sigma = get_option(options, 'reset_sigma', 0.0075) .* span;
steps = get_option(options, 'snap_steps', [1.0, 0.5, 0.25, 2.0]);
step_weights = get_option(options, 'snap_step_weights', [0.32, 0.38, 0.24, 0.06]);
snap_rate = get_option(options, 'snap_rate', 0.42);
teacher_scale = get_option(options, 'snap_teacher_scale', 0.58);
learner_scale = get_option(options, 'snap_learner_scale', 0.52);
diff_scale = get_option(options, 'snap_diff_scale', 0.44);
stall = 0;
refine_iter = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    refine_iter = refine_iter + 1;
    [learner_fit, order] = sort(learner_fit);
    learners = learners(order, :);
    if learner_fit(1) < best_raw
        best_raw = learner_fit(1);
        best_x = learners(1, :);
    end
    teacher = learners(1, :);
    elite_count = max(4, min(size(learners, 1), round(0.24 * size(learners, 1))));
    elite = learners(1:elite_count, :);
    elite_center = mean(elite, 1);
    mean_x = mean(learners, 1);
    count = min(size(learners, 1), max_fes - eval_count);
    if count <= 0
        break;
    end
    candidates = learners(1:count, :);
    for i = 1:count
        mask = rand(1, D) < subspace_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        mode = rand();
        x = candidates(i, :);
        if mode < 0.42
            a = randi(elite_count);
            b = randi(elite_count);
            while b == a && elite_count > 1
                b = randi(elite_count);
            end
            step = teacher_scale .* rand(1, D) .* (teacher - x) + ...
                diff_scale .* (0.35 + 0.65 * rand()) .* (elite(a, :) - elite(b, :));
            x(mask) = x(mask) + step(mask);
        elseif mode < 0.70
            teaching_factor = 1 + double(rand() < 0.5);
            step = teacher_scale .* rand(1, D) .* (teacher - teaching_factor .* mean_x);
            x(mask) = x(mask) + step(mask);
        elseif mode < 0.88
            partner = randi(size(learners, 1));
            if learner_fit(i) < learner_fit(partner)
                direction = x - learners(partner, :);
            else
                direction = learners(partner, :) - x;
            end
            x(mask) = x(mask) + learner_scale .* rand(1, sum(mask)) .* direction(mask);
        else
            noise = tan(pi * (rand(1, D) - 0.5));
            noise = min(max(noise, -7), 7);
            x(mask) = teacher(mask) + noise(mask) .* sigma(mask);
        end
        if rand() < snap_rate
            if rand() < 0.65
                anchor = teacher;
            else
                anchor = elite_center;
            end
            x = snap_to_anchor(x, anchor, mask, steps, step_weights, get_option(options, 'snap_jitter', 0.014));
        end
        candidates(i, :) = min(max(x, lb), ub);
    end
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    improved = values(:) < learner_fit(1:count);
    learners(improved, :) = candidates(improved, :);
    learner_fit(improved) = values(improved);

    remaining = max_fes - eval_count;
    cross_count = min(max(4, round(0.18 * size(learners, 1))), remaining);
    if cross_count > 0
        cross_candidates = repmat(teacher, cross_count, 1);
        for k = 1:cross_count
            a = randi(elite_count);
            b = randi(elite_count);
            mask = rand(1, D) < get_option(options, 'snap_cross_rate', 0.36);
            if ~any(mask)
                mask(randi(D)) = true;
            end
            x = elite(a, :);
            x(mask) = elite(b, mask);
            fine_mask = rand(1, D) < subspace_rate;
            if ~any(fine_mask)
                fine_mask(randi(D)) = true;
            end
            x(fine_mask) = x(fine_mask) + randn(1, sum(fine_mask)) .* sigma(fine_mask);
            if rand() < snap_rate
                x = snap_to_anchor(x, teacher, fine_mask, steps, step_weights, get_option(options, 'snap_jitter', 0.014));
            end
            cross_candidates(k, :) = min(max(x, lb), ub);
        end
        cross_values = SOP_cec_evaluate(cross_candidates, problem);
        eval_count = eval_count + numel(cross_values);
        [worst_fit, worst_order] = sort(learner_fit, 'descend');
        replace_count = min(numel(cross_values), numel(worst_order));
        for k = 1:replace_count
            row = worst_order(k);
            if cross_values(k) < worst_fit(k)
                learners(row, :) = cross_candidates(k, :);
                learner_fit(row) = cross_values(k);
            end
        end
    end

    [trial_raw, idx] = min(learner_fit);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = learners(idx, :);
        sigma = max(0.988 .* sigma, min_sigma);
        stall = 0;
    else
        sigma = max(0.94 .* sigma, min_sigma);
        stall = stall + 1;
    end
    if stall >= get_option(options, 'snap_reset_stall', 16)
        replace_count = min(max(2, round(get_option(options, 'snap_reset_rate', 0.14) * size(learners, 1))), max_fes - eval_count);
        if replace_count > 0
            rows = size(learners, 1) - replace_count + 1:size(learners, 1);
            learners(rows, :) = repmat(best_x, replace_count, 1) + randn(replace_count, D) .* ...
                repmat(reset_sigma .* (0.70 + 0.60 * rand(1, D)), replace_count, 1);
            learners(rows, :) = min(max(learners(rows, :), lb), ub);
            learner_fit(rows) = SOP_cec_evaluate(learners(rows, :), problem);
            eval_count = eval_count + replace_count;
        end
        sigma = max(sigma, 0.55 .* reset_sigma);
        stall = 0;
    end
    if mod(refine_iter, 10) == 0
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
    end
end
end

function x = snap_to_anchor(x, anchor, mask, steps, weights, jitter)
if nargin < 5 || isempty(weights)
    weights = ones(size(steps));
end
weights = weights ./ sum(weights);
step = steps(find(cumsum(weights) >= rand(), 1, 'first'));
delta = (x(mask) - anchor(mask)) ./ step;
x(mask) = anchor(mask) + round(delta) .* step + jitter .* step .* randn(1, sum(mask));
end

function [best_x, best_raw, eval_count, curve, refine_iter] = eda_block_tlbo_gsk_refine(problem, base, best_x, best_raw, ...
    eval_count, curve, max_fes, max_runtime_sec, t_start, options)
D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
learner_count = get_option(options, 'eda_block_learners', max(52, round(0.20 * get_option(options, 'population_num', 180))));
learner_count = max(12, learner_count);
if isfield(base, 'final_population') && ~isempty(base.final_population)
    rows = min(learner_count, size(base.final_population, 1));
    learners = base.final_population(1:rows, :);
    if isfield(base, 'final_fitness') && numel(base.final_fitness) >= rows
        learner_fit = base.final_fitness(1:rows);
    else
        learner_fit = SOP_cec_evaluate(learners, problem);
        eval_count = eval_count + numel(learner_fit);
    end
else
    rows = 0;
    learners = zeros(0, D);
    learner_fit = zeros(0, 1);
end
if size(learners, 1) < learner_count && eval_count < max_fes
    extra_count = min(learner_count - size(learners, 1), max_fes - eval_count);
    init_sigma = get_option(options, 'eda_block_init_sigma', 0.0030) .* span;
    extra = repmat(best_x, extra_count, 1) + randn(extra_count, D) .* repmat(init_sigma, extra_count, 1);
    if get_option(options, 'eda_block_init_cauchy', true)
        cauchy = tan(pi * (rand(extra_count, D) - 0.5));
        cauchy = min(max(cauchy, -7), 7);
        use_cauchy = rand(extra_count, D) < 0.25;
        base_extra = repmat(best_x, extra_count, 1);
        sigma_matrix = repmat(init_sigma, extra_count, 1);
        extra(use_cauchy) = base_extra(use_cauchy) + cauchy(use_cauchy) .* sigma_matrix(use_cauchy);
    end
    extra = min(max(extra, lb), ub);
    extra_fit = SOP_cec_evaluate(extra, problem);
    eval_count = eval_count + numel(extra_fit);
    learners = [learners; extra]; %#ok<AGROW>
    learner_fit = [learner_fit(:); extra_fit(:)]; %#ok<AGROW>
end
if isempty(learners)
    learners = best_x;
    learner_fit = best_raw;
end

sigma = get_option(options, 'eda_block_sigma', 0.0025) .* span;
min_sigma = get_option(options, 'min_sigma', 1e-7) .* span;
reset_sigma = get_option(options, 'eda_block_reset_sigma', 0.0075) .* span;
block_size = get_option(options, 'eda_block_size', 8);
block_partitions = get_option(options, 'eda_block_partitions', 3);
batch = get_option(options, 'eda_block_batch', max(48, round(0.16 * get_option(options, 'population_num', 180))));
operator_weights = get_option(options, 'eda_block_operator_weights', [0.34, 0.28, 0.25, 0.13]);
operator_weights = operator_weights ./ sum(operator_weights);
teacher_scale = get_option(options, 'eda_block_teacher_scale', 0.58);
learner_scale = get_option(options, 'eda_block_learner_scale', 0.50);
gsk_scale = get_option(options, 'eda_block_gsk_scale', 0.46);
rime_rate = get_option(options, 'eda_block_rime_rate', 0.35);
stall = 0;
refine_iter = 0;
blocks = {};

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    refine_iter = refine_iter + 1;
    [learner_fit, order] = sort(learner_fit(:));
    learners = learners(order, :);
    if learner_fit(1) < best_raw
        best_raw = learner_fit(1);
        best_x = learners(1, :);
    end
    elite_count = max(4, min(size(learners, 1), round(0.28 * size(learners, 1))));
    elite = learners(1:elite_count, :);
    weights = log(elite_count + 0.5) - log((1:elite_count)');
    weights = weights ./ sum(weights);
    elite_center = weights' * elite;
    elite_sigma = sqrt(max(weights' * ((elite - elite_center) .^ 2), 0)) + 0.35 .* sigma;
    elite_sigma = max(elite_sigma, min_sigma);
    if isempty(blocks) || mod(refine_iter, get_option(options, 'eda_block_rebuild_interval', 8)) == 1
        blocks = build_eda_blocks(elite, D, block_size, block_partitions);
    end
    block_order = randperm(numel(blocks));
    blocks_this_iter = min(numel(block_order), get_option(options, 'eda_blocks_per_iter', max(2, ceil(0.35 * numel(blocks)))));
    improved_any = false;
    for b = 1:blocks_this_iter
        if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
            break;
        end
        dims = blocks{block_order(b)};
        count = min([batch, size(learners, 1), max_fes - eval_count]);
        if count <= 0
            break;
        end
        parent_idx = randperm(size(learners, 1), count);
        candidates = learners(parent_idx, :);
        mean_x = mean(learners, 1);
        worst_order = (size(learners, 1):-1:1)';
        for k = 1:count
            x = candidates(k, :);
            op = pick_weighted_index(operator_weights);
            if op == 1
                sample = elite_center(dims) + randn(1, numel(dims)) .* elite_sigma(dims);
                if rand() < get_option(options, 'eda_block_blx_rate', 0.38)
                    a = randi(elite_count);
                    b2 = randi(elite_count);
                    lo = min(elite(a, dims), elite(b2, dims));
                    hi = max(elite(a, dims), elite(b2, dims));
                    width = hi - lo;
                    alpha = get_option(options, 'eda_block_blx_alpha', 0.18);
                    sample = lo - alpha .* width + rand(1, numel(dims)) .* (1 + 2 * alpha) .* width;
                end
                x(dims) = sample;
            elseif op == 2
                teaching_factor = 1 + double(rand() < 0.5);
                step = teacher_scale .* rand(1, D) .* (best_x - teaching_factor .* mean_x);
                partner = randi(size(learners, 1));
                if learner_fit(parent_idx(k)) < learner_fit(partner)
                    direction = learners(parent_idx(k), :) - learners(partner, :);
                else
                    direction = learners(partner, :) - learners(parent_idx(k), :);
                end
                x(dims) = x(dims) + step(dims) + learner_scale .* rand(1, numel(dims)) .* direction(dims);
            elseif op == 3
                good = randi(elite_count);
                r1 = randi(size(learners, 1));
                r2 = randi(size(learners, 1));
                while r2 == r1 && size(learners, 1) > 1
                    r2 = randi(size(learners, 1));
                end
                direction = (elite(good, :) - abs(learners(r1, :))) + 0.5 .* (best_x - learners(r2, :));
                x(dims) = x(dims) + gsk_scale .* rand(1, numel(dims)) .* direction(dims);
            else
                if rand() < rime_rate
                    x(dims) = best_x(dims);
                end
                cauchy = tan(pi * (rand(1, numel(dims)) - 0.5));
                cauchy = min(max(cauchy, -7), 7);
                anchor = elite(randi(elite_count), :);
                x(dims) = 0.72 .* x(dims) + 0.28 .* anchor(dims) + cauchy .* sigma(dims);
            end
            if rand() < get_option(options, 'eda_block_snap_rate', 0.18)
                x(dims) = best_x(dims) + round((x(dims) - best_x(dims)) ./ 0.5) .* 0.5 + ...
                    get_option(options, 'eda_block_snap_jitter', 0.006) .* randn(1, numel(dims));
            end
            candidates(k, :) = min(max(x, lb), ub);
        end
        values = SOP_cec_evaluate(candidates, problem);
        eval_count = eval_count + numel(values);
        for k = 1:numel(values)
            row = parent_idx(k);
            if values(k) < learner_fit(row)
                learners(row, :) = candidates(k, :);
                learner_fit(row) = values(k);
                improved_any = true;
            elseif values(k) < learner_fit(worst_order(k))
                row = worst_order(k);
                learners(row, :) = candidates(k, :);
                learner_fit(row) = values(k);
                improved_any = true;
            end
        end
        [trial_raw, idx] = min(learner_fit);
        if trial_raw < best_raw
            best_raw = trial_raw;
            best_x = learners(idx, :);
        end
    end
    if improved_any
        sigma = max(0.988 .* sigma, min_sigma);
        stall = 0;
    else
        sigma = max(0.94 .* sigma, min_sigma);
        stall = stall + 1;
    end
    if stall >= get_option(options, 'eda_block_reset_stall', 12) && eval_count < max_fes
        replace_count = min(max(2, round(get_option(options, 'eda_block_reset_rate', 0.16) * size(learners, 1))), max_fes - eval_count);
        if replace_count > 0
            rows = size(learners, 1) - replace_count + 1:size(learners, 1);
            learners(rows, :) = repmat(best_x, replace_count, 1) + randn(replace_count, D) .* ...
                repmat(reset_sigma .* (0.65 + 0.70 * rand(1, D)), replace_count, 1);
            learners(rows, :) = min(max(learners(rows, :), lb), ub);
            learner_fit(rows) = SOP_cec_evaluate(learners(rows, :), problem);
            eval_count = eval_count + replace_count;
        end
        sigma = max(sigma, 0.60 .* reset_sigma);
        stall = 0;
    end
    if mod(refine_iter, 8) == 0
        curve(end + 1, 1) = best_raw; %#ok<AGROW>
    end
end
end

function blocks = build_eda_blocks(elite, D, block_size, partitions)
block_size = max(1, min(D, round(block_size)));
partitions = max(1, round(partitions));
if size(elite, 1) > 2
    corr_matrix = abs(corrcoef(elite));
    corr_matrix(~isfinite(corr_matrix)) = 0;
else
    corr_matrix = zeros(D, D);
end
corr_matrix(1:(D + 1):end) = 0;
variances = var(elite, 0, 1) + eps;
blocks = {};
for p = 1:partitions
    remaining = true(1, D);
    score_jitter = 0.80 + 0.40 * rand(1, D);
    while any(remaining)
        available = find(remaining);
        [~, seed_pos] = max(variances(available) .* score_jitter(available));
        seed_dim = available(seed_pos);
        scores = corr_matrix(seed_dim, available);
        scores(seed_pos) = inf;
        [~, order] = sort(scores, 'descend');
        take = min(block_size, numel(available));
        dims = available(order(1:take));
        remaining(dims) = false;
        blocks{end + 1} = dims; %#ok<AGROW>
    end
    variances = variances .* (0.85 + 0.30 * rand(1, D));
end
if isempty(blocks)
    blocks = {1:D};
end
end

function idx = pick_weighted_index(weights)
edges = cumsum(weights(:));
idx = find(edges >= rand(), 1, 'first');
if isempty(idx)
    idx = numel(weights);
end
end
