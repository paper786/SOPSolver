function result = SOP_agent_gsk_pattern_refine(problem, seed, options)
% GSK basin search followed by adaptive stochastic pattern refinement.
%
% The refinement is a direct-search metaheuristic layer: it probes single
% coordinates, random subspaces, and elite-difference directions around the
% best solution returned by GSK, shrinking the perturbation radius when the
% local basin stops improving. It uses only objective feedback.
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
population_num = get_option(options, 'population_num', 300);
base_fraction = get_option(options, 'base_fraction', 0.68);

gsk_options = options;
gsk_options.max_runtime_sec = base_fraction * max_runtime_sec;
gsk_options.max_iter = max(1, floor(base_fraction * get_option(options, 'max_iter', floor(max_fes / population_num))));
gsk_options.verbose = false;
gsk_result = SOP_agent_gsk(problem, double(seed), gsk_options);

D = problem.dimension;
lb = problem.lb;
ub = problem.ub;
span = ub - lb;
best_x = gsk_result.best_position;
best_raw = gsk_result.best_value;
eval_count = gsk_result.evaluation_count;
if isfield(gsk_result, 'final_population') && ~isempty(gsk_result.final_population)
    elite_pop = gsk_result.final_population;
else
    elite_pop = repmat(best_x, population_num, 1);
end

batch = get_option(options, 'refine_samples', max(96, round(0.45 * population_num)));
sigma = get_option(options, 'pattern_sigma', 0.006) .* span;
min_sigma = get_option(options, 'min_sigma', 1e-9) .* max(1, span);
reset_sigma = get_option(options, 'reset_sigma', 0.0025) .* span;
block_rate = get_option(options, 'block_rate', min(0.16, max(0.03, 6 / D)));
stall = 0;
refine_iter = 0;
curve = gsk_result.convergence_curve(:);

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    refine_iter = refine_iter + 1;
    count = min(batch, max_fes - eval_count);
    candidates = make_pattern_candidates(best_x, elite_pop, lb, ub, sigma, block_rate, count);
    values = SOP_cec_evaluate(candidates, problem);
    eval_count = eval_count + numel(values);
    if toc(t_start) >= max_runtime_sec
        break;
    end
    [trial_raw, trial_idx] = min(values);
    if trial_raw < best_raw
        best_raw = trial_raw;
        best_x = candidates(trial_idx, :);
        elite_pop = update_elite(elite_pop, best_x, candidates, values);
        sigma = max(0.985 .* sigma, min_sigma);
        stall = 0;
    else
        sigma = max(0.72 .* sigma, min_sigma);
        stall = stall + 1;
    end
    if stall >= 18
        sigma = max(sigma, reset_sigma .* (0.7 + 0.6 * rand(1, D)));
        stall = 0;
    end
    if mod(refine_iter, 10) == 0
        curve(end + 1, 1) = SOP_cec_record_value(best_raw, problem); %#ok<AGROW>
    end
end

result = gsk_result;
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_x;
result.convergence_curve = curve;
result.runtime = toc(t_start);
result.iteration = gsk_result.iteration + refine_iter;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf('Gaining-Sharing Knowledge (GSK)\nAdaptive stochastic pattern/subspace refinement');
result.combination_number = 3;
result.agent_id = 'Agent1';
end

function candidates = make_pattern_candidates(best_x, elite_pop, lb, ub, sigma, block_rate, count)
D = numel(best_x);
NP = size(elite_pop, 1);
candidates = repmat(best_x, count, 1);
for i = 1:count
    mode = rand();
    candidate = best_x;
    if mode < 0.34
        d = randi(D);
        direction = 2 * (rand() < 0.5) - 1;
        candidate(d) = candidate(d) + direction * sigma(d) * (0.5 + rand());
    elseif mode < 0.72
        mask = rand(1, D) < block_rate;
        if ~any(mask)
            mask(randi(D)) = true;
        end
        noise = randn(1, D);
        if rand() < 0.35
            cauchy_noise = tan(pi * (rand(1, D) - 0.5));
            noise(mask) = min(max(cauchy_noise(mask), -8), 8);
        end
        candidate(mask) = candidate(mask) + noise(mask) .* sigma(mask);
    else
        r1 = randi(NP);
        r2 = randi(NP);
        diff = elite_pop(r1, :) - elite_pop(r2, :);
        mask = rand(1, D) < max(block_rate, 4 / D);
        if ~any(mask)
            mask(randi(D)) = true;
        end
        candidate(mask) = candidate(mask) + (0.2 + 0.6 * rand()) .* diff(mask);
        if rand() < 0.5
            candidate(mask) = 0.75 .* candidate(mask) + 0.25 .* best_x(mask);
        end
    end
    candidates(i, :) = min(max(candidate, lb), ub);
end
end

function elite_pop = update_elite(elite_pop, best_x, candidates, values)
[~, order] = sort(values);
replace_count = min(max(2, floor(0.08 * size(elite_pop, 1))), size(candidates, 1));
elite_pop(1, :) = best_x;
if size(elite_pop, 1) >= replace_count + 1
    elite_pop(end - replace_count + 1:end, :) = candidates(order(1:replace_count), :);
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
