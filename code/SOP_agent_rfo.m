function result = SOP_agent_rfo(problem, seed, options)
% Ruppell's Fox Optimizer based on the public mathematical model.
%
% The implementation follows the paper's four sequential behaviors:
% day/night sight-hearing search, smell search, school movement toward the
% best fox, and worst-case random exploration. It is an original MATLAB
% implementation and uses objective feedback only.
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
NP = get_option(options, 'population_num', 100);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
beta = get_option(options, 'beta', 1e-10);
e0 = get_option(options, 'e0', 1.0);
e1 = get_option(options, 'e1', 3.0);
c0 = get_option(options, 'c0', 2.0);
c1 = get_option(options, 'c1', 2.0);
a0 = get_option(options, 'a0', 2.0);
a1 = get_option(options, 'a1', 3.0);
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
personal_best = population;
personal_best_fitness = fitness;
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
estimated_iterations = max(1, ceil((max_fes - NP) / max(1, 4 * NP)));
curve = zeros(estimated_iterations, 1);
iter = 0;

while eval_count < max_fes && toc(t_start) < max_runtime_sec
    iter = iter + 1;
    progress_index = min(estimated_iterations, iter);
    logistic_argument = (estimated_iterations / 2 - progress_index) / 100;
    logistic_argument = min(60, max(-60, logistic_argument));
    sight = 1 / (1 + exp(logistic_argument));
    hearing = 1 / (1 + exp(-logistic_argument));
    is_day = rand() >= 0.5;

    candidate = population;
    for i = 1:NP
        direct_search = false;
        rotation_angle = 0;
        if is_day && sight >= hearing
            direct_search = rand() >= 0.25;
            rotation_angle = 260;
        elseif is_day
            direct_search = rand() >= 0.75;
            rotation_angle = 150;
        elseif hearing >= sight
            direct_search = rand() >= 0.25;
            rotation_angle = 150;
        else
            direct_search = rand() >= 0.75;
            rotation_angle = 260;
        end

        if direct_search
            random_best = personal_best(randi(NP), :);
            candidate(i, :) = population(i, :) ...
                + signed_random() .* (random_best - population(i, :)) ...
                + signed_random() .* (best_position - population(i, :));
        else
            center = personal_best(randi(NP), :);
            candidate(i, :) = rotate_pairs(population(i, :), center, rotation_angle) ...
                + beta .* randn(1, D) .* span .* random_sign();
        end
    end
    [population, fitness, personal_best, personal_best_fitness, best_position, best_raw, ...
        eval_delta] = greedy_phase(problem, population, fitness, candidate, personal_best, ...
        personal_best_fitness, best_position, best_raw, lb, ub, max_fes - eval_count);
    eval_count = eval_count + eval_delta;
    if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
        curve(iter) = best_raw;
        break;
    end

    smell = 0.1 * abs(cos(2 * sight));
    candidate = population;
    for i = 1:NP
        random_best = personal_best(randi(NP), :);
        epr = e0 + rand() * (e1 - e0);
        alpha = abs(2 * rand() - (rand() + rand())) ^ (0.5 * epr);
        if rand() >= smell
            candidate(i, :) = population(i, :) ...
                + alpha .* (random_best - population(i, :)) .* rand() ...
                + alpha .* (best_position - population(i, :)) .* rand();
        else
            candidate(i, :) = random_best + beta .* randn(1, D) .* span;
        end
    end
    [population, fitness, personal_best, personal_best_fitness, best_position, best_raw, ...
        eval_delta] = greedy_phase(problem, population, fitness, candidate, personal_best, ...
        personal_best_fitness, best_position, best_raw, lb, ub, max_fes - eval_count);
    eval_count = eval_count + eval_delta;
    if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
        curve(iter) = best_raw;
        break;
    end

    candidate = population;
    for i = 1:NP
        random_best = personal_best(randi(NP), :);
        if rand() >= 0.1
            candidate(i, :) = population(i, :) ...
                + c0 .* (random_best - population(i, :)) .* rand() ...
                + c1 .* (best_position - population(i, :)) .* rand();
        else
            peer_best = personal_best(randi(NP), :);
            updated = population(i, :) ...
                + a0 .* (best_position - population(i, :)) .* rand() ...
                + a1 .* (peer_best - personal_best(i, :)) .* rand();
            candidate(i, :) = 0.5 .* (updated + population(i, :));
        end
    end
    [population, fitness, personal_best, personal_best_fitness, best_position, best_raw, ...
        eval_delta] = greedy_phase(problem, population, fitness, candidate, personal_best, ...
        personal_best_fitness, best_position, best_raw, lb, ub, max_fes - eval_count);
    eval_count = eval_count + eval_delta;
    if eval_count >= max_fes || toc(t_start) >= max_runtime_sec
        curve(iter) = best_raw;
        break;
    end

    candidate = population;
    active = rand(NP, 1) >= 0.5;
    candidate(active, :) = candidate(active, :) + beta .* randn(nnz(active), D) .* repmat(span, nnz(active), 1);
    [population, fitness, personal_best, personal_best_fitness, best_position, best_raw, ...
        eval_delta] = greedy_phase(problem, population, fitness, candidate, personal_best, ...
        personal_best_fitness, best_position, best_raw, lb, ub, max_fes - eval_count);
    eval_count = eval_count + eval_delta;
    curve(iter) = best_raw;
end

curve = curve(1:iter);
runtime = toc(t_start);
[final_fitness, order] = sort(fitness);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(curve, problem);
result.raw_convergence_curve = curve;
result.runtime = runtime;
result.iteration = iter;
result.population_num = NP;
result.evaluation_count = eval_count;
result.algorithm_combination = sprintf(['Ruppell''s Fox Optimizer (RFO)\n' ...
    'Day/night sight-hearing search, smell search, schooling, and worst-case exploration']);
result.combination_number = 1;
result.agent_id = 'Agent2';
result.problem = problem;
result.final_population = population(order, :);
result.final_fitness = final_fitness;

if verbose
    fprintf('RFO finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function [population, fitness, personal_best, personal_best_fitness, best_position, best_raw, eval_count] = ...
        greedy_phase(problem, population, fitness, candidate, personal_best, personal_best_fitness, ...
        best_position, best_raw, lb, ub, remaining)
eval_count = 0;
if remaining <= 0
    return;
end
candidate = min(max(candidate, lb), ub);
rows = min(size(candidate, 1), remaining);
candidate = candidate(1:rows, :);
candidate_fitness = SOP_cec_evaluate(candidate, problem);
eval_count = numel(candidate_fitness);
improved = candidate_fitness <= fitness(1:rows);
improved_rows = find(improved);
population(improved_rows, :) = candidate(improved_rows, :);
fitness(improved_rows) = candidate_fitness(improved_rows);
personal_improved = candidate_fitness < personal_best_fitness(1:rows);
personal_rows = find(personal_improved);
personal_best(personal_rows, :) = candidate(personal_rows, :);
personal_best_fitness(personal_rows) = candidate_fitness(personal_rows);
[current_best, current_idx] = min(fitness);
if current_best < best_raw
    best_raw = current_best;
    best_position = population(current_idx, :);
end
end

function value = signed_random()
value = random_sign() * rand();
end

function value = random_sign()
if rand() < 0.5
    value = -1;
else
    value = 1;
end
end

function rotated = rotate_pairs(point, center, max_angle_degree)
D = numel(point);
rotated = point;
offset = point - center;
order = randperm(D);
for k = 1:2:(D - 1)
    d1 = order(k);
    d2 = order(k + 1);
    theta = rand() * max_angle_degree * pi / 180;
    c = cos(theta);
    s = sin(theta);
    rotated(d1) = center(d1) + c * offset(d1) - s * offset(d2);
    rotated(d2) = center(d2) + s * offset(d1) + c * offset(d2);
end
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
