function result = SOP_agent_fno(problem, seed, options)
% Far and Near Optimization (FNO).
%
% Literature basis: Far and Near Optimization (FNO). The implementation
% follows the paper's two greedy phases: movement toward each member's
% farthest population member for exploration and nearest population member
% for exploitation. It only uses benchmark objective calls.
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
NP = get_option(options, 'population_num', 30);
max_fes = get_option(options, 'max_fes', 10000 * D);
max_iter = get_option(options, 'max_iter', floor((max_fes - NP) / (2 * NP)));
max_iter = max(1, max_iter);
update_mode = lower(string(get_option(options, 'update_mode', 'batch')));
batch_size = get_option(options, 'batch_size', min(NP, 5));
max_runtime_sec = get_option(options, 'max_runtime_sec', inf);
boundary_mode = lower(string(get_option(options, 'boundary_mode', 'clamp')));
verbose = get_option(options, 'verbose', false);

population = lb + rand(NP, D) .* span;
if get_option(options, 'include_center', false)
    population(1, :) = 0.5 * (lb + ub);
end
fitness = SOP_cec_evaluate(population, problem);
eval_count = numel(fitness);
[best_raw, best_idx] = min(fitness);
best_position = population(best_idx, :);
convergence_raw = zeros(max_iter, 1);

for iter = 1:max_iter
    if toc(t_start) >= max_runtime_sec
        convergence_raw = convergence_raw(1:max(0, iter - 1));
        break;
    end
    if update_mode == "generation"
        [far_idx, near_idx] = far_near_indices(population);

        rows = trim_rows((1:NP)', max_fes - eval_count);
        if ~isempty(rows)
            parents = population(rows, :);
            I = repmat(randi(2, numel(rows), 1), 1, D);
            p1 = parents + rand(numel(rows), D) .* (population(far_idx(rows), :) - I .* parents);
            p1 = repair_bounds(p1, parents, lb, ub, boundary_mode);
            f1 = SOP_cec_evaluate(p1, problem);
            eval_count = eval_count + numel(f1);
            improved = f1 <= fitness(rows);
            if any(improved)
                population(rows(improved), :) = p1(improved, :);
                fitness(rows(improved)) = f1(improved);
            end
        end

        rows = trim_rows((1:NP)', max_fes - eval_count);
        if ~isempty(rows) && toc(t_start) < max_runtime_sec
            parents = population(rows, :);
            I = repmat(randi(2, numel(rows), 1), 1, D);
            p2 = parents + rand(numel(rows), D) .* (population(near_idx(rows), :) - I .* parents);
            p2 = repair_bounds(p2, parents, lb, ub, boundary_mode);
            f2 = SOP_cec_evaluate(p2, problem);
            eval_count = eval_count + numel(f2);
            improved = f2 <= fitness(rows);
            if any(improved)
                population(rows(improved), :) = p2(improved, :);
                fitness(rows(improved)) = f2(improved);
            end
        end

        [current_best, current_idx] = min(fitness);
        if current_best < best_raw
            best_raw = current_best;
            best_position = population(current_idx, :);
        end
    elseif update_mode == "sequential" || batch_size <= 1
        for i = 1:NP
            [far_idx, near_idx] = far_near_for_member(population, i);

            if eval_count < max_fes
                parent = population(i, :);
                p1 = parent + rand(1, D) .* (population(far_idx, :) - randi(2) .* parent);
                p1 = repair_bounds(p1, parent, lb, ub, boundary_mode);
            f1 = SOP_cec_evaluate(p1, problem);
            eval_count = eval_count + 1;
            if toc(t_start) >= max_runtime_sec
                break;
            end
                if f1 <= fitness(i)
                    population(i, :) = p1;
                    fitness(i) = f1;
                end
            end

            if eval_count < max_fes
                parent = population(i, :);
                p2 = parent + rand(1, D) .* (population(near_idx, :) - randi(2) .* parent);
                p2 = repair_bounds(p2, parent, lb, ub, boundary_mode);
            f2 = SOP_cec_evaluate(p2, problem);
            eval_count = eval_count + 1;
            if toc(t_start) >= max_runtime_sec
                break;
            end
                if f2 <= fitness(i)
                    population(i, :) = p2;
                    fitness(i) = f2;
                end
            end

            if fitness(i) < best_raw
                best_raw = fitness(i);
                best_position = population(i, :);
            end

            if eval_count >= max_fes
                break;
            end
        end
    else
        block = max(1, min(NP, round(batch_size)));
        for first = 1:block:NP
            rows = first:min(NP, first + block - 1);
            if eval_count < max_fes
                rows = trim_rows(rows, max_fes - eval_count);
                [far_idx, ~] = far_near_indices(population);
                parents = population(rows, :);
                I = repmat(randi(2, numel(rows), 1), 1, D);
                p1 = parents + rand(numel(rows), D) .* (population(far_idx(rows), :) - I .* parents);
                p1 = repair_bounds(p1, parents, lb, ub, boundary_mode);
                f1 = SOP_cec_evaluate(p1, problem);
                eval_count = eval_count + numel(f1);
                if toc(t_start) >= max_runtime_sec
                    break;
                end
                improved = f1 <= fitness(rows);
                if any(improved)
                    population(rows(improved), :) = p1(improved, :);
                    fitness(rows(improved)) = f1(improved);
                end
            end

            if eval_count < max_fes
                rows = first:min(NP, first + block - 1);
                rows = trim_rows(rows, max_fes - eval_count);
                [~, near_idx] = far_near_indices(population);
                parents = population(rows, :);
                I = repmat(randi(2, numel(rows), 1), 1, D);
                p2 = parents + rand(numel(rows), D) .* (population(near_idx(rows), :) - I .* parents);
                p2 = repair_bounds(p2, parents, lb, ub, boundary_mode);
                f2 = SOP_cec_evaluate(p2, problem);
                eval_count = eval_count + numel(f2);
                if toc(t_start) >= max_runtime_sec
                    break;
                end
                improved = f2 <= fitness(rows);
                if any(improved)
                    population(rows(improved), :) = p2(improved, :);
                    fitness(rows(improved)) = f2(improved);
                end
            end

            [current_best, current_idx] = min(fitness);
            if current_best < best_raw
                best_raw = current_best;
                best_position = population(current_idx, :);
            end

            if eval_count >= max_fes
                break;
            end
        end
    end

    convergence_raw(iter) = best_raw;
    if eval_count >= max_fes
        convergence_raw = convergence_raw(1:iter);
        break;
    end
end

runtime = toc(t_start);
[final_fitness, final_order] = sort(fitness(:));
final_population = population(final_order, :);
result = struct();
result.best_value = best_raw;
result.record_value = SOP_cec_record_value(best_raw, problem);
result.best_position = best_position;
result.convergence_curve = SOP_cec_record_value(convergence_raw, problem);
result.raw_convergence_curve = convergence_raw;
result.runtime = runtime;
result.iteration = numel(convergence_raw);
result.population_num = NP;
result.evaluation_count = eval_count;
result.final_population = final_population;
result.final_fitness = final_fitness;
result.algorithm_combination = sprintf('Far and Near Optimization (FNO)\nFarthest-nearest greedy population update');
result.combination_number = 1;
result.agent_id = 'Agent1';
result.problem = problem;

if verbose
    fprintf('FNO finished %s %dD F%d: best %.12g, runtime %.4f s, eval %d.\n', ...
        problem.suite, D, problem.func_num, result.record_value, runtime, eval_count);
end
end

function [far_idx, near_idx] = far_near_for_member(population, member_idx)
diff = population - population(member_idx, :);
dist2 = sum(diff .^ 2, 2);
dist2(member_idx) = -inf;
[~, far_idx] = max(dist2);
dist2(member_idx) = inf;
[~, near_idx] = min(dist2);
end

function [far_idx, near_idx] = far_near_indices(population)
norms = sum(population .^ 2, 2);
dist2 = norms + norms' - 2 * (population * population');
dist2 = max(dist2, 0);
NP = size(population, 1);
dist2(1:NP + 1:end) = -inf;
[~, far_idx] = max(dist2, [], 2);
dist2(1:NP + 1:end) = inf;
[~, near_idx] = min(dist2, [], 2);
end

function rows = trim_rows(rows, remaining)
if numel(rows) > remaining
    rows = rows(1:remaining);
end
end

function trial = repair_bounds(trial, parent, lb, ub, boundary_mode)
if boundary_mode == "clamp"
    trial = min(max(trial, lb), ub);
    return;
end
low = trial < lb;
high = trial > ub;
lb_matrix = repmat(lb, size(trial, 1), 1);
ub_matrix = repmat(ub, size(trial, 1), 1);
trial(low) = 0.5 * (parent(low) + lb_matrix(low));
trial(high) = 0.5 * (parent(high) + ub_matrix(high));
trial = min(max(trial, lb_matrix), ub_matrix);
end

function value = get_option(options, name, default_value)
if isstruct(options) && isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default_value;
end
end
