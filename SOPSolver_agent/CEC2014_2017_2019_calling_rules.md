# CEC2014 / CEC2017 / CEC2019 Calling Rules

This document provides a concise reference for future conversations or code development, explaining how to call the locally organized CEC2014, CEC2017, and CEC2019 benchmark functions.

Local root directory: *(provide the local root directory path)*

```matlab
ROOT = 'provide your local root';
```

## 1. General Principles

Many `.m` files in the local folders are only **interfaces** or **wrappers**. This is not a missing-file issue. The official CEC MATLAB packages commonly follow the structure below:

- MATLAB entry `.m` files
- C/C++ MEX core files, such as `cec14_func.cpp`
- Compiled MEX files, such as `cec14_func.mexw64`
- Shift / rotation / shuffle data stored in `input_data`

Therefore, it is **not recommended** to manually write or modify the mathematical formulas. Subsequent algorithms only need to call the already organized MATLAB entry functions.

It is recommended to call the outer wrappers first:

- `CEC2014_F01(x)` to `CEC2014_F30(x)`
- `CEC2017_F01(x)`, `CEC2017_F03(x)` to `CEC2017_F30(x)`
- `CEC2019_F01(x)` to `CEC2019_F10(x)`

## 2. Path Initialization for CEC2014 / CEC2017 / CEC2019

Each time a new MATLAB session is started, add the required paths first:

```matlab
addpath('...\CEC2014');
addpath('...\CEC2017');
addpath('...\CEC2019');
```

If a `MissingMex` error occurs, or if MATLAB reports that `cecxx_func.mexw64` cannot be found, run the corresponding compilation scripts:

```matlab
CEC2014_compile_mex;
CEC2017_compile_mex;
CEC2019_compile_mex;
```

The current local verification status is as follows: CEC2014 has been successfully compiled into a 64-bit MEX file; CEC2017 and CEC2019 already contain `mexw64` files.

## 3. CEC Input Format Rules

The following two input formats are recommended:

```matlab
x = zeros(1, D);       % A single candidate solution, 1 × D
X = rand(N, D);        % N candidate solutions, N × D
```

The local `CEC2014_evaluate.m`, `CEC2017_evaluate.m`, and `CEC2019_evaluate.m` files have already been adapted for compatibility:

- A `1 × D` row vector can be passed directly.
- A `D × 1` column vector can be passed directly.
- An `N × D` population matrix can be passed directly.
- The wrapper automatically converts the input into the `D × N` format required by the official MEX core.

If the wrapper is bypassed and the official MEX function is called directly, for example `cec17_func(x, func_num)`, note that the official interface usually requires the `D × N` format, meaning that each column represents one candidate solution.

## 4. CEC2014 Calling Rules

Path: *(provide the corresponding local storage path)*

Core official source code: *(provide the corresponding local storage path)*

Function numbers:

```matlab
1:30
```

Available dimensions:

```matlab
D = 2, 10, 20, 30, 50, 100
```

Note: The official source code indicates that when `D=2`, some hybrid functions and composition functions are unavailable, especially `F17-F22` and `F29-F30`. Conventional paper experiments usually use `D=10, 30, 50, 100`.

Single-function call:

```matlab
x = zeros(1, 10);
f = CEC2014_F01(x);
```

General dispatcher call:

```matlab
func_num = 10;
x = zeros(1, 30);
f = CEC2014_evaluate(x, func_num);
```

Population call:

```matlab
N = 50;
D = 30;
X = rand(N, D) * 200 - 100;    % Common CEC2014 search range: [-100, 100]^D
f = CEC2014_F10(X);            % Return N fitness values
```

Wrapping the objective function inside an algorithm:

```matlab
func_num = 10;
obj = @(x) CEC2014_evaluate(x, func_num);

x = rand(1, 30) * 200 - 100;
fitness = obj(x);
```

## 5. CEC2017 Calling Rules

Path: *(provide the corresponding local storage path)*

Core official source code: *(provide the corresponding local storage path)*

Function numbers:

```matlab
F01, F03-F30
```

The official CEC2017 benchmark removed `F02`, so it is normal that there is no local `CEC2017_F02.m` file.

Available dimensions:

```matlab
D = 2, 10, 20, 30, 50, 100
```

Similarly, paper experiments are recommended to prioritize `D=10, 30, 50, 100`.

Single-function call:

```matlab
x = zeros(1, 10);
f = CEC2017_F01(x);
```

General dispatcher call:

```matlab
func_num = 3;
x = zeros(1, 30);
f = CEC2017_evaluate(x, func_num);
```

Population call:

```matlab
N = 50;
D = 30;
X = rand(N, D) * 200 - 100;
f = CEC2017_F03(X);
```

When iterating through all CEC2017 functions, do not include function 2:

```matlab
func_list = [1 3:30];
D = 30;
N = 50;

for func_num = func_list
    X = rand(N, D) * 200 - 100;
    f = CEC2017_evaluate(X, func_num);
end
```

## 6. CEC2019 Calling Rules

Path: *(provide the corresponding local storage path)*

Core official source code: *(provide the corresponding local storage path)*

Function numbers:

```matlab
1:10
```

CEC2019 does not use a unified dimension. The dimension must be set according to the function number:

| Function | Dimension |
| --- | ---: |
| F01 | 9 |
| F02 | 16 |
| F03 | 18 |
| F04-F10 | 10 |

Calling examples:

```matlab
f1 = CEC2019_F01(zeros(1, 9));
f2 = CEC2019_F02(zeros(1, 16));
f3 = CEC2019_F03(zeros(1, 18));
f4 = CEC2019_F04(zeros(1, 10));
```

General dispatcher call:

```matlab
func_num = 4;
x = zeros(1, 10);
f = CEC2019_evaluate(x, func_num);
```

When iterating through CEC2019 functions, the following structure is recommended:

```matlab
for func_num = 1:10
    if func_num == 1
        D = 9;
    elseif func_num == 2
        D = 16;
    elseif func_num == 3
        D = 18;
    else
        D = 10;
    end

    X = zeros(20, D);
    f = CEC2019_evaluate(X, func_num);
end
```

## 7. Recommended Unified Interface for Algorithms

For unconstrained CEC functions:

```matlab
obj = @(x) CEC2017_evaluate(x, 3);
fitness = obj(x);
```

If the algorithm internally supports only single-objective unconstrained values, define the constraint-handling strategy separately, such as a penalty function:

```matlab
penalty_weight = 1e8;
tol = 1e-8;

bench = @PB_RWCO20_F18_PressureVesselDesign;

obj_penalty = @(x) local_penalty_obj(bench, x, penalty_weight, tol);

function val = local_penalty_obj(bench, x, penalty_weight, tol)
    [f, g, h] = bench(x);
    g_violation = sum(max(0, g), 2);
    h_violation = sum(max(0, abs(h) - tol), 2);
    val = f + penalty_weight .* (g_violation + h_violation);
end
```

## 8. Common Errors and Handling Methods

### `MissingMex`

```matlab
CEC2014_compile_mex;
CEC2017_compile_mex;
CEC2019_compile_mex;
```

### Dimension Errors

- CEC2014 / CEC2017 only accept `2, 10, 20, 30, 50, 100`.
- CEC2019 must use F01=9, F02=16, F03=18, and F04-F10=10.
- CEC2017 does not contain F02.

### Path Errors

```matlab
addpath('local path');
addpath('local path');
addpath('local path');
```

### ENOPPY Python Errors

- Preferably, do not use ENOPPY for the main MATLAB experiments.
- If ENOPPY must be used, first configure a MATLAB-supported Python version and confirm that `numpy` is available.
- MATLAB R2022a does not support Python 3.12. This is an environment compatibility issue, not a missing benchmark source-code issue.

## 9. Minimum Runnable Check Script

```matlab
addpath('local path');
addpath('local path');
addpath('local path');
addpath('local path');

disp(CEC2014_F01(zeros(1,10)));
disp(CEC2017_F01(zeros(1,10)));
disp(CEC2019_F01(zeros(1,9)));
disp(CEC2019_F04(zeros(1,10)));
```

## 10. Authenticity Statement

- The core calculations of CEC2014 / CEC2017 / CEC2019 call the downloaded official source code and data files; the formulas are not manually rewritten.
