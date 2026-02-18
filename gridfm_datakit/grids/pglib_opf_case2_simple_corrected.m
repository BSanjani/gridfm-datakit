%% MATPOWER Case Format : Version 2
function mpc = pglib_opf_case2_simple
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100.0;

%% bus data
%    bus_i    type    Pd    Qd    Gs    Bs    area    Vm    Va    baseKV    zone    Vmax    Vmin
mpc.bus = [
    1    3    0.0    0.0    0.0    0.0    1    1.0    0.0    138.0    1    1.10    0.90;
    2    1    100.0    30.0    0.0    0.0    1    1.0    0.0    138.0    1    1.10    0.90;
];

%% generator data
%    bus    Pg    Qg    Qmax    Qmin    Vg    mBase    status    Pmax    Pmin
mpc.gen = [
    1    110.0    30.0    300.0    -300.0    1.0    100.0    1    300.0    0.0;
];

%% branch data
%    f_bus    t_bus    r    x    b    rateA    rateB    rateC    ratio    angle    status    angmin    angmax
mpc.branch = [
    1    2    0.01938    0.05917    0.0528    472.0    472.0    472.0    0    0    1    -360.0    360.0;
];

%%-----  OPF Data  -----%%
%% cost data
%    2    startup    shutdown    n    c(n-1)    ...    c0
mpc.gencost = [
    2    0.0    0.0    3    0.01    20.0    0.0;
];
