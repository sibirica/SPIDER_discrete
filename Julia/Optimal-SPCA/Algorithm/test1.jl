using Test
using JLD
mutable struct problem
    data::Array{Float64}
    Sigma::Array{Float64}
end
###############################################################################
### We have hard-coded the test data here, in order to avoid Julia dependency issues
###############################################################################
include(normpath(joinpath(@__FILE__,"..",".."))*"Algorithm/branchAndBound.jl")
include(normpath(joinpath(@__FILE__,"..",".."))*"Algorithm/utilities.jl")
include(normpath(joinpath(@__FILE__,"..",".."))*"Algorithm/multiComponents.jl")

################################################################################
######## Test #1: Pitprops data set
################################################################################
@testset "Berk and Bertsimas SPCA Tables 6-7" begin

pitprops=[[1,0.954,0.364,0.342,-0.129,0.313,0.496,0.424,0.592,0.545,0.084,-0.019,0.134];
       [0.954,1,0.297,0.284,-0.118,0.291,0.503,0.419,0.648,0.569,0.076,-0.036,0.144];
       [0.364,0.297,1,0.882,-0.148,0.153,-0.029,-0.054,0.125,-0.081,0.162,0.22,0.126];
       [0.342,0.284,0.882,1,0.22,0.381,0.174,-0.059,0.137,-0.014,0.097,0.169,0.015];
       [-0.129,-0.118,-0.148,0.22,1,0.364,0.296,0.004,-0.039,0.037,-0.091,-0.145,-0.208];
       [0.313,0.291,0.153,0.381,0.364,1,0.813,0.09,0.211,0.274,-0.036,0.024,-0.329];
       [0.496,0.503,-0.029,0.174,0.296,0.813,1,0.372,0.465,0.679,-0.113,-0.232,-0.424];
       [0.424,0.419,-0.054,-0.059,0.004,0.09,0.372,1,0.482,0.557,0.061,-0.357,-0.202];
       [0.592,0.648,0.125,0.137,-0.039,0.211,0.465,0.482,1,0.526,0.085,-0.127,-0.076];
       [0.545,0.569,-0.081,-0.014,0.037,0.274,0.679,0.557,0.526,1,-0.319,-0.368,-0.291];
       [0.084,0.076,0.162,0.097,-0.091,-0.036,-0.113,0.061,0.085,-0.319,1,0.029,0.007];
       [-0.019,-0.036,0.22,0.169,-0.145,0.024,-0.232,-0.357,-0.127,-0.368,0.029,1,0.184];
       [0.134,0.144,0.126,0.015,-0.208,-0.329,-0.424,-0.202,-0.076,-0.291,0.007,0.184,1];]
pitprops=reshape(pitprops, (13,13));
B=sqrt(pitprops);
theProb=problem(B, pitprops)
println("Running pitprops data set k=5")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 5)
@show explored, timetoConverge
println("Testing pitprops data set k=5")
@show @test(abs(obj-3.40615)<1e-4)
println("Running pitprops data set k=10")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 10)
@show explored, timetoConverge
println("Testing pitprops data set k=10")
@show @test(abs(obj-4.17264)<1e-4)

################################################################################
#### Clear data
################################################################################
theProb=nothing;
pitprops=nothing;
B=nothing;
################################################################################
######## Test #2: Wine data set.
###############################################################################
wine=[[ 0.659062    , 0.0856113  , 0.0471152    ,  -0.841093   ,  3.13988   , 0.146887   ,  0.192033   , -0.0157543   , 0.0635175   ,  1.02828    ,-0.0133134   , 0.0416978   ,   164.567		];
[   0.0856113  ,  1.24802   ,  0.050277    ,    1.07633   ,  -0.87078  , -0.234338  ,  -0.45863   ,   0.0407334  , -0.141147   ,   0.644838  , -0.143326   , -0.292447   ,    -67.5489   ];
[   0.0471152  ,  0.050277  ,  0.0752646   ,    0.406208  ,   1.12294  ,  0.0221456 ,   0.0315347 ,   0.00635847 ,  0.00151558 ,   0.164654  , -0.00468215 ,  0.000761836,     19.3197   ];
[  -0.841093   ,  1.07633   ,  0.406208    ,   11.1527    ,  -3.97476  , -0.671149  ,  -1.17208   ,   0.150422   , -0.377176   ,   0.145024  , -0.209118   , -0.656234   ,   -463.355    ];
[   3.13988    , -0.87078   ,  1.12294     ,   -3.97476   , 203.989    ,  1.91647   ,   2.79309   ,  -0.455563   ,  1.93283    ,   6.62052   ,  0.180851   ,  0.669308   ,   1769.16     ];
[   0.146887   , -0.234338  ,  0.0221456   ,   -0.671149  ,   1.91647  ,  0.39169   ,   0.54047   ,  -0.0350451  ,  0.219373   ,  -0.0799975 ,  0.0620389  ,  0.311021   ,     98.1711   ];
[   0.192033   , -0.45863   ,  0.0315347   ,   -1.17208   ,   2.79309  ,  0.54047   ,   0.997719  ,  -0.066867   ,  0.373148   ,  -0.399169  ,  0.124082   ,  0.558262   ,    155.447    ];
[  -0.0157543  ,  0.0407334 ,  0.00635847  ,    0.150422  ,  -0.455563 , -0.0350451 ,  -0.066867  ,   0.0154886  , -0.0260599  ,   0.0401205 , -0.00747118 , -0.0444692  ,    -12.2036   ];
[   0.0635175  , -0.141147  ,  0.00151558  ,   -0.377176  ,   1.93283  ,  0.219373  ,   0.373148  ,  -0.0260599  ,  0.327595   ,  -0.0335039 ,  0.0386646  ,  0.210933   ,     59.5543   ];
[   1.02828    ,  0.644838  ,  0.164654    ,    0.145024  ,   6.62052  , -0.0799975 ,  -0.399169  ,   0.0401205  , -0.0335039  ,   5.37445   , -0.276506   , -0.705813   ,    230.767    ];
[  -0.0133134  , -0.143326  , -0.00468215  ,   -0.209118  ,   0.180851 ,  0.0620389 ,   0.124082  ,  -0.00747118 ,  0.0386646  ,  -0.276506  ,  0.052245   ,  0.0917662  ,     17.0002   ];
[   0.0416978  , -0.292447  ,  0.000761836 ,   -0.656234  ,   0.669308 ,  0.311021  ,   0.558262  ,  -0.0444692  ,  0.210933   ,  -0.705813  ,  0.0917662  ,  0.504086   ,     69.9275   ];
[ 164.567      ,-67.5489    , 19.3197      , -463.355     ,1769.16     , 98.1711    , 155.447     , -12.2036     , 59.5543     , 230.767     , 17.0002     , 69.9275     ,  99166.7      ];]
wine=reshape(wine, (13,13))
B=sqrt(wine);
theProb=problem(B, wine)
println("Running wine data set k=5")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 5)

println("Testing wine data set k=5")
@show @test(abs(obj-99201.31)<1.0)


println("Running wine data set k=10")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 10)

println("Testing wine data set k=10")
@show @test(abs(obj-99201.78)<1.0)
################################################################################
#### Clear data
################################################################################
theProb=nothing;
wine=nothing;
B=nothing;
################################################################################
######## Test #3: Norm wine data set.
###############################################################################

normwine=[[1.0       ,  0.0943969 ,  0.211545   , -0.310235  ,  0.270798  ,  0.289101 ,   0.236815 , -0.155929 ,  0.136698    , 0.546364   ,-0.0717472 ,  0.0723432 ,   0.64372   ];
[0.0943969 ,  1.0       ,  0.164045   ,  0.2885    , -0.0545751 , -0.335167 ,  -0.411007 ,  0.292977 , -0.220746    , 0.248985   ,-0.561296  , -0.36871   ,  -0.192011  ];
[0.211545  ,  0.164045  ,  1.0        ,  0.443367  ,  0.286587  ,  0.12898  ,   0.115077 ,  0.18623  ,  0.00965194  , 0.258887   ,-0.0746669 ,  0.00391123,   0.223626  ];
[-0.310235 ,   0.2885   ,   0.443367  ,   1.0      ,  -0.0833331,  -0.321113,   -0.35137 ,   0.361922,  -0.197327   ,  0.018732  , -0.273955 ,  -0.276769 ,   -0.440597 ];
[0.270798  , -0.0545751 ,  0.286587   , -0.0833331 ,  1.0       ,  0.214401 ,   0.195784 , -0.256294 ,  0.236441    , 0.19995    , 0.0553982 ,  0.0660039 ,   0.393351  ];
[0.289101  , -0.335167  ,  0.12898    , -0.321113  ,  0.214401  ,  1.0      ,   0.864564 , -0.449935 ,  0.612413    ,-0.0551364  , 0.433681  ,  0.699949  ,   0.498115  ];
[0.236815  , -0.411007  ,  0.115077   , -0.35137   ,  0.195784  ,  0.864564 ,   1.0      , -0.5379   ,  0.652692    ,-0.172379   , 0.543479  ,  0.787194  ,   0.494193  ];
[-0.155929 ,   0.292977 ,   0.18623   ,   0.361922 ,  -0.256294 ,  -0.449935,   -0.5379  ,   1.0     ,  -0.365845   ,  0.139057  , -0.26264  ,  -0.50327  ,   -0.311385 ];
[0.136698  , -0.220746  ,  0.00965194 , -0.197327  ,  0.236441  ,  0.612413 ,   0.652692 , -0.365845 ,  1.0         ,-0.0252499  , 0.295544  ,  0.519067  ,   0.330417  ];
[0.546364  ,  0.248985  ,  0.258887   ,  0.018732  ,  0.19995   , -0.0551364,  -0.172379 ,  0.139057 , -0.0252499   , 1.0        ,-0.521813  , -0.428815  ,   0.3161    ];
[-0.0717472,  -0.561296 ,  -0.0746669 ,  -0.273955 ,   0.0553982,   0.433681,    0.543479,  -0.26264 ,   0.295544   , -0.521813  ,  1.0      ,   0.565468 ,    0.236183 ];
[0.0723432 , -0.36871   ,  0.00391123 , -0.276769  ,  0.0660039 ,  0.699949 ,   0.787194 , -0.50327  ,  0.519067    ,-0.428815   , 0.565468  ,  1.0       ,   0.312761  ];
[0.64372   , -0.192011  ,  0.223626   , -0.440597  ,  0.393351  ,  0.498115 ,   0.494193 , -0.311385 ,  0.330417    , 0.3161     , 0.236183  ,  0.312761  ,   1.0	];]
normwine=reshape(normwine, (13,13))
B=sqrt(normwine);
theProb=problem(B, normwine)
println("Running normwine data set k=5")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 5)

println("Testing wine data set k=5")
@show @test(abs(obj-3.43978)<1e-4)

println("Running normwine data set k=10")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 10)

println("Testing wine data set k=10")
@show @test(abs(obj-4.59429)<1e-4)
################################################################################
#### Clear data
################################################################################
theProb=nothing;
normwine=nothing;
B=nothing;


################################################################################
######## Test #4: miniBooNE data set.
###############################################################################
miniboone=load("data/miniBoone.jld",  "miniBooNE")
B=sqrt(miniboone)
theProb=problem(B, miniboone)
println("Running miniBooNE data set k=5")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 5)

println("Testing miniBooNE data set k=5")
@show @test(abs(obj-1.96827e9)<1e7) # source: Berk+B paper
                                                    # Large numbers so numerical precision won't be great, hence tolerance is loose.
println("Running miniBooNE data set k=10")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 10)

println("Testing miniBooNE data set k=10")
@show @test(abs(obj-1.96827e9)<1e7)
################################################################################
#### Clear data
################################################################################
spca_miniboone=nothing;
miniboone=nothing;
B=nothing;

################################################################################
######## Test #4: normMiniBooNE data set.
###############################################################################
normminiboone=load("data/miniBoone.jld",  "normMiniBooNE")
B=sqrt(normminiboone)
theProb=problem(B, normminiboone)
println("Running normMiniBooNE data set k=5")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 5)

println("Testing normMiniBooNE data set k=5")
@show @test(abs(obj-5.0000)<1e-2)

println("Running normMiniBooNE data set k=10")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 10)

println("Testing normMiniBooNE data set k=10")
@show @test(abs(obj-9.99999)<1e-2)
################################################################################
#### Clear data
################################################################################
theProb=nothing;
normminiboone=nothing;
B=nothing;
################################################################################
######## Test #6: Communities data set.
###############################################################################

communities=load("data/communities.jld",  "communities");
B=sqrt(communities);
theProb=problem(B, communities)

println("Running communities data set k=5");
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 5)
@show explored, timetoConverge
println("Testing communities data set k=5")
@show @test(abs(obj-0.2771)<1e-2)

println("Running communities data set k=10")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 10)
@show explored, timetoConverge
println("Testing communities data set k=10")
@show @test(abs(obj-0.4452)<1e-2)

################################################################################
#### Clear data
################################################################################
theProb=nothing;
communities=nothing;
B=nothing;
################################################################################
######## Test #7: Norm Communities data set.
###############################################################################
normcommunities=load("data/communities.jld", "normCommunities")
B=sqrt(normcommunities);
theProb=problem(B, normcommunities)
println("Running norm communities data set k=5")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 5)
@show explored, timetoConverge
println("Testing norm communities data set k=5")

@show @test(abs(obj-4.86051)<1e-2)

println("Running norm communities data set k=10")
obj, xVal, timetoBound, timetoConverge, timeOut, explored, toPrint=branchAndBound(theProb, 10)
@show explored, timetoConverge
println("Testing norm communities data set k=10")
@show @test(abs(obj-8.8236)<1e-2)
################################################################################
#### Clear data
################################################################################
spca_normcommunities=nothing;
normcommunities=nothing;
B=nothing;
end;
################################################################################
##### End test
################################################################################
