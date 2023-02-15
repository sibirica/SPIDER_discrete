using Test
using JLD

include("core_julia1p3.jl")
###############################################################################
### This test script aims to verify correctness of our code,
### by rerunning the results reported in Table 2 of our paper.
### Note that we obtained the "optimal" value of these problems by running both
### the code of Berk+Bertsimas and our outer-approximation method, and verifying
### that we obtained the same optimal value from both methods.
### We have hard-coded some of the smaller test instances here, for simplicity.
###############################################################################

################################################################################
######## Test #1: Pitprops data set
################################################################################
@testset "Table 2" begin

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

println("Running pitprops data set k=5 with Problem (18)")

k=5
lambda_true=3.406
@time UB, LB=getSDPUpperBound_gd(pitprops, k, false, true)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

println("Running pitprops data set k=5 with Problem (19)")

@time UB, LB=getSDPUpperBound_d2008_gd(pitprops, k, false)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

k=10
println("Running pitprops data set k=10 with Problem (18)")
lambda_true=4.173
@time UB, LB=getSDPUpperBound_gd(pitprops, k, false, true)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

println("Running pitprops data set k=10 with Problem (19)")

@time UB, LB=getSDPUpperBound_d2008_gd(pitprops, k, false)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

# ################################################################################
# #### Clear data
# ################################################################################
pitprops=nothing;
#
# ###############################################################################
# ####### Test #2: Norm wine data set.
# ##############################################################################
#
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

println("Running wine data set k=5 with Problem (18)")

k=5
lambda_true=3.43978
@time UB, LB=getSDPUpperBound_gd(normwine, k, false, true)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

println("Running wine data set k=5 with Problem (19)")

@time UB, LB=getSDPUpperBound_d2008_gd(normwine, k, false)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

k=10
println("Running wine data set k=10 with Problem (18)")
lambda_true=4.59429
@time UB, LB=getSDPUpperBound_gd(normwine, k, false, true)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

println("Running wine data set k=10 with Problem (19)")

@time UB, LB=getSDPUpperBound_d2008_gd(normwine, k, false)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

# # ################################################################################
# # #### Clear data
# # ################################################################################
normwine=nothing;


################################################################################
######## Test #3: normMiniBooNE data set.
###############################################################################
normminiboone=load("data/miniBoone.jld",  "normMiniBooNE")

println("Running normMiniBooNE data set k=5 with Problem (18)")

k=5
lambda_true=5.0000
@time UB, LB=getSDPUpperBound_gd(normminiboone, k, false, true)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

println("Running normMiniBooNE data set k=5 with Problem (19)")

@time UB, LB=getSDPUpperBound_d2008_gd(normminiboone, k, false)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

k=10
println("Running normMiniBooNE data set k=10 with Problem (18)")
lambda_true=9.9999999
@time UB, LB=getSDPUpperBound_gd(normminiboone, k, false, true)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB

println("Running normMiniBooNE data set k=10 with Problem (19)")

@time UB, LB=getSDPUpperBound_d2008_gd(normminiboone, k, false)
@show "R-gap is:" (UB-lambda_true)/lambda_true
@show "O-gap is:" (lambda_true-LB)/LB


end
