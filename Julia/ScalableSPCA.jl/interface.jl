include("core_julia1p3.jl")

# access command line args
k, save_loc, output_loc = parse(Int64, ARGS[1]), ARGS[2], ARGS[3]

# load matrix
df = DataFrame(CSV.File(save_loc, header=false))
sigma = Matrix(df)

UB, LB, xi = getSDPUpperBound_gd(sigma, k, true, false)
#output_loc = @printf("temp/output_%f", k)

# save UB, LB, xi to file
open(output_loc, "w") do io
      print(io, UB, ",",)
      println(io, LB, ",")
      # save vector as comma separated without brackets
      println(io, string(xi)[2:end-1])
end