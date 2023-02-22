include("core_julia1p3.jl")

# access command line args
max_k, save_loc, out_template = parse(Int64, ARGS[1]), ARGS[2], ARGS[3]

# load matrix
df = DataFrame(CSV.File(save_loc, header=false))
sigma = Matrix(df)

for k in max_k:-1:1
      output_loc = replace(out_template, "@" => string(k))
      # note - probably the other way around now
      UB, LB, xi = getSDPUpperBound_gd(sigma, k, true, true)
      #UB, LB, xi = getSDPUpperBound_gd_highdim(sigma, k, true, true)

      # save UB, LB, xi to file
      open(output_loc, "w") do io
            print(io, UB, ",",)
            println(io, LB, ",")
            # save vector as comma separated without brackets
            println(io, string(xi)[2:end-1])
      end
end