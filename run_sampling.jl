#!/usr/bin/env julia

using PyCall

pushfirst!(PyVector(pyimport("sys")."path"), "/sdf/home/m/mati/SampleHybridPriorWavletFlow")
sample_model = pyimport("HybridFlow.sample_model")
sample_obj = sample_model.sample_HF(config="/sdf/home/m/mati/SampleHybridPriorWavletFlow/configs/saved_model_config.py", verbose=true)

# First sampling (16 samples)
println("Sampling 16 samples:")
@time begin
    samples = sample_obj.sample(sample_number=16, verbose=false)
end
println("Type of samples: ", typeof(samples))

# Second sampling (1200 samples)
println("\nSampling 1200 samples:")
@time begin
    samples = sample_obj.sample(sample_number=1200, verbose=false)
end
println("Type of samples: ", typeof(samples))
