using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
include("$(@__DIR__)/../ITensorCPD.jl")
using .ITensorCPD:
  als_optimize,
  direct,
  random_CPD,
  random_CPD_square_network,
  row_norm,
  reconstruct,
  had_contract

##################################################################
# refs

theor = [
  0.0,
  0.00996425164601078,
  0.020011770018868447,
  0.030031532816110484,
  0.04010680676458378,
  0.05023759186428833,
  0.06034062138837726,
  0.0705546732149287,
  0.08087974734394265,
  0.09120482147295661,
  0.10169642905566434,
  0.11224354778960333,
  0.12301271112846734,
  0.13369860774048448,
  0.14468981568427353,
  0.15584755708175635,
  0.16708856520608606,
  0.1785516179353408,
  0.1901534485426737,
  0.20200507933054723,
  0.2140787547233458,
  0.22648549702353193,
  0.23900326162618057,
  0.251909604287448,
  0.2650379915536405,
  0.27861046802968303,
  0.2925160114131131,
  0.3067823772795464,
  0.32143732120459845,
  0.3365363543395006,
  0.3521904989867153,
  0.36839975514624257,
  0.3852196339693137,
  0.40265013545592865,
  0.42088554863539684,
  0.4399536290833339,
  0.45988213237535547,
  0.4809208586920022,
  0.5034028749406616,
  0.52716164766764,
  0.5530298441414061,
  0.5812572645425007,
  0.6130651541980114,
  0.6502298699473386,
  0.7011058400507864,
  0.7563949466771192,
  0.7905620602599583,
  0.817068634972884,
  0.8388845174067683,
  0.8571476861618521,
  0.8727463196578356,
  0.8862355294070312,
  0.8981149157705204,
  0.9083844787483031,
  0.9174327963989981,
  0.9255374244787617,
  0.9327538741388253,
  0.9390821453791887,
  0.9447997939560082,
  0.9499623310205152,
  0.9544587342702471,
  0.9587330929150539,
  0.9623413177450857,
  0.9657274979701924,
  0.9688361224391429,
  0.9715561688494745,
  0.9739431483524186,
  0.9762746167041314,
  0.9782730181484567,
  0.9801603972903195,
  0.9818812429784884,
  0.9834355552129637,
  0.9847678228425139,
  0.9860445793208328,
  0.9872658246479205,
  0.9883205365213144,
  0.989319737243477,
  0.9900968933607146,
  0.9909295606291835,
  0.9917067167464211,
  0.9924838728636587,
  0.9929279620735088,
  0.9935385847370526,
  0.9940936962493652,
  0.9945932966104465,
  0.9949818746690653,
  0.9954259638789154,
  0.9958700530887654,
  0.996203119996153,
  0.9965361869035405,
  0.9967582315084655,
  0.9969802761133906,
  0.9972023207183156,
  0.9974243653232406,
  0.9976464099281657,
  0.9977574322306282,
  0.9978684545330907,
  0.9983125437429408,
  0.9984235660454033,
  0.9984235660454033,
  0.9985345883478658,
]

betas = [1.0 - i for i in 0:0.01:1]
# cp = 4.3732548624393e10 / 8.3001368444895e10

full_szsz = Vector{Float64}([])
bp_szsz = Vector{Float64}([])
using ITensorNetworks: BeliefPropagationCache, update, environment
for beta in betas
  tn = ising_network(Float64, IndsNetwork(named_grid((nx, ny))), beta; h)
  tnO = ising_network(
    Float64, IndsNetwork(named_grid((nx, ny))), beta; h, szverts=[(5, 5), (5, 6)]
  )

  using OMEinsumContractionOrders: OMEinsumContractionOrders
  using ITensorNetworks: contraction_sequence
  seq = contraction_sequence(tn; alg="sa_bipartite")
  szsz = @time contract(tnO; sequence=seq)[] / contract(tn; sequence=seq)[]
  push!(full_szsz, szsz)

  vs_centre = [(4, 4), (4, 5)]
  s = IndsNetwork(named_grid((nx, ny)); link_space=2)
  tn = ising_network(Float64, s, beta; h)
  tnO = ising_network(Float64, s, beta; h, szverts=vs_centre)
  tn_bpc = BeliefPropagationCache(tn)
  tn_bpc = update(tn_bpc; maxiter=50)
  envs = environment(tn_bpc, vs_centre)

  numer = contract([[tn[v] for v in vs_centre]; envs]; sequence="automatic")
  denom = contract([[tnO[v] for v in vs_centre]; envs]; sequence="automatic")

  szsz_bp_vbetter = denom[] / numer[]
  push!(bp_szsz, szsz_bp_vbetter)
end

#########################################################################
# plots

using Plots
plot(betas, theor[end:-1:1]; label="Infinite Lattice", s=:solid, )
plot!(betas, full_szsz; label="Exact Contraction", s=:dash,)
plot!(betas, bp_szsz; label="BP Contraction", s=:dot,)
plot!(betas, cp_szsz[1]; label="CP rank 1", s=:auto,)
plot!(betas, cp_szsz[2]; label="CP rank 6", s=:auto, )
plot!(betas, cp_szsz[3]; label="CP rank 15", s =:auto,)
plot!(;
  xlabel="Inverse Temparature",
  ylabel="SZ Correlation",
  legend=:bottomright,
  title="CPD contraction of 2D network",
)
savefig("$(@__DIR__)/../../experiment_plots/cp_ising_5x6.pdf")

# plot(betas, abs.(full_szsz .- cp_szsz[1]); label="CP rank 1")
# plot!(betas, abs.(full_szsz .- cp_szsz[2]); label="CP rank 6")
# plot!(betas, abs.(full_szsz .- cp_szsz[3]); label="CP rank 10")
# plot!(betas, abs.(full_szsz .- cp_szsz[4]); label="CP rank 15")
# plot!(betas, abs.(full_szsz .- bp_szsz); label="BP")
# plot!(; xlabel="Inverse Temparature", ylabel="SZ Correlation error")
# savefig("../CP_ising_2site_error_exact_center.pdf")