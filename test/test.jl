using SparseArrays
s1 = sparse([1,2],[2,3],[1.,3.])
@edit sum(s1)
#%%
using SparseArrays

C = 30
idx = rand(1:10, C)
idx2 = rand(1:10, C)
vals = rand(1.f0:10.f0, C)
r, c, v = idx,idx2,vals
s1 = sparse(r, c, v)
s2 = sparse(r, c, v)
using InteractiveUtils
# @edit sparse(r, c, v)
arr = Array(s1)
@time sum(s1,dims=1)
@time sum(arr,dims=1)

@time sum(s1)
@time sum(v)
@time sum(arr)

;
#%%
# @edit spzeros(Float32, 3)
nnz(s1)
# s1
#%%
using SparseArrayVectors: sparsev, sum2

C=1024
r, c, v = [1,2,1],[2,3,2],[1.,3.,4.]
r,c,v = rand(1:10, C), rand(1:10, C), rand(1.f0:10.f0, C)
a1 = sparsev(r, c, v)
a2 = sparsev(r, c, v)
a1 .* a2
@time sum(a1)
@time sum(a1)
@time sum2(a1)
@time sum2(a1)
#%%
using Boilerplate
#%%
@sizes a1
#%%
using Boilerplate
using SparseArrayVectors: softmax2,sparsev,sum2
C=20
M=90
s = 15
# r, c, v = [1,2,1],[2,3,2],[1.,3.,4.]
r,c,v = rand(1:s, C), rand(1:s, C), rand(1.f0:10.f0, M, C)
# r,c,v = rand(1:s, C), rand(1:s, C), rand(1.f0:10.f0, C)
# a1 = sparsev(r, c, cat(v, v, dims=2))
a1 = sparsev(r, c, v)
# @display a1.nzval
sum2(a1)
@time sum2(a1)
softmax2(a1)
@time softmax2(a1)
@display a1
@sizes a1
;
#%%
using Boilerplate
using InteractiveUtils
using SparseArrayVectors: eye

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.00

@typeof (a1)
@typeof eye(a1)
# @show eye(a1)
# @show a1
a1 .+ eye(a1)
a1 .+ eye(a1)
a1[:,:,:] .+ eye(a1)[:,:,:]
ey1 = eye(a1)[:,:,:]
ar1 = a1[:,:,:]
@btime ey1 .+ ar1
# @code_warntype Base.broadcasted(Main.:+, a1, eye(a1))
@btime a1 .+ eye(a1)
# @show (a1 .+ eye(a1)).nzval
;
# @show a1
#%%
a1[1,:,:]
# @display reshape(1:12, 3, 4)
# @show reshape([(i-1)*4+j for i in 1:3 for j in 1:4], 3, 4)
#%%
using InteractiveUtils
@code_warntype softmax2(a1)
#%%
using SparseArrays
using ZygoteExtensions: softmax_dim

a1 = sparsev(r, c, v)
# @display sparse(r,c,v[:,1])
# softmax_dim(1)(sparse(r,c,v[:,1]))
# @time softmax_dim(1)(sparse(r,c,v[:,1]))
softmax_dim(1)(Array(sparse(r,c,v[1, :])))
ar1 = a1[:,:,:]
@btime softmax_dim(3)(ar1)
@btime softmax2(a1)
@btime sum(ar1, dims=3)
@btime sum2(a1)
;
#%%
softmax_dim(1)([1,2])
#%%
using InteractiveUtils
@code_warntype sum2(a1)
#%%
using Boilerplate
arr = reshape(1:4,(2,2))
@sizes arr
@show arr[(1,2)...]
a1 = sparsev(r, c, arr)
@show a1

@time sum2(a1)
# @code_warntype sum2(a1)
# @time sum2(a1)