module SparseArrayVectors

using ZygoteExtensions: softmax_dim
import ZygoteExtensions
using Boilerplate

export sparsev, sum2, softmax2, fill_like, eye

struct SparseArrayVectorsCSC{T, N} <: AbstractArray{T, N}
  m::Int                   # Number of rows
  n::Int                   # Number of columns
  colptr::Vector{Int}      # Column j is in colptr[j]:(colptr[j+1]-1)
  rowval::Vector{Int}      # Row indices of stored values
  nzval::Array{T, N}       # Stored values, typically nonzeros
  # colidxs::Vector{Vector{Int}}  # Store the vector of existing idxs in a column
end

greet() = print("Hello World!")
sparsev(col,row,val) = begin
  m = max(maximum(col), maximum(row))
  SparseArrayVectorsCSC(m,m,col,row,val)
end
Base.length(a::SparseArrayVectorsCSC) = length(a.nzval)
@inline nnz(a::SparseArrayVectorsCSC) = size(a.nzval,ndims(a.nzval))

Base.broadcasted(::typeof(+), a::SparseArrayVectorsCSC{T, 2}, b::SparseArrayVectorsCSC{T,2}) where {T} = begin
  # @assert all(size(a) .== size(b)) "Sizes must match."

  # b_idxs = Int[]
  # a_idxs = Int[]
  # for (i, c2) in enumerate(b.colptr)
  #   j = 1
  #   while j <= length(a.colptr) && !(c2 ===  a.colptr[j] && b.rowval[i] === a.rowval[j])
  #     j+=1
  #   end
  #   if j >= length(a.colptr)
  #     push!(b_idxs, i)
  #   else
  #     push!(a_idxs, j)
  #   end
  # end
  b_new = 0
  @inbounds for (i, c2) in enumerate(b.colptr)
    j = 1
    while j <= length(a.colptr) && !(c2 ===  a.colptr[j] && b.rowval[i] === a.rowval[j])
      j+=1
    end
    if j >= length(a.colptr)
      b_new += 1
    end
  end
  N = length(a.colptr)
  newnzval = zeros(eltype(a),(size(a.nzval)[1:end-1]..., N + b_new))  
  newcol = zeros(Int, N+ b_new)
  newrow = zeros(Int, N + b_new)
  newnzval[:, 1:nnz(a)] .= a.nzval
  collision = 1
  @inbounds for (i, c2) in enumerate(b.colptr)
    j = 1
    while j <= length(a.colptr) && !(c2 ===  a.colptr[j] && b.rowval[i] === a.rowval[j])
      j+=1
    end
    if j >= length(a.colptr)
      # push!(b_idxs, i)y
      newcol[N+collision] = b.colptr[i]
      newrow[N+collision] = b.rowval[i]
      newnzval[:, nnz(a) + collision] .= b.nzval[:, i]
      collision += 1
    else
      # push!(a_idxs, j)
      newnzval[:, j] .+= b.nzval[:, i]
    end
  end
  # @time newnzval[:, nnz(a)+1:nnz(a) + length(b_idxs)] .= b.nzval[:, b_idxs]
  # @time newnzval[:, a_idxs] .+= a.nzval[:, a_idxs]
  # append!(newcol, b.colptr[b_idxs])
  # append!(newrow, b.rowval[b_idxs])
  SparseArrayVectorsCSC(a.m, a.m, newcol, newrow, newnzval)  
end
Base.Array(x::SparseArrayVectorsCSC) = begin
  arr = zeros(Float32, size(x))
  B = size(x, 1)
  for k in 1:B
    for (l, c) in enumerate(x.colptr)
      arr[k, c, x.rowval[l]] = x.nzval[k, l]
    end
  end
  arr
end
Base.broadcasted(::typeof(*), a::SparseArrayVectorsCSC, b::SparseArrayVectorsCSC) = begin
  SparseArrayVectorsCSC(a.m, a.m, a.colptr, a.rowval, a.nzval .* b.nzval)  
end
@inline Base.broadcasted(::typeof(*), a::SparseArrayVectorsCSC, b::Number) = begin
  SparseArrayVectorsCSC(a.m, a.m, a.colptr, a.rowval, a.nzval .* b)  
end
@inline Base.broadcasted(::typeof(*), a::Number, b::SparseArrayVectorsCSC) = b .* a
@inline Base.broadcasted(::typeof(:*=), a::SparseArrayVectorsCSC, b::Number) = a.nzval .*= b
Base.sum(a::SparseArrayVectorsCSC) = sum(a.nzval)
Base.eltype(a::SparseArrayVectorsCSC) = eltype(a.nzval)
@inline Base.size(a::SparseArrayVectorsCSC, dim::Int) = dim >= ndims(a.nzval) ? a.m : size(a.nzval, dim)
Base.size(a::SparseArrayVectorsCSC{T,N}) where {T, N} = (size(a.nzval)[1:end-1]..., a.m, a.n, )
eye(x::SparseArrayVectorsCSC{T,N}) where {T, N} = begin
  sparsev(collect(1:x.m), collect(1:x.m), fill(1f0,(size(x)[1:end-2]...,x.m)))::SparseArrayVectorsCSC{T,N}
end
fill_like(x::SparseArrayVectorsCSC, value::Number) = sparsev(x.colptr,x.rowval, fill(value, size(x.nzval)))
Base.getindex(A::SparseArrayVectorsCSC{T, 2}, i::Int, j::Int) where {T} = begin
  for (ic, c) in enumerate(A.colptr)
    if c === i && A.rowval[ic] === j
      return A.nzval[1, ic]
    end
  end
  0
end
Base.getindex(A::SparseArrayVectorsCSC{T, 2}, idx::Int, i::Int, j::Int) where {T} = begin
  @inbounds for (ic, c) in enumerate(A.colptr)
    if c === i && A.rowval[ic] === j
      return A.nzval[idx, ic]
    end
  end
  0
end
# This only runs on Colon cases.
Base.getindex(x::SparseArrayVectorsCSC{T, 2}, idx, i, j) where {T} = begin
  @info "Unoptimized code. Please avoid."
  # [A.nzval[idx, ic] for (ic, c) in enumerate(A.colptr) if (c === i || i isa Colon) && (A.rowval[ic] === j || j isa Colon)]
  Array(x)[idx, i, j]
end
Base.getindex(x::SparseArrayVectorsCSC{T, 1}, i, j) where {T} = begin
  @info "Unoptimized code. Please avoid."
  arr = zeros(Float32, size(x))
  for (l, c) in enumerate(x.colptr)
    arr[k, c, x.rowval[l]] = x.nzval[k, l]
  end
  arr[i, j]
end
function sum2(x::SparseArrayVectorsCSC{T, 2}) where {T}
  o = zeros(eltype(x), size(x.nzval)[1:end-1]..., x.m, ) # TODO .m or .n?
  # @assert ndims(x.nzval)<=2 "Still only 2D is supported."
  @inbounds for i=1:nnz(x)
    for j in 1:size(x.nzval,1)
      o[j, x.rowval[i]] += x.nzval[j, i]
    end
    # o[x.rowval[i]] += x.nzval[i]
  end
  o
end
function sum2(x::SparseArrayVectorsCSC{T, 1}) where {T}
  o = zeros(eltype(x), x.m) # TODO .m or .n?
  # @assert ndims(x.nzval)<=2 "Still only 2D is supported."
  @inbounds for i=1:nnz(x)
    o[x.rowval[i]] += x.nzval[i]
  end
  o
end
softmax2(x::SparseArrayVectorsCSC{T, 2}) where {T} = begin
  # TODO filter is not really an optimal thing.
  # @sizes x.nzval
  # @sizes x.colptr
  res = zero(x.nzval)
  @inbounds for idx in 1:x.m
    idxs = [i for (i,v) in enumerate(x.colptr) if v===idx]
    if length(idxs)>0
      res[:,idxs] .= softmax_dim(2)(x.nzval[:,idxs])
    end
  end
  # signal_norm = Vector{Matrix{T}}[(x.nzval[:,]) for idx in 1:x.m]
  # @show signal_norm
  # arrs = [size(signal,1) >0 ? softmax_dim(1)(signal) : zero(signal) for signal in signal_norm]
  # sparsev(x.colptr,x.rowval, cat(arrs..., dims=2))
  # cat(arrs..., dims=1)
  sparsev(x.colptr,x.rowval,res)
end
softmax2(x::SparseArrayVectorsCSC{T, 1}) where {T} = begin
  res = zero(x.nzval)
  for idx in 1:x.m
    idxs = [i for (i,v) in enumerate(x.colptr) if v===idx]
    if length(idxs)>0
      res[idxs] .= softmax_dim(2)(x.nzval[idxs])
    end
  end
  sparsev(x.colptr,x.rowval,res)
end
Base.show(io::IO,x::SparseArrayVectorsCSC{T, 1}) where {T}= begin
  arr = x[:,:] # it reconstucts it as an array.
  show(io, arr)
end
Base.show(io::IO,x::SparseArrayVectorsCSC{T, 2}) where {T}= begin
  arr = x[:,:,:] # it reconstucts it as an array.
  show(io, arr)
end
Base.sum(x::SparseArrayVectorsCSC; dims) = begin
  @assert dims===3 || (dims isa Vector && dims[1] ===3) "Still only sum on 3rd dim supported."
  sum2(x)
end
ZygoteExtensions.softmax(x::SparseArrayVectorsCSC; dims) = begin
  @assert dims===3 || (dims isa Vector && dims[1] ===3) "Still only sum on 3rd dim supported."
  softmax2(x)
end
end # module
