Test for a sparse-dense recurrent network 
where we compute a sparse representation from the input+state (concatenated, standard RNN-like) and then we compute an update to the recurrent state applying a linear layer to the sparse representation.
The recurrent state is than updated by leaky-accumulating the output produced by the sparse representation through th elinear layer.
(could be substituted by gating as well, instead of leaky accumulate)