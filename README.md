# DL-NGS
An unsupervised and universal deep learning based framework for sequencing error profiling and correcting

#File format for input data
seq_len           : sequence length
gc                : GC-content
p_min             : minimum p-value of mismatch rate in this region       
err_pos           : the position sequencing error occurs
marked_err_seq    : sequences with errors marked
err_seq           : sequences with errors
matched_pos       : position of GC matched error-free sequences
matched_seq       : sequences of GC matched error-free sequences
fwd_err           : if contains errors in the forward strand
rvs_err           : if contains errors in the reverse strand   
both_err          : if contains errors in both the strands
depth_err         : average sequencing depth of error sequence
depth_cor         : average sequencing depth of error-free sequence
comment           : comment
marked_seq        : html marked error sequences
err_type          : base changes of each error position
