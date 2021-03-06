import dca2pdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("talk")

####################################################################################################
#reproduce results Dib et al., PloS-Compt Biol, 2018
#download data using Biodownloader from terminal
#BioDownloader pfam --pfam PF00129 (in stockholm)



####################################################################################################
#ANALYSIS OF A PDB
#pdb = './dib_plos/2bnr.trunc.fit.pdb'
path_pdb = './2bnr.pdb'
pdb = dca2pdb.load_pdb(path_pdb)

#get chain->pdb_seq,pdb_idx
d_chain_seq, d_chain_idx = dca2pdb.get_pdb_sequences(pdb)

d_intrachain_dist  = dca2pdb.intrachain_dist(pdb, chain = "A", distance_type = "min")

m_intrachain = dca2pdb.make_intrachain_matrix(d_intrachain_dist)

plt.imshow(m_intrachain)

dca2pdb.plot_cm_interchain(d_intrachain_dist, cut_off_dist = 5)

d_interchain_dist  = dca2pdb.interchain_dist(pdb, chain1 = "A", chain2 = "D",  distance_type = "min")
m_interchain = dca2pdb.make_interchain_matrix(d_interchain_dist)
plt.imshow(m_interchain)
dca2pdb.plot_cm_interchain(d_interchain_dist, cut_off_dist = 5)


####################################################################################################
#ANALYSIS OF THE MSA and mapping MSA-PDB

path_msa = './dib_plos/PF00129_EUKARIOTS.txt'

#load dictionary id->sequence
d_id_seq = dca2pdb.get_seq_from_msa(path_msa, remove_dot_lower= True)
#rm gapped seq
#d_id_seq = dca2pdb.trim_gapped_seq(d_id_seq, max_gap_fraction_seq = 0.5)
#rm gapped col
#d_id_seq = dca2pdb.trim_gapped_col(d_id_seq, max_gap_fraction= 15)

path_trimmed_msa = './dib_plos/PF00129_EUKARIOTS.txt'
#dca2pdb.save_trimmed_msa(d_id_seq, path_trimmed_msa)

#seq_test
seq = d_id_seq[list(d_id_seq.keys())[0]]
dca2pdb.all_standard_aa(seq)
consensus_seq = dca2pdb.get_consensus(d_id_seq)
dca2pdb.compute_seqid(consensus_seq,seq)

#>A0A1W2PR11_HUMAN/25-203
ref_seq = "GSHSMRYFDTAVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQADRVNLRKLRGYYNQSEDGSHTLQWMYGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQWRAYLEGTCVEWLRRYLENGKETL"
fi = dca2pdb.compute_freq(d_id_seq, pseudocount = 0)
S_ci = dca2pdb.compute_entropy_context_ind(d_id_seq, pseudocount = 0, remove_gaps = True, base2 = True)

#run dca computation
#h, J, fn_dca = dca2pdb.run_dca(path_trimmed_msa)
h, J, fn_dca  = dca2pdb.load_obj('h_J_fn_dca_vertebrate.pkl')
S_cd = dca2pdb.compute_entropy_context_dep(ref_seq, h,J )
plt.plot(S_ci)
plt.plot(S_cd)
plt.scatter(S_ci, S_cd)
plt.xlabel("S_ci")
plt.ylabel("S_cd")


############
#map pdb to msa
chain_pdb = "A"
path_hmm = './dib_plos/MHC_I.hmm'
df_pdb_msa = dca2pdb.map_pdb_hmm(path_pdb, chain_pdb,  path_hmm)

df_dca_pdb = dca2pdb.match_dca_pdb(d_intrachain_dist, fn_dca, df_pdb_msa)

#keep only distant contact
min_distance_msa = 4
df_distant = df_dca_pdb.loc[df_dca_pdb['idx_msa2'] - df_dca_pdb['idx_msa1']>min_distance_msa]
df_distant = df_distant.sort_values(by = 'fn_dca', ascending = False)

#plot PPV
dist_tp = 8 #below dist_tp is tp=1 (true contact)
tp = (df_distant['dist_pdb']< dist_tp ) + 0
num_pred = (range(1, (len(tp)+1)))
ppv = np.cumsum(tp.values)/ num_pred
plt.plot(num_pred, ppv)

plt.xlim(0,100)

