from ipymol import viewer as pymol
import dca2pdb

#0.  read msa data
path_msa = './dib_plos/PF00129_full.txt'
d_id_seq = dca2pdb.get_seq_from_msa(path_msa, format_msa = 'fasta', remove_dot_lower= True)
path_pdb = './dib_plos/2bnr.trunc.fit.pdb'
#load pdb
pdb = dca2pdb.load_pdb(path_pdb, structure_id = 'test_pdb')
#get chain->pdb_seq,pdb_idx
d_chain_seq, d_chain_idx = dca2pdb.get_pdb_sequences(pdb)
d_intrachain_dist  = dca2pdb.intrachain_dist(pdb, chain = "A", distance_type = "min")
chain_pdb = "A"
path_hmm = './dib_plos/MHC_I.hmm'
#map pdb_msa
df_pdb_msa = dca2pdb.map_pdb_hmm(path_pdb, chain_pdb,  path_hmm)
#rm gapped pos
gap_fraction, gap_idx = dca2pdb.get_gapped_col(d_id_seq, max_gap_fraction = 0.5)
df_pdb_msa = df_pdb_msa.loc[df_pdb_msa['idx_msa']!=-1]
df_pdb_msa['gap_fraction'] = gap_fraction

max_gap_fraction = 0.5
df_pdb_msa = df_pdb_msa.loc[df_pdb_msa['gap_fraction']< max_gap_fraction]


#1. pymol analysis
#get id_pdb

res_msa = df_pdb_msa['idx_pdb'].values
str_res_msa = '+'.join(str(res_msa[i]) for i in range(0, len(res_msa)))


pymol.start()
pymol.load('./dib_plos/2bnr.trunc.fit.pdb')
pymol.bg_color('black') # Set background color to white
pymol.show_as('cartoon') # Show as cartoon
pymol.select('epitope', 'chain C')
pymol.show_as('sticks', 'epitope')
pymol.select('TRA', 'chain D')
pymol.select('TRB', 'chain E')

#show domain coverage
pymol.hide('cartoon', 'TRA')
pymol.hide('cartoon', 'TRB')

#select res on a chain
pymol.select('PF00129', 'chain A and resi '+str_res_msa)

pymol.color('blue', 'PF00129')

