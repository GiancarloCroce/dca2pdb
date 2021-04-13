import pandas as pd
from Bio.PDB import *
from Bio import SeqIO
from Bio import pairwise2
from Bio import AlignIO
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import pickle
import subprocess

#aa list and dictionaries
valid_aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']
aa_3= ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR','-']
d_aa_num= {a:i for i,a in enumerate(valid_aa)}
d_num_aa= {i:a for i,a in enumerate(valid_aa)}
d_3to1 = {a3:a1 for a3,a1 in zip(aa_3,valid_aa)}

####################################################################################################
#### FUNCTIONS ####

#save and load any object with pickle
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)


#### DCA ANALYSIS ####
def run_dca(path_msa, theta = 0.2, path_julia = "/home/giancarlo/Documents/programs/julia-1.3.1/bin/julia"):
    '''run plmDCA, python wrapper to plmDCA_julia (Pagnani version )'''
    #julia.install(path_julia)
    from julia import Julia
    from julia.api import Julia
    #compiled_modules == True doesn't work.. It recompiles the module each time (slow)
    jl = Julia(runtime=path_julia, compiled_modules=False)
    from julia import PlmDCA
    dca_res = PlmDCA.plmdca(path_msa, theta = 0.2)
    J = dca_res.Jtensor
    h = dca_res.htensor
    fn_dca = dca_res.score
    return h,J, fn_dca

def match_dca_pdb(d_dist, fn_dca, df):
    '''df = map_pdb_msa // fn_dca= dca_results //d_dist = d_intrachain_dist
    return dataframe: (aa1, aa2, idx_pdb1, idx_pdb2, dist, idx_msa1, idx_msa2, dca_score) '''
    #reformat dca results
    d_fn_dca = {}
    for idx1, idx2, dca in fn_dca:
        assert idx2>idx1
        d_fn_dca[(idx1,idx2)] = dca
    #match with pdb
    all_aa1, all_aa2 = [], []
    all_idx_pdb1, all_idx_pdb2 = [], []
    all_idx_msa1, all_idx_msa2= [], []
    all_dist = []
    all_dca = []
    for keys, item in tqdm(d_dist.items()):
        idx_pdb1, idx_pdb2 = keys
        if len(df.loc[df['idx_pdb'] == idx_pdb1])>0 and len(df.loc[df['idx_pdb'] == idx_pdb2])>0:
            aa1= df.loc[df['idx_pdb'] == idx_pdb1, 'aa_pdb_msa'].values[0]
            aa2= df.loc[df['idx_pdb'] == idx_pdb2, 'aa_pdb_msa'].values[0]
            idx_msa1 = df.loc[df['idx_pdb'] == idx_pdb1, 'idx_msa'].values[0]
            idx_msa2 = df.loc[df['idx_pdb'] == idx_pdb2, 'idx_msa'].values[0]
            dist = item
            if (idx_msa1, idx_msa2) in list(d_fn_dca.keys()):
                dca_score = d_fn_dca[(idx_msa1,idx_msa2)]
                all_aa1.append(aa1)
                all_aa2.append(aa2)
                all_idx_pdb1.append(idx_pdb1)
                all_idx_pdb2.append(idx_pdb2)
                all_idx_msa1.append(idx_msa1)
                all_idx_msa2.append(idx_msa2)
                all_dist.append(dist)
                all_dca.append(dca_score)
                #print(aa1, aa2, idx_pdb1, idx_pdb2, dist, idx_msa1, idx_msa2, dca_score)
    df_results_dca = pd.DataFrame({'aa1':all_aa1, 'aa2':all_aa2, 'idx_pdb1':all_idx_pdb1, 'idx_pdb2':all_idx_pdb2, 'idx_msa1':all_idx_msa1, 'idx_msa2':all_idx_msa2, 'dist_pdb':all_dist, 'fn_dca':all_dca})
    return df_results_dca


def compute_sm_energy_dict(seq, h ,J):
    ''' for SINGLE MUTANTS, return a dictionary d['idx', 'mutated_aa'] = energy - energy_wild_type '''
    ''' it can be VERY SLOW and d_sm BIG(all possible sm ~ 21*L) '''
    ''' see below to speed it up '''
    E0 = compute_energy(seq,h,J)
    d_sm = {}
    for i in range(0, len(seq)):
        print(i, len(seq))
        #add also the gap
        for aa in valid_aa:
            new_seq = seq[:i] + aa + seq[(i+1):]
            E = compute_energy(new_seq,h,J)
            d_sm[i,aa] = np.round(E-E0,4)
    return d_sm


def compute_sm_energy(seq, h ,J, idx, aa ):
    ''' for SINGLE MUTANTS, given the ref_seq,h,J and idx(pos_mutations) aa(mutated_aa)
    return energy_sum_single_mutants - energy_wild_type '''
    E0 = compute_energy(seq,h,J)
    E_sum_sm = 0
    for i,a_i in zip(idx, aa):
        new_seq = seq[:i] + a_i + seq[(i+1):]
        E = compute_energy(new_seq,h,J)
        E_sum_sm += (E-E0)
    return np.round(E_sum_sm,4)


def compute_energy(seq, h, J, parallel = False):
    if all_standard_aa(seq):
        if(parallel == True):
            #DO NOT USE FOR NOW!!!
            #something weird... E_parallel != E_non_parallel
            # parallel actually slower than non parallel (execution time limited by memory access and not processor time??)
            E = 0
            all_ei = Parallel(n_jobs=num_cores_energy)(delayed(compute_energy_given_ai)(seq, h, J, idx_ai) for idx_ai in range(0,len(seq)))
            E = np.sum(all_ei)
            return E
        if(parallel == False):
            E = 0
            for idx_aa1 in range(0, len(seq)):
                aa1 = seq[idx_aa1]
                E -= h[d_aa_num[aa1], idx_aa1]
                for idx_aa2 in range(idx_aa1+1, len(seq)):
                    aa2 = seq[idx_aa2]
                    E -= J[d_aa_num[aa1], d_aa_num[aa2], idx_aa1, idx_aa2]
            return E

def compute_energy_given_ai(seq,h,J, idx_ai):
    '''e.g. idx_ai=1; computing E_1 = h_1 + J_12 + J_13 etc. (good for parallelization)'''
    ai = seq[idx_ai]
    #print("**", idx_ai, ai)
    ei = h[d_aa_num[ai], idx_ai]
    for idx_aj in range(idx_ai+1, len(seq)):
        aj = seq[idx_aj]
        #print(idx_aj, aj)
        ei -= J[d_aa_num[ai], d_aa_num[aj], idx_ai, idx_aj]
    return ei

#### ENTROPY ####

def compute_entropy_context_dep(ref_seq, h,J ):
    ''' compute context-DEPENDENT entropy (from ref_seq, h, J)'''
    q, N = h.shape
    fi_plm = np.zeros((h.shape))
    #same convections than in Eq.5.8 (PhD thesis)
    for i in range(0,N):
        #compute denominator
        denom = 0
        for b in range(0,q):
            arg_denom = h[b,i]
            for j in range(0,N):
                if(j!=i):
                    aj = d_aa_num[ref_seq[j]]
                    arg_denom += J[b, aj ,i, j]
            denom += np.exp(arg_denom)
        # compute numerator
        for ai in range(0,q):
            arg_num = h[ai,i]
            for j in range(0,N):
                if(j!=i):
                    aj = d_aa_num[ref_seq[j]]
                    arg_num += J[ai, aj ,i, j]
            num = np.exp(arg_num)
            fi_plm[ai,i] = num/denom
    #return the entropy
    S = compute_entropy_from_freq(fi_plm)
    return S

def compute_entropy_context_ind(d_id_seq, pseudocount = 0, remove_gaps = True, base2 = True):
    ''' compute context-independent entropy (from msa)'''
    fi = compute_freq(d_id_seq, pseudocount)
    S = compute_entropy_from_freq(fi, remove_gaps, base2)
    return S

def compute_entropy_from_freq(fi, remove_gaps = True, base2 = True):
    if remove_gaps:
        fi = (fi[:20,:])/np.sum(fi[:20,:], axis = 0)
    qq, N = fi.shape
    S = []
    for i in range(0,N):
        si = 0
        for q in range(0,qq):
            si -= fi[q,i]*np.log(fi[q,i])
        if base2:
            si /= np.log(2)
        S.append(si)
    return S

#### MSA ANALYSIS ####
def get_seq_from_msa(path_msa, format_msa = 'fasta',remove_dot_lower= True):
    ''' read seq in msa, return dictionary id_prot -> seq_prot
    if remove_dot_lower==True: remove dots and lower cases
    multiple formats are supported (e.g. fasta and stockholm)
    BUG with stockholm: SeqIO interprets . as - '''
    d_id_seq = {}
    record_msa = list(SeqIO.parse(open(path_msa,'r'), format_msa))
    for idx_rec, rec in enumerate(record_msa):
        name = rec.id
        seq = rec.seq
        if remove_dot_lower:
            seq = ''.join(char for char in seq if (char.isupper() or char == "-") )
            seq = ''.join(char for char in seq if char !='.')
        d_id_seq[name] = str(seq)
    return d_id_seq

def get_gapped_col(d_id_seq, max_gap_fraction = 0.5):
    fi = compute_freq(d_id_seq)
    all_gap_fraction = []
    gap_idx = []
    for idx in range(0, fi.shape[1]):
        gap_fraction = fi[-1][idx]
        all_gap_fraction.append(gap_fraction)
        if gap_fraction > max_gap_fraction:
            gap_idx.append(idx)
    print('max_gap_fraction={0}, remove {1} columns: {2}'.format(max_gap_fraction, len(gap_idx), gap_idx))
    return all_gap_fraction, gap_idx


def trim_gapped_col(d_id_seq, max_gap_fraction = 0.5):
    ''' rm positions (aa) with more than max_gap_fraction gaps
    return trimmed d_id_seq'''
    _, gap_idx = get_gapped_col(d_id_seq, max_gap_fraction)
    new_d_id_seq = {}
    for seq_id, seq in d_id_seq.items():
        new_seq = [j for i, j in enumerate(seq) if i not in gap_idx]
        new_seq = ''.join(new_seq)
        new_d_id_seq[seq_id] = new_seq
    return new_d_id_seq

def trim_gapped_seq(d_id_seq, max_gap_fraction_seq = 0.5):
    ''' rm a seq if num_gap/len_seq > max_gap_fraction_seq
    return trimmed d_id_seq'''
    new_d_id_seq = {}
    num_trimmed_seq = 0
    for seq_id, seq in d_id_seq.items():
        if compute_gap_fraction(seq)> max_gap_fraction_seq:
            num_trimmed_seq +=1
            continue
        new_d_id_seq[seq_id] = seq
    print('max_gap_fraction_seq={0}, removing {1} sequences'.format(max_gap_fraction_seq, num_trimmed_seq))
    return new_d_id_seq

def get_taxonomy_msa(d_id_seq, df_speclist):
    ''' for each seq in the msa, get the species, taxonomy number and kingdom '''
    all_species = []
    all_tax_number = []
    all_kingdom= []
    for key in list(d_id_seq.keys()):
        species = key.split("_")[-1].split("/")[0]
        tax_number = df_speclist.loc[df_speclist['species'] == species, 'tax_code'].values[0]
        kingdom = df_speclist.loc[df_speclist['species'] == species, 'kingdom'].values[0]
        all_species.append(species)
        all_tax_number.append(int(tax_number))
        all_kingdom.append(kingdom)
    return all_species, all_tax_number, all_kingdom

def read_speclist(path_speclist = './dib_plos/speclist.txt'):
    ''' read speclist file (from Uniprot)
    return df: species, kingdom '''
    all_species = []
    all_tax_code = []
    all_kingdom = []
    with open(path_speclist,'r') as f:
        lines = f.readlines()
        for line in lines:
            #keep only N= line
            if len(line.split())>3:
                if line.split()[3][:2] == "N=":
                    #print(line.split())
                    name = line.split()[0]
                    kingdom = line.split()[1]
                    tax_code= line.split()[2][:-1]
                    all_species.append(name)
                    all_kingdom.append(kingdom)
                    all_tax_code.append(tax_code)
        df = pd.DataFrame({'species':all_species, 'tax_code': all_tax_code, 'kingdom':all_kingdom})
        return df

def trim_msa_taxonomy(d_id_seq, tax_min = None, kingdom = None):
    '''
    a) select only species with taxonomy higher than tax_min
    b) select only belonging to kingdom
    '''
    df_speclist = read_speclist()
    s, t, k = get_taxonomy_msa(d_id_seq, df_speclist)
    if tax_min != None:
        #select only species with taxonomy higher than tax_min
        tax_idx = [i for i in range(0,len(t)) if (t[i] > tax_min) ]
        print("keep only seq with taxonomy < {0}".format(tax_min))
        #new d_id_seq
        new_d_id_seq = {}
        for idx in tax_idx:
            key = list(d_id_seq.keys())[idx]
            new_seq = d_id_seq[key]
            new_d_id_seq[key] = new_seq
        return new_d_id_seq
    if kingdom != None:
        #idx ok
        king_idx = [i for i in range(0,len(k)) if (k[i] == kingdom) ]
        print("keep only {0}".format(kingdom))
        #new d_id_seq
        new_d_id_seq = {}
        for idx in king_idx:
            key = list(d_id_seq.keys())[idx]
            new_seq = d_id_seq[key]
            new_d_id_seq[key] = new_seq
        return new_d_id_seq


def save_trimmed_msa(d_id_seq, path_out_msa):
    ''' save fasta from d_id_seq '''
    f = open(path_out_msa, 'w')
    for seq_id, seq in d_id_seq.items():
        print('>{0}\n{1}'.format(seq_id, seq), file = f)
    f.close()
    return 0

def all_standard_aa(seq):
    '''return True if sequence contains only standard-aa'''
    for char in seq:
        if((char not in valid_aa) and char !='-'):
            #print("seq containing non standard aa: "+char)
            return False
            break
    return True

def get_consensus(d_id_seq):
    ''' get consensus sequence of a MSA '''
    fi = compute_freq(d_id_seq)
    idx_max = (lambda x:np.argmax(fi[:,x]))
    aa_max = [d_num_aa[idx_max(i)] for i in range(0, fi.shape[1])]
    consensus_seq = ''.join(aa for aa in aa_max)
    return consensus_seq

def compute_num_gap(seq):
    '''return the num of gap in a sequence '''
    num_gap = 0
    for _,char in enumerate(seq):
        if(char == '-'):
            num_gap += 1
    return num_gap

def compute_gap_fraction(seq):
    num_gap = compute_num_gap(seq)
    frac_gap = (num_gap+0.0)/len(seq)
    return frac_gap

def compute_diff(ref_seq, seq):
    ''' compute the mutations between two strings, return idx_mut, aa_first_seq(wt), aa_second_seq(mutant)'''
    vec_idx = []
    vec_aa1 = []
    vec_aa2 = []
    for idx, aa in enumerate(zip(ref_seq,seq)):
        aa1 = aa[0]
        aa2 = aa[1]
        if (aa1.lower() != aa2.lower()):
            vec_idx.append(idx)
            vec_aa1.append(aa1)
            vec_aa2.append(aa2)
    return vec_idx, vec_aa1, vec_aa2

def compute_dist(ref_seq, seq):
    distance = sum([1 for x, y in zip(ref_seq, seq) if x.lower() != y.lower()])
    return distance

def compute_seqid(ref_seq, seq):
    '''return the sequence identity (seqid) '''
    distance = compute_dist(ref_seq,seq)
    distance /= len(seq)
    seqid = 1 - distance
    return seqid


def compute_freq(d_id_seq, pseudocount = 0):
    ''' compute single point frequencies of an MSA (from d_id_seq of msa) '''
    N = len(list(d_id_seq.items())[0][1])
    fi = np.zeros(( len(d_aa_num), N))
    for name, seq in d_id_seq.items():
        #print(name)
        for idx_aa, amino_a in enumerate(seq):
            #non standard aa mapped to -
            if amino_a not in valid_aa:
                amino_a = '-'
            fi[d_aa_num[amino_a], idx_aa] += 1
    #add (small) pseudocount to take into account 0 frequencies (0*log(0))
    if pseudocount > 0:
        fi = (1-pseudocount)*fi + pseudocount/2
    #normalize
    fi /= fi.sum(axis = 0)
    return fi

def compute_entropy_from_freq(fi, remove_gaps = True, base2 = True):
    if remove_gaps:
        fi = (fi[:20,:])/np.sum(fi[:20,:], axis = 0)
    qq, N = fi.shape
    S = []
    for i in range(0,N):
        si = 0
        for q in range(0,qq):
            if fi[q,i] == 0:
                si -= 0
            else:
                si -= fi[q,i]*np.log(fi[q,i])
        if base2:
            si /= np.log(2)
        S.append(si)
    return S


def map_pdb_hmm(path_pdb, chain_pdb,  path_hmm):
    ''' map the pdb sequences to an hmm model using hhmalign
    output = dataframe (seq_pdb, idx_pdb, idx_msa)
    '''
    #0. write tmp file with the seq of the pdb(chain)
    pdb = load_pdb(path_pdb, structure_id = 'test_pdb')
    #get chain->pdb_seq,pdb_idx
    d_chain_seq, d_chain_idx = get_pdb_sequences(pdb)
    seq_pdb = d_chain_seq[chain_pdb]
    path_pdb_seq = "tmp_seq_pdb.fa"
    f = open(path_pdb_seq,"w")
    print(">seq_pdb_chain_{0}".format(chain_pdb), file = f)
    print(seq_pdb, file = f)
    f.close()
    #1. run hmmalign
    path_out_hmm = "tmp_hmmalign.txt"
    FOUT = open(path_out_hmm, 'w')
    subprocess.run(['hmmalign', path_hmm, path_pdb_seq], stdout=FOUT, stderr=subprocess.STDOUT)
    FOUT.close()
    #2. parse hmm file
    alignment = AlignIO.read(open(path_out_hmm), "stockholm")
    for record in alignment:
        #print(record.id)
        seq_pdb_aligned_to_msa  = str(record.seq)
    #3. get mapping seq aligned to pdb
    idx_pdb_mapped_to_msa = []
    idx_char = 0
    for char in seq_pdb_aligned_to_msa:
        #print(idx_char, char)
        if char=='-':
            idx_pdb_mapped_to_msa.append(-1)
        else:
            idx_pdb_mapped_to_msa.append(d_chain_idx[chain_pdb][idx_char])
            idx_char += 1
    #4. get mapping seq aligned to msa (match)
    idx_msa = []
    idx_char = 0
    for char in seq_pdb_aligned_to_msa:
        if char != char.upper():
            idx_msa.append(-1)
        else:
            idx_msa.append(idx_char)
            idx_char += 1
    #make dataframe
    df_tmp = pd.DataFrame({'aa_pdb_msa': list(seq_pdb_aligned_to_msa), 'idx_msa': idx_msa, 'idx_pdb': idx_pdb_mapped_to_msa})
    return df_tmp

#### PDB ANALYSIS ####
def get_pdb_sequences(structure):
    ''' return dictionary chain->seq_chain
    also dict chain -> idx_pdb of the aa
    N.B. idx_pdb often is NOT position in pdb seq!'''
    d_chain_seq = {}
    d_chain_idx = {}
    for model in structure:
        for chain_model in model:
            all_res = []
            idx_res_pdb = []
            for residue in chain_model:
                if (residue.resname) not in d_3to1.keys():
                    print('Chain:{0}, not-standard AA: {1}'.format(chain_model.id, residue.resname))
                    continue
                res_1letter = d_3to1[residue.resname]
                idx_res = residue.id[1]
                all_res.append(res_1letter)
                idx_res_pdb.append(idx_res)
            d_chain_seq[chain_model.id] = ''.join(all_res)
            d_chain_idx[chain_model.id] = idx_res_pdb
    return d_chain_seq, d_chain_idx


def load_pdb(path_pdb, structure_id = "id"):
    parser = PDBParser()
    structure = parser.get_structure(structure_id, path_pdb)
    return structure

def intrachain_dist(structure, chain, distance_type = "min"):
    d = interchain_dist(structure, chain, chain, distance_type = distance_type)
    return d

def interchain_dist(structure, chain1, chain2, distance_type = "min"):
    model = structure[0]
    chain1_model = model[chain1]
    chain2_model = model[chain2]
    d = {}
    for _, res1 in enumerate(chain1_model):
        for _, res2 in enumerate(chain2_model):
            idx1 = res1.id[1]
            idx2 = res2.id[1]
            if distance_type == "alpha":
                #distance between carbon alpha
                ca1 = res1["CA"]
                ca2 = res2["CA"]
                distance = ca1 - ca2
                #print(idx1, res1, idx2, res2, distance)
            if distance_type == "min":
                #minimal dist between heavy atoms
                distance = 10000
                for a1 in res1:
                    for a2 in res2:
                        tmp_dist = a1 - a2
                        if tmp_dist< distance:
                            distance=tmp_dist
            d[(idx1,idx2)] = distance
    return d

def make_intrachain_matrix(d_dist):
    matrix  = make_interchain_matrix(d_dist)
    return matrix

def make_interchain_matrix(d_dist):
    n1 = list(d_dist.keys())[-1][0]
    n2 = list(d_dist.keys())[-1][1]
    matrix = np.zeros((n1+1,n2+1))
    for i in range(0, n1):
        for j in range(0, n2):
            matrix[i,j] = d_dist.get((i,j),0)
    return matrix

def plot_cm_intrachain(d_dist, cut_off_dist = -1):
    plot_cm_interchain(d_dist, cut_off_dist = cut_off_dist)
    return 0

def plot_cm_interchain(d_dist, cut_off_dist = -1):
    m = make_interchain_matrix(d_dist)
    if cut_off_dist != -1:
        m = ((m>0) & (m< cut_off_dist)) + 0
        plt.title('cut_off_dist: {0}A'.format(cut_off_dist))
    #for plotting
    plt.imshow(m, cmap = 'Greys', vmin=0, vmax = 2)
    plt.colorbar()
    return 0
