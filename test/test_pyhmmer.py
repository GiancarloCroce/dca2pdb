import pyhmmer

#load hmm
path_hmm = './dir_test_pyhmmer/MHC_I.hmm'
hmm_file = pyhmmer.plan7.HMMFile(path_hmm)
hmm = next(hmm_file)
#pipeline
alphabet = pyhmmer.easel.Alphabet.amino()
background = pyhmmer.plan7.Background(alphabet)
pipeline = pyhmmer.plan7.Pipeline(alphabet, background=background, report_e=1e-15)
#apply hmm to a sequence dataset
seq_file = pyhmmer.easel.SequenceFile("./dir_test_pyhmmer/PF00129_rp15.txt_a")
seq_file.set_digital(alphabet)
hits = pipeline.search_hmm(query=hmm, sequences=seq_file)
print(len(hits))
