import os

data_file = 'yelp_downstream_D1_and_D2.csv'  # This file contains text and its star rating (1-5).
common_words = '../../../yelp/common_words.txt'   # common_words E1, E2, Prediction

d2_small = ''
log_path = 'logs_downstream_using_baseline_'  + d2_small + data_file[:-4]

out_dir = 'result_downstream_D1_D2_baseline_' + d2_small +'/'
epochs = 100
batch_size = 64
learning_rate = 0.001

try:
    os.mkdir(out_dir)
except:
    pass

inp_dir = "../../../yelp/pred_file_30_common_words"  # path of input directory storing embeddings
emb_name = 'pred_' # prefix of input filename.
out_file = "pred"  # Output file name

f = open(out_dir + "param_" + data_file[:-4] +".txt", "w")
f.write("INPUT EMB: " + inp_dir + "\n")
f.write("EPS: " + str(epochs) + "\n")
f.write("BATCH SIZE: " + str(batch_size) + "\n")
f.write("learning_rate: " + str(learning_rate) + "\n")
f.close()


