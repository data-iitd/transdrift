import os

data_file = "electronics.csv"   #Input file with path containing text and labels.
common_words = '../fastText/common_words.txt'   # common_words in E1, E2, Prediction

out_dir = 'result_fastText_30/'  # Outpur directory path
os.mkdirs(out_dir, exist_ok=True)

log_path = 'logs_fastText_30_' + data_file[:-4] + '/'
d2_small = '30'   # Percent of D2_small

epochs = 100 
batch_size = 64
learning_rate = 0.0001

inp_dir = '../fastText/pred_file_30_common_words' # Input directory of word-embeddings.
emb_name = 'pred_' # Prefix of embedding filename
out_file = 'P_fastText'

# Saving parameters
f = open(out_dir + "param_" + data_file[:-4] +".txt", "w")
f.write("INPUT EMB: " + inp_dir + "\n")
f.write("EPS: " + str(epochs) + "\n")
f.write("BATCH SIZE: " + str(batch_size) + "\n")
f.write("learning_rate: " + str(0.0001) + "\n")
f.close()