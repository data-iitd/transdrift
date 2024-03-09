import os

# Data loader	
BATCH_SIZE = 15 		 	# Batch size
SHUFFLE_TRAIN = True 		# Train Loader shuffle 
SHUFFLE_VAL = False		    # Val Loader shuffle 
SHUFFLE_TEST = False		# Test Loader shuffle 

POSITIONAL_ENCODING = False # Transformer positional encoding

# Model Training
INPUT = 50				# Input dimension
MODEL = 192				# Model dimension
HEADS = 4				# Number of heads
CLASSES = 50			# Number of classes
LAYERS = 4				# Number of layers
DROPOUT = 0.0 			# Dropout
LR = 5e-4				# Learning rate

EPS = 100				# Number of Epochs
WARMUP = 30				# Warmup for cosine scheduler
GRADIENT_CLIP_VAL = 5   


# Data Location
DATASET = '/DATA/covidwiki/yelp/w2v_embedding_yelp/'		    # Input directory 
D1 = '/D1_50d.txt'						            # D1 Dataset
D2_SMALL = '/D2_small_30_50d.txt'				    # D2_small Dataset
D2 = '/D2_50d.txt'						            # D2_Dataset

# COMMON_WORDS = "common_words.txt"
COMMON_WORDS = "../../yelp/common_words.txt"
COMMON_WORDS_D2_SMALL = "../../yelp/common_words_small_30.txt"

SAVE_PRED = True 						# Flag for deciding whether to save predictions
PRED_FOLDER = "../../yelp/pred_file_30_common_words"
log_path = "../../yelp/logs_30"

try:
    os.mkdir(PRED_FOLDER)
except:
    pass
f = open(PRED_FOLDER + "/param.txt", "w")

f.write("BATCH_SIZE "+ str(BATCH_SIZE)+"\n")
f.write("SHUFFLE_TRAIN "+ str(SHUFFLE_TRAIN)+"\n")
f.write("SHUFFLE_VAL "+str(SHUFFLE_VAL)+"\n")
f.write("SHUFFLE_TEST "+str(SHUFFLE_TEST)+"\n")
f.write("POSITIONAL_ENCODING "+str(POSITIONAL_ENCODING)+"\n")
f.write("INPUT "+str(INPUT)+"\n")

f.write("MODEL "+str(MODEL)+"\n")
f.write("HEADS "+str(HEADS)+"\n")
f.write("CLASSES "+str(CLASSES)+"\n")
f.write("LAYERS "+str(LAYERS)+"\n")
f.write("DROPOUT "+str(DROPOUT)+"\n")
f.write("LR "+str(LR)+"\n")

f.write("EPS "+str(EPS)+"\n")
f.write("WARMUP "+str(WARMUP)+"\n")
f.write("GRADIENT_CLIP_VAL "+str(GRADIENT_CLIP_VAL)+"\n")

f.write("DATASET "+DATASET+"\n")
f.write("D1 "+D1+"\n")
f.write("D2_SMALL "+D2_SMALL+"\n")
f.write("D2 "+D2+"\n")
f.write("COMMON_WORDS "+COMMON_WORDS+"\n")
f.write("COMMON_WORDS_D2_SMALL "+COMMON_WORDS_D2_SMALL+"\n")

f.write("SAVE_PRED "+str(SAVE_PRED)+"\n")
f.write("PRED_FOLDER "+PRED_FOLDER+"\n")

f.close()
