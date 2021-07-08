# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:13:21 2021

@author: sense
"""

import tensorflow as tf

# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm

import collections
import operator
# read the json file
annotation_file = 'v2_mscoco_train2014_annotations.json'
with open(annotation_file, 'r') as f:
    annotations = json.load(f)
    
from encoder_transf_model import Encoder
from decoder import Decoder
    
PATH = 'train2014/'

# storing the captions and the image name in vectors
all_answers = []
all_answers_qids = []
all_img_name_vector = []
all_accepted_answers = []
for annot in annotations['annotations']:
#     print(annot)
#     break
    ans_dic = collections.defaultdict(int)
    for each in annot['answers']:
      diffans = each['answer']
      if diffans in ans_dic:
        #print(each['answer_confidence'])
        if each['answer_confidence']=='yes':
          ans_dic[diffans]+=4
        if each['answer_confidence']=='maybe':
          ans_dic[diffans]+= 2
        if each['answer_confidence']=='no':
          ans_dic[diffans]+= 1
      else:
        if each['answer_confidence']=='yes':
          ans_dic[diffans]= 4
        if each['answer_confidence']=='maybe':
          ans_dic[diffans]= 2
        if each['answer_confidence']=='no':
          ans_dic[diffans]= 1
#     print(ans_dic.items()) 
#     break
    all_accepted_answers.append(['<start> '+a[0] +' <end>' for a in ans_dic.items()])
    most_fav = max(ans_dic.items(), key=operator.itemgetter(1))[0]
    #print(most_fav)
    caption = '<start> ' + most_fav + ' <end>' #each['answer']
    
    image_id = annot['image_id']
    question_id = annot['question_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_answers.append(caption)
    all_answers_qids.append(question_id)
    
question_file = 'v2_OpenEnded_mscoco_train2014_questions.json'
with open(question_file, 'r') as f:
    questions = json.load(f)

# storing the captions and the image name in vectors
question_ids =[]
all_questions = []
all_img_name_vector_2 = []

for annot in questions['questions']:
    caption = '<start> ' + annot['question'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector_2.append(full_coco_image_path)
    all_questions.append(caption)
    question_ids.append(annot['question_id'])
    
train_answers, train_questions, img_name_vector,train_accepted_answers = shuffle(all_answers,all_questions,
                                          all_img_name_vector,all_accepted_answers,
                                          random_state=1)

# selecting the first 30000 captions from the shuffled set
num_examples = 443757#1000
train_answers = train_answers[:num_examples]
train_questions = train_questions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
train_accepted_answers = train_accepted_answers[:num_examples]
print(img_name_vector[0],train_questions[0],train_answers[0])

print(len(img_name_vector),len(train_questions),len(train_answers))

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    #224 x 224 for VGG 299x299 for Inception
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

image_model = tf.keras.applications.VGG16(include_top=False,
                                                weights='imagenet',input_shape = (224,224,3))
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

encode_train = sorted(set(img_name_vector))

# feel free to change the batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
  #print(batch_features.shape)

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())
    

# choosing the top 10000 words from the vocabulary
top_k = 10000
question_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
question_tokenizer.fit_on_texts(train_questions)
train_question_seqs = question_tokenizer.texts_to_sequences(train_questions)

#new edit
print(question_tokenizer.word_index)
ques_vocab = question_tokenizer.word_index

question_tokenizer.word_index['<pad>'] = 0
question_tokenizer.index_word[0] = '<pad>'

train_question_seqs = question_tokenizer.texts_to_sequences(train_questions)
question_vector = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post')
max_length = calc_max_length(train_question_seqs)
print(max_length)

#new edit
max_q = max_length

##answer vocab
# choosing the top 10000 words from the vocabulary
top_k = 10000
answer_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
answer_tokenizer.fit_on_texts(train_answers)
train_answer_seqs = answer_tokenizer.texts_to_sequences(train_answers)
#new edit
print(answer_tokenizer.word_index)
ans_vocab = answer_tokenizer.word_index

answer_tokenizer.word_index['<pad>'] = 0
answer_tokenizer.index_word[0] = '<pad>'

train_answer_seqs = answer_tokenizer.texts_to_sequences(train_answers)
answer_vector = tf.keras.preprocessing.sequence.pad_sequences(train_answer_seqs, padding='post')
max_length = calc_max_length(train_answer_seqs)
print(max_length)

#new edit
max_a = max_length

img_name_train, img_name_val, question_train, question_val,answer_train, answer_val  = train_test_split(img_name_vector,
                                                                    question_vector,
                                                                    answer_vector,
                                                                    test_size=0.1,
                                                                    random_state=0)

# feel free to change these parameters according to your system's configuration

BATCH_SIZE = 128 #2 #64
BUFFER_SIZE = 300 #1000
num_steps = len(img_name_train) // BATCH_SIZE
# shape of the vector extracted from VGG is (49, 512)
# these two variables represent that
features_shape = 512
attention_features_shape = 49

# loading the numpy files
def map_func(img_name, cap,ans):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor,cap,ans


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, question_train.astype(np.float32), answer_train.astype(np.float32)))

# using map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# shuffling and batching
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

i = 0
for ele1,ele2,ele3 in dataset:
  if i != 1:
    print(ele1.shape)
    print(ele2.shape)
    print(ele3.shape)
    i += 1
  else:
    break


encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=len(ques_vocab),
               maximum_position_encoding=10000)
decoder = Decoder()

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred,loss_object):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
  
    return enc_padding_mask


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

#Start checkpointing from the checkpoint last saved
start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])


'''Maaaaanooonnnss have achieved training'''
EPOCHS = 5

for epoch in range(start_epoch,EPOCHS):
    start = time.time()

    total_loss = 0
    
    for (b,(img,ques,ans)) in enumerate(dataset):
        loss = 0
        enc_padding_mask = create_masks(ques)
        with tf.GradientTape() as tape:
            enc_output = encoder(ques,training=True, mask=enc_padding_mask)
            dec_hidden = enc_output
            
            dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']] * ans.shape[0], 1)        
            
            # Teacher forcing - feeding the target as the next input
            for t in range(1, ans.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden,img)
                
                loss += loss_function(ans[:, t], predictions, loss_object)
                
                # using teacher forcing
                dec_input = tf.expand_dims(ans[:, t], 1)
        
        total_loss += (loss / int(ans.shape[1]))
        
        variables = encoder.variables + decoder.variables
        
        gradients = tape.gradient(loss, variables)
      
        optimizer.apply_gradients(zip(gradients, variables))

        if b % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         b,
                                                         loss.numpy() / int(ans.shape[1])))
    
    
    if epoch % 1 == 0:
        ckpt_manager.save()
        
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss/num_steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start)) 

def evaluate(image,question):
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))
    question_tokenized = question_tokenizer.texts_to_sequences(question)
    question_tensor = tf.keras.preprocessing.sequence.pad_sequences(question_tokenized, maxlen=24,padding='post')
    padding_mask = create_masks(question_tensor)
    enc_output = encoder(question_tensor,training=False, mask=padding_mask)
    dec_hidden = enc_output
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']], 0)
    result = []
    for i in range(max_a):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden,img_tensor_val)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            
        result.append(answer_tokenizer.index_word[predicted_id])
        if answer_tokenizer.index_word[predicted_id] == '<end>':
            return result
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return result

random_image_input = img_name_vector[12]
im = Image.open(random_image_input)
im.show()
test_question = [train_questions[12]]
print(test_question)
test_out = evaluate(random_image_input,test_question)


    

    