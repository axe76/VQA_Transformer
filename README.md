# VQA_Transformer
This repo details the implementation of a Transformer based model for Visual Question Answering. Given an input image and a question about that image, it aims to answer the question appropriately. 

# Architecture
The input questions are passed through an embedding layer and a Transformer encoder model as shown below. <br> 
![transformer](https://user-images.githubusercontent.com/36445587/139669479-acafa32f-8cb8-46db-b53e-092ae6ea8ea9.png)

The output of shape (batch_size,ques_seq_length,d_model) is average pooled along the temporal dimension. The decoder takes the attention vector output by the transformer and the input image (which has been passed through VGG16 except the top layer) and passes it through a Bahdanau attention mechanism followed by GRU layers to give an output sequence representing the answer of the question. <br>

The decoder is based on the Bahdanau Attention based seq2seq model decoder utilised for many text generation tasks as shown below:<br>
![attention_mechanism](https://user-images.githubusercontent.com/36445587/139669770-02c1c1b8-6ffc-4ac6-8ca2-5191996b71c0.jpg)

The resulting model is one that achieves an accuracy comparable to the latest implementations while being more lightweight.

# Usage
To train the model: <br>
```bash
$ python3 main.py
```

The last sections of main.py contain the code for inference and can either be selectively run or utilised in another python file.

# Working
Input Image:<br>
![COCO_train2014_000000027511](https://user-images.githubusercontent.com/36445587/139670430-aead92a7-bb08-4850-b852-2f92fc401ebe.jpg)

Input Question and Output Answer:<br>
![Capture](https://user-images.githubusercontent.com/36445587/139670883-60e4ae25-d445-4e5d-8c7e-ba6f2507aacd.JPG)


