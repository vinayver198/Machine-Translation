# Machine-Translation
This is a self motivated project for learning the development of seq2seq models. Machine translation is a good example for seq2seq model development.

Steps Involved :
1. The data was preprocessed.
2. It was splitted into test and train dataset
3. The data was encoded into sequences so that it can fed into the network.
4. A Encoder-Decoder Model is used to implement the model.

Architecture of the model Used:
1. The first layer is Embedding layer .
2. Second layer is LSTM layer with 256 memory units
3. The RepeatVector layer is used to repeat the output so that it can be fed as input of decoder model.
4. The decoder is composed of LSTM layer.
5. Finally the last Dense is wrapped with TimeDistributed Layer.

The whole  dataset was not used. This project is for learning purpose and also be extended by tuning the layer, memory units, using birectional layer etc.

