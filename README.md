# J-Cole-Lyrics-Generator
Natural Language Processing project to generate Rap lyrics based on J Cole's music.

  For this project i have used a Tensorflow configuration for the NN containing 4 layers.
I used CuDNNLSTM for speed since LSTM was very slow for this dataset.

![image](https://user-images.githubusercontent.com/112874778/190962936-f704f4ea-e5ef-4458-9652-7f55ef0ed64e.png)

  After reading the data and store every lyrics in a variable I used the Tokenizer module with the method tokenize(), taking the input as string and store it with lower case everywhere. Tokenization is a way of separating a piece of text into smaller units called tokens. Tokens can be either words, characters or subwords.
  After the tokenization, I created 3 variables that help the program, the first one counts the total number of words, second one the total verses and the last one saves the words and how many times they appear as a dictionary, this was created for better visualization in the terminal of some words and their frequency because sometimes sequences were repeating very often

![image](https://user-images.githubusercontent.com/112874778/190962978-cf526128-e017-4898-a03f-8929ed6f81f6.png)

  In this sequence the variable inputData stores every tokenization in an array and maxVerse computes the maximum length of a lyric. After this inputVerse stores the tokenized data based on the maximum length while being inside pad_sequences, this makes every lyric being the same length.

![image](https://user-images.githubusercontent.com/112874778/190962291-4ca43137-b6e5-4aa3-a906-e8add9d223bc.png)

  This method takes a random verse, number of words to generate and when to hit newline, in order to make it look like a song. The first step is to tokenize the input string and then make it fit in the max sequence verse that was created before. After this the next word gets predicted based on the average from the predicted array. The output word is found and stored after the same string, forming the song. 

Results:

Up, up and away
hey do you trust me do the wait
think he ain't tell


Be wary of any man that claims

flow with the boulevard lot of body

won't craft my nc mane know thicker

tellin' views hop credit diamonds juice happens

glass backpack lately pregnant you've prayin' harder

mos styles potential changed treasure to freakin'


