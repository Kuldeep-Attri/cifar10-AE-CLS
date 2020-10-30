# :dart: CIFAR-10 Classification: With Autoencoder & Classifier

> This is the first time I am working with Autoencoders. I have read the theory at my past work and during my undergraduate, but this is the first time implementing and experimenting with the autoencoders and using them for supervised learning task(Classification of CIFAR-10 here). So I will be doing experiments to understand it in more detail and also to answer the questions I have regarding the autoencodrs. Hopefully this document will take you through my thinking process on a newer problem as well as it will show my implementation skills. Please be critical and provide meaningful suggestions about something I did wrong or understood incorrectly. Thank you :)


**For all experiments: I keep similar image processing.**

I pass the input image with some image processing steps:
- I scale to 256 and crop to 224(for Standards Archs. e.g. VGG16, ResNet50 etc.), but for my experiments I keep it 32\*32 as the original size unless specified.
- I use convert image 0-255 -> 0-1
- Then I normalize it with subtracting mean(ImageNet data) and dividing by std(ImageNet data)
- For training:
	- I do RandomHorizontalFlip as well as Shuffle the data


**For all experiments: loss function I combine the both losses from classifier and autoencoder as shown below.**
```
loss = alpha*classifier_loss + beta*autoencoder_loss
```

This is to help in training end2end(alpha and beta > 0) or separately(by setting alpha or beta = 0) as well as if we just want to train supervised architecture without using the unsupervised architecture(beta=0, this to varify the hypothesis that unsupervised architecture helps in improving the supervised task's performance.)


For my first 3 experiments(v1, v2 & v3), I use SGD as the Optimizer with base learning rate = 0.01 and then reducing it by a factor of 0.1 with every 10 epochs with total number of epochs = 40. In other experiments I specified the Optimizer and other details.
```
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
adaptive_lr_rate = 10 epochs by a factor of 0.1
```


## Experiments 1(v1): A simple Linear Layer based Neural Network. 

During my undergraduate, I worked on a simple digit classifier using MNIST dataset and for that I designed a very simple only Linear Layer based Neural Network and it achieved really high performance. I do know that CIFAR10 is a much more complex dataset compared to the MNIST but I still was curious about the performance of simple Linear Layers based NN(Neural Network) so I started my experiments with it.

For experiment 1: 

After the image processing step we flatten the 32\*32\*3 array to a 3072 vector.

Run the command below to see the help message:
```
python train.py  -h
```

Run the command below to train the model of architecture version 1 with the default settings:
```
python train.py  --version v1
```

**Results of Classification Accuracy:**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- |
| 58.95 | Complete | v1 | 3072 | L(512)>L(256)>L(128) | L(128)>L'(10) | 1.0 | 1.0 |
| 59.58 | Complete | v1.1 | 3072 | L(1024)>L(512) | L(128)>L'(10) | 1.0 | 1.0 |
| 60.76 | Complete | v1.2 | 3072 | L(2048)>L(1024)>L(512) | L(128)>L'(10) | 1.0 | 1.0 |
| 58.50 | Complete | v1 | 3072 | L(512)>L(256)>L(128) | L(128)>L'(10) | 1.0 | 0.0 |
| 59.79 | Complete | v1.1 | 3072 | L(1024)>L(512) | L(128)>L'(10) | 1.0 | 0.0 |
| 59.94 | Complete | v1.2 | 3072 | L(2048)>L(1024)>L(512) | L(128)>L'(10) | 1.0 | 0.0 |

L = linear + relu

L' = linear

But this model is not just about analyzing the classification results, I also wanted to see the value of the Autoencoder part of Network and how well it is able to reconstruct the input images or how well the latent layers stores important features from the data to be able to regenerate original image/features again.

**Results of Image Reconstruction:**

![Version1 Reconstruction Images](/examples/1000_dog.png)![Version1 Reconstruction Images](/examples/1001_airplane.png)![Version1 Reconstruction Images](/examples/1002_ship.png)![Version1 Reconstruction Images](/examples/1003_deer.png)![Version1 Reconstruction Images](/examples/1005_automobile.png)![Version1 Reconstruction Images](/examples/1008_truck.png)![Version1 Reconstruction Images](/examples/1009_frog.png)![Version1 Reconstruction Images](/examples/1024_cat.png)![Version1 Reconstruction Images](/examples/1037_horse.png)![Version1 Reconstruction Images](/examples/1038_bird.png)

![Version1 Reconstruction Images](/outputs/output_v1/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v1/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v1_1/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_1/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v1_2/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v1_2/1038_bird_recon.png)


As expected results are not as good on CIFAR-10 as I have achieved on the MNIST dataset by just using a simple Linear Layer based NN. Surprisingly, accuracy was slightly lower when we use 3 layers(I was expecting deeper might be better) in Autoencoder architecture compare to using 2 layers(in version 1.1) but this might also be due to having higher dimension of the last latent layer's features(128 vs 512). Even the reconstructed images(v1 vs v1.1) does not look quite good which suggests that we are losing some information in the latent layers and hence not able to generate image completely. We can clearly see that the reconstructed images with v1.1(row 3) are better than the v1(row 2). Now this raises a question: is this because of the higher dimension of the last latent layer or because of the lower number of layers in v1.1, and to verify this I did one more experiment(v1.2) of having 3 layers with higher dimension in the last latent layer.

From the last experiment i.e. v1.2, I observed that the model's classification performance increases with deeper network and having higher dimension of the last latent layer but the reconstruction becomes little more difficult as we go far(deep in number of layers) from the original data i.e. more number of layers. 

**Conclusion:**

- Performance is not great using Linear NN.
- Having lower depth helps in better reconstruction(as I was expecting) but does not necessarily mean in having higher classification performance.
- In fact, having more depth in latent layer helps in storing higher dimension features that can help in classification and having more the number of features might give small boost in classification performance.
- The other really important observation is that for this dataset and Neural Network combination, using an Autoencoder does not necessarily improves the performance of classification task. Even simply training the classifier achieves similar results.


## Experiments 2(v2): A Neural Network using both Convolution and Linear Layers.

Now as expected, just a simple Linear Layer Based NN did not achieve really high performance. Next I moved to trying the famous convolution layer. In this experiment, I used both Convolution(conv) Layers as well as Linear Layers. Convolution layers helps in learning spatial features and so can learn some geometrical shapes which is a big help in computer vision tasks and so I was expecting this to work much better than experiment 1.


*For some part of this experiment we Rescale and CenterCrop the image to a bigger size.(Please see the table below: Input Size)*

Run the command below to see the help message:
```
python train.py  -h
```

Run the command below to train the model of architecture version 1 with the default settings:
```
python train.py  --version v2
```

**Results of Classification Accuracy:**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- |
| 80.11 | Complete | v2 | 32\*32 | C(32)>C(32)>C(64)>L(512) | L(128)>L'(10) | 1.0 | 1.0 |
| 77.72 | Complete | v2.1 | 32\*32 | C(32)>C(32)>L(512) | L(128)>L'(10) | 1.0 | 1.0 |
| 82.69 | Complete | v2.2 | 32\*32 | C'(32)>C(32)>C'(32)>C(64)>C(64)>L(512) | L(128)>L'(10) | 1.0 | 1.0 |
| 84.11 | Complete | v2.3 | 64\*64 | C'(32)>C(32)>C'(32)>C(64)>C(64)>L(512) | L(128)>L'(10) | 1.0 | 1.0 |
| 79.92 | Complete | v2 | 32\*32 | C(32)>C(32)>C(64)>L(512) | L(512)>L'(10) | 1.0 | 0.0 |
| 77.36 | Complete | v2.1 | 32\*32 | C(32)>C(32)>L(512) | L(128)>L'(10) | 1.0 | 0.0 |
| 82.95 | Complete | v2.2 | 32\*32 | C'(32)>C(32)>C'(32)>C(64)>C(64)>L(512) | L(128)>L'(10) | 1.0 | 0.0 |
| 83.80 | Complete | v2.3 | 64\*64 | C'(32)>C(32)>C'(32)>C(64)>C(64)>L(512) | L(128)>L'(10) | 1.0 | 0.0 |

C --> conv + relu + max_pool

C' --> conv + relu

L --> linear + relu

L' --> linear

**Results of Image Reconstruction:**

![Version1 Reconstruction Images](/examples/1000_dog.png)![Version1 Reconstruction Images](/examples/1001_airplane.png)![Version1 Reconstruction Images](/examples/1002_ship.png)![Version1 Reconstruction Images](/examples/1003_deer.png)![Version1 Reconstruction Images](/examples/1005_automobile.png)![Version1 Reconstruction Images](/examples/1008_truck.png)![Version1 Reconstruction Images](/examples/1009_frog.png)![Version1 Reconstruction Images](/examples/1024_cat.png)![Version1 Reconstruction Images](/examples/1037_horse.png)![Version1 Reconstruction Images](/examples/1038_bird.png)

![Version1 Reconstruction Images](/outputs/output_v2/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v2/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v2_1/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_1/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v2_2/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_2/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v2_3/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v2_3/1038_bird_recon.png)


As expected, the performance increased by more than or close to 20% after using the convolution layers. Even keeping the same number of layer(v2.1) as the Linear Layer based NN we can see a high jump in performance. After seeing this I tried and played around having more number of layers in Autoencoder. In this also we see that the better classification performance comes when we have a deeper architecture(v2.2). In this we use 5 conv layer and 1 linear layer in Autoencoder. This whole experiments also follows similar pattern in results as experiment 1. We get the best reconstruction results when we have less number of layers in the Autoencoder(v2.1, row3) but gives the worst performance in terms of classification. And v2.2, which gives best results in classification has really bad results in Autoencoder. *This does point out what if having deeper decoder is not able to recover from the features of encoders because classification performance on these features is better and so features are good.* I also wanted to see the impact of the size of input image and we can see from v2.3 that both classification and reconstruction results has improved. Using a 64\*64 input size improves the whole model. Now we can clearly see that having bigger input can help in getting superior features at the end of the encoder which improves classification performance but what I also observed is that even a deeper decoder works better when we have bigger input size. I do not know if it is True for all cases that having bigger input size will help in decoding the features(Need to try more experiments for that).  


**Conclusion:**
- Using Convolution layers improves the performance of the model significantly.
- Again, deeper network gives better classification performance(similar logic as expt. 1) but does not necessarily gives better reconstruction results. A smaller autoencoder gives better reconstruction of data.
- Using bigger input size can improve both classification and reconstruction results. Helps in learning better higher dimension features.
- Again, even in this experiment, using a autoencoder does not necessarily improves the performance of the classifier and even simply training a classifier can achieve similar accuracy. 


## Experiments 3(v3): A Neural Network using only the Convolution Layers.

Now I even wanted to go one step further, I only used the convolution layer(Except: 1 linear layer for mapping final conv layer to number of outputs). I wanted to see if removing Linear Layers has any impact on the Classification performance as well as on Autoencoder. I was doing this because researchers lately have shown that a deep network can work same or even better when you don't use Linear Layers e.g. ResNet, DenseNet, SqueezeNet vs VGG16, AlexNet(these uses Linear Layers).


Run the command below to see the help message:
```
python train.py  -h
```

Run the command below to train the model of architecture version 1 with the default settings:
```
python train.py  --version v3
```

**Results of Classification Accuracy:**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- |
| 79.23 | Complete | v3 | 32\*32 | C(32)>C(32)>C(64)>C(64) | C'(64)>L'(10) | 1.0 | 1.0 |
| 78.72 | Complete | v3.1 | 32\*32 | C(32)>C(32)>C(64) | C(64)>C'(64)>L'(10) | 1.0 | 1.0 |
| 80.63 | Complete | v3.2 | 32\*32 | C'(32)>C(32)>C'(32)>C(32)>C'(64)>C(64) | C(64)>L'(10) | 1.0 | 1.0 |
| 81.51 | Complete | v3.3 | 32\*32 | C'(32)>C(32) | C'(32)>C(32)>C'(64)>C(64)>C(64)>L'(10) | 1.0 | 1.0 |
| 79.25 | Complete | v3 | 32\*32 | C(32)>C(32)>C(64)>C(64) | C'(64)>L'(10) | 1.0 | 0.0 |
| 79.30 | Complete | v3.1 | 32\*32 | C(32)>C(32)>C(64) | C(64)>C'(64)>L'(10) | 1.0 | 0.0 |
| 81.17 | Complete | v3.2 | 32\*32 | C'(32)>C(32)>C'(32)>C(32)>C'(64)>C(64) | C'(128)>L'(10) | 1.0 | 0.0 |
| 81.38 | Complete | v3.3 | 32\*32 | C'(32)>C(32)>C'(32)>C(32) | C'(64)>C(64)>C'(128)>L'(10) | 1.0 | 0.0 |

C --> conv + relu + max_pool

C' --> conv + relu

L' --> linear

**Results of Image Reconstruction:**

![Version1 Reconstruction Images](/examples/1000_dog.png)![Version1 Reconstruction Images](/examples/1001_airplane.png)![Version1 Reconstruction Images](/examples/1002_ship.png)![Version1 Reconstruction Images](/examples/1003_deer.png)![Version1 Reconstruction Images](/examples/1005_automobile.png)![Version1 Reconstruction Images](/examples/1008_truck.png)![Version1 Reconstruction Images](/examples/1009_frog.png)![Version1 Reconstruction Images](/examples/1024_cat.png)![Version1 Reconstruction Images](/examples/1037_horse.png)![Version1 Reconstruction Images](/examples/1038_bird.png)

![Version1 Reconstruction Images](/outputs/output_v3/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v3/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v3_1/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_1/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v3_2/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_2/1038_bird_recon.png)

![Version1 Reconstruction Images](/outputs/output_v3_3/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v3_3/1038_bird_recon.png)


In this experiment I did not see any significant improvment in the Classification performance. We were achieving similar performance of accuracy being close to 80%. So having only the Convolutional layer in the architecture did not increase the classification performance. But I observed something really interesting that having only convolutional layers in the Autoencoder arch achieves better reconstruction results. The results still follow similar pattern when it comes to number of layers i.e. deeper network gives better classification results but poor reconstrution results. But in this case reconstruction results are much better compared to the last 2 experiments. In fact when we use smaller autoencoder(v3.3) it is really hard to distinguish between the real images(row 1) vs reconstrcuted images from v3.3.(row 5) so reconstruction is almost close to perfect. 

In this experiment, we moved around number of layers in autoencoder and classifier archs. So using low number of layer in Autoencoder archs and and then more layers in classifier achieves the best results for both classifier and reconstruction. **Well this is really interesting because this is what we have also discovered in the above experiments that shallower autoencoder achieves good results for reconstruction and to achieve better results for classification task we can get deeper or higher dimensional features by using the more layers in the classifier architecture.** This is definitely a new and interesting finding for me at least cause I was not aware of this behavior before I started all this experiments and this observation can help everyone when it comes to designing an architecture and I will also try to use it for getting even a better model.  

**Conclusion:**
- Using only convolution layers does not necessarily improves the performance on CIFAR10.
- Using only convolution layers improves the reconstruction results significantly up to a point where it is hard to distinguish between the real images and reconstructed images.
- Even this experiment follows similar pattern, having deeper architecture achieves better classification results and having shallower autoencoder achieves better results.
- Again, even in this experiment, using a autoencoder does not necessarily improves the performance of the classifier and even simply training a classifier can achieve similar accuracy.


**I would like to comment on the hypothesis of using Autoencoder's encoder part for features can improve the classification part. I personally think that CIFAR10 might be really simple dataset where even a simple supervised learning can help in achieving good results. So I do not think hypothesis is incorrect but I believe on simpler dataset such as CIFAR10 we might get salient features just from the classification based arch and do not need to use the autoencoder to get best features for the supervised task.**



## Experiments 4(v4): A Neural Network designed from the observations made in above 3 experiments.

After doing the above three experiments, I have better understanding of autoencoder and their use for supervised tasks. From experiment 2 and 3 we can see having a shallower autoencoder is good for reconstruction and having deeper classifier is good for classification performance. We also notices that having linear layers towards the end in the model achieves higher performance than compared to just using convolution layer(v3.3 vs. v2.2). So based on all these observations I made a very simple new arch named model_v4. This one has 2 conv layers in the autoencoder just as v3.2 and more layers(3 conv + 3 linear) in classifier because we want a deeper architecture. I used Linear layers in the classifier this time because of what we observed in the v2.2. 

We started with this architecture and trained and end-to-end model and achieved results as shown below.

**Results of Classification Accuracy:**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | Optimizer | Epochs |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | --------- | ------ |
| 82.92 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | SGD | 40 |

Even though this model architecture achieved the highest accuracy performance, I observed something training that did not happen for the above experiments. I saw how the test classification loss started increasing while the train classification loss what still decreasing.

**Graph of Train_Cls loss vs. Test_Cls loss**
![Graph1 Images](/outputs/graphs/graph1.png)

**From the above graph we can clearly see that this model is overfitting, and overfitting is common problem but something is really interesting here because I did not see such behavior of the loss function in above experiments(1-3). Mainly because we did not have too many parameters in experiment 3 as all conv layers but we had it in experiment 2 similar to this experiment 4. In experiment 2 the most parameters based Linear Layer was present in the Autoencoder. This means basically the autoencoder loss was acting as regularizer on the Linear as well as other layers. So this shows an additional good behavior of Autoencoder that it can also act as some sort of regularizer to keep a track on overfitting of the model. Again, this observation is also really new for me and might be for you as well and I would recommend to try more experiments from your side to prove or disprove this theory.**


Now to remedy the above situation I added Dropout(p=0.3) layers in the classifier arch after conv1 and conv2 layers.

**Results of Classification Accuracy after Dropout:**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | Optimizer | Epochs |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | --------- | ------ |
| 83.56 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | SGD | 40 |

We can see that how just adding the Dropout layer in the Classifier increased the accuracy. We can see that how Dropout stop from model to overfit in the graph below.

**Graph of Train_Cls loss vs. Test_Cls loss**
![Graph2 Images](/outputs/graphs/graph2.png)

Now to further improve the above situation I also added a L2 Norm regularizer on the weights of the model.

```
loss = alpha*classifier_loss + beta*autoencoder_loss + gamma*l2_reg_loss
```


**Results of Classification Accuracy after Regularizer:**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | 
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ |
| 85.69 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 |

We can see that how adding the L2 Norm Regularizer on weights in the loss has increased the accuracy. We can see that how adding Regularizer in loss stop from model to overfit in the graph below. We can also see the reconstructed images below, reconstructed images looks so similar to real images here and it is hard to distinguish between the 2.

**Graph of Train_Cls loss vs. Test_Cls loss**
![Graph3 Images](/outputs/graphs/graph3.png)

**Results of Image Reconstruction after Regularizer:**

![Version1 Reconstruction Images](/examples/1000_dog.png)![Version1 Reconstruction Images](/examples/1001_airplane.png)![Version1 Reconstruction Images](/examples/1002_ship.png)![Version1 Reconstruction Images](/examples/1003_deer.png)![Version1 Reconstruction Images](/examples/1005_automobile.png)![Version1 Reconstruction Images](/examples/1008_truck.png)![Version1 Reconstruction Images](/examples/1009_frog.png)![Version1 Reconstruction Images](/examples/1024_cat.png)![Version1 Reconstruction Images](/examples/1037_horse.png)![Version1 Reconstruction Images](/examples/1038_bird.png)

![Version1 Reconstruction Images](/outputs/output_v4_reg/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg/1038_bird_recon.png)


I also tried using only the Conv layer with bigger stride to reduce spatial dimension instead of the Max Pooling layer in the Autoencoder arch. This was done to see how much the Max Pooling affects the Autoencoder performance but I also wanted to see its impact on the full model including the Classification task. I have read before that just using higher stride we might get better result for reconstruction, I wanted to confirm.


**Results of Classification Accuracy after Regularizer with stride=2:**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ |
| 82.99 | Complete | v4 | 32\*32 | C'(32)>C'(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 |


**Results of Image Reconstruction after Regularizer:**

![Version1 Reconstruction Images](/examples/1000_dog.png)![Version1 Reconstruction Images](/examples/1001_airplane.png)![Version1 Reconstruction Images](/examples/1002_ship.png)![Version1 Reconstruction Images](/examples/1003_deer.png)![Version1 Reconstruction Images](/examples/1005_automobile.png)![Version1 Reconstruction Images](/examples/1008_truck.png)![Version1 Reconstruction Images](/examples/1009_frog.png)![Version1 Reconstruction Images](/examples/1024_cat.png)![Version1 Reconstruction Images](/examples/1037_horse.png)![Version1 Reconstruction Images](/examples/1038_bird.png)

![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_v4_reg_wstd2/1038_bird_recon.png)

The performance of the Classification model decreased with using the Stride=2 instead of using max Pooling layer. The reconstruction results are really good but nothing much better than the above so I did not explore this side more. 

**So I fixed my model architecture to the the model with dropout + l2_regularizer for further experiments.**

I tried the different optimizer(Adam) this time to see the impact of it: 

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | 
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ |
| 82.89 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | Adam | 40 |

The Accuracy with using Adam was lower than the SGD when I trained with same number of epochs 40. So I went ahead and tried 1 more optimizer. 


I tried the different optimizer(Adagrad) this time to see the impact of it: 

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | 
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ |
| 76.66 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | Adagrad | 40 |

The Accuracy with using Adagrad was even worse than the SGD and Adam when I trained with same number of epochs 40. So from now I will use SGD which achieves best results. Note that for both these experiments of different optimizer I also played with base lr (0.1 -- 0.0001) and so the above results are on the best base lr for that optimizer. 


**Results of Classification Accuracy**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 85.69 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | True |
| 84.01 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 0.0 | 0.001 | SGD | 40 | True |
| 84.56 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | False |


*Setting beta=0, in theory it should not train the autodeocder's decoder and hence the model should only be a classifier models but in my past experiences when we do backprop model also adjust the parameters of decoder, so to make sure that I am comparing with right model I trained separate models only using the classifier by removing the decoder part from the model.*

Results for this:

| Accuracy(%) | Data | Version | Input Size | Arch Encoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ------------ | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 85.27 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 0.0 | 0.001 | SGD | 40 | True |

As we can see even a network without having the decoder works as good as with having decoder in the above experiments. 

**Conclusion:**
- From our first 3 experiments we were able to gather information to design this neural network which achieves  the highest performance of almost 86% when it comes to simple neural network.
- We can also see how the results of reconstruction is really good, and looks exactly like real images.
- We also tried using the stride=2 in the Autoencoder part instead of using the Max Pooling layer, and we saw the reconstruction is really amazing with it.(but even max pool does not perform bad at all, please have a look.)
- We observed how having Reconstruction loss can act as a Regularizer which controls the overfitting of the model.
- But in case we have more number of parameters in Classifier we can see the huge impact of Dropout and Weight regularizer in improving the performance of the Classification task.
- We also observed the impact of different type of optimizer and learning rate for this given 40 epochs we achieved the best results using a simple SGD optimizer. 
- We also trained a model by completely removing the decoder part just to verify if setting beta=0 is good practice or not. We found that even this gives similar performance as using the Autoencoder hence undermining the value of the *Hypothesis* for this specific dataset and network combination.
- We also tried with different alpha and beta values and we achieved similar results
	- Though the most difference I ever observed was in the experiment 4 but still not significant.
- Now the reason for hypothesis to be not working quite well here might just the simplicity of the dataset and I would like to perform and experiment to confirm if that is the case.
- Overall using a simple CNN with 32\*32 image and no pre-training on any dataset before achieving 86% accuracy seems quite reasonable to me.
- Please note that I am not saying this is the best we can, I am saying given the time constraints and time required for running many experiments this is best I could achieve.
- When we compare the results of this with some standard architecture such as VGG16 or Resnet50 we will see a huge difference in accuracy roughly ~ 7-10%. 
	- I think this mainly because of 3 reasons:
		- First, very well designed architecture, took months of experimentation in reaching the arch.	
		- Second, the input image size is also 224 vs 32. And in a small experiment above we saw how 64\*64 input size can increase the performance and so using 224 might be even more helpful.
		-Third, having the pre knowledge about the world or how the features look like. Models are trained on ImageNet dataset and hence they have learned better basic features to extract salient features.
- Please feel free to suggest any modification to improve further.


## Using Standard and Famous Deep Neural Network.

### VGG16 as Encoder + Classifier:

In this I use the VGG16 architecture as the encoder of images as well as the classifier and for decoding I design a simple 5 layer decoder.

Decoder Arch --> C(512)>C(256)>C(128)>C(64)>C(3)

C --> ConvTranspose2d + ReLU

**Results of Classification Accuracy:**

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | alpha | beta | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ----- | ---- | --------- | ------ | ---------- |
| 93.58 | Complete | vgg | 224\*224 | C(512)>C(256)>C(128)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | True |
| 93.49 | Complete | vgg | 224\*224 | C(512)>C(256)>C(128)>C(64)>C(3) | 1.0 | 0.0 | SGD | 40 | True |
| 93.99 | Complete | vgg | 224\*224 | C(512)>C(256)>C(128)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | False |


*Setting beta=0, in theory it should not train the autoencoder and hence the model should only be a classifier models but in my past experiences when we do backprop model also adjust the parameters of decoder, so to make sure that I am comparing with right model I trained separate models only using the classifier by removing the decoder part from the model.*

Results for this:

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | No Decoder | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ---------- | --------- | ------ | ---------- |
| 93.61 | Complete | vgg | 224\*224 | C(512)>C(256)>C(128)>C(64)>C(3) | True| SGD | 40 | True |


**Results of Image Reconstruction after Regularizer:**

![Version1 Reconstruction Images](/examples/1000_dog.png)![Version1 Reconstruction Images](/examples/1001_airplane.png)![Version1 Reconstruction Images](/examples/1002_ship.png)![Version1 Reconstruction Images](/examples/1003_deer.png)![Version1 Reconstruction Images](/examples/1005_automobile.png)![Version1 Reconstruction Images](/examples/1008_truck.png)![Version1 Reconstruction Images](/examples/1009_frog.png)![Version1 Reconstruction Images](/examples/1024_cat.png)![Version1 Reconstruction Images](/examples/1037_horse.png)![Version1 Reconstruction Images](/examples/1038_bird.png)

![Version1 Reconstruction Images](/outputs/output_vgg_conv/1000_dog_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1001_airplane_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1002_ship_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1003_deer_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1005_automobile_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1008_truck_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1009_frog_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1024_cat_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1037_horse_recon.png)![Version1 Reconstruction Images](/outputs/output_vgg_conv/1038_bird_recon.png)


As we can see that using the VGG16 as the encoder and classifier we can achieve really high accuracy for the Classification task but when we try to reconstruct the images the results does not seem really great.  I am wondering if it is because of a very simple Decoder Architecture that I used. In my next experiment I would like to try the exact same architecture of Decoder as the Encoder and see if it improves any results.


### ResNet50 as Encoder + Classifier:

In this I use the ResNet50 architecture as the encoder of images as well as the classifier and for this also I design a simple 5 layer decoder.

Decoder Arch --> C(1024)>C(512)>C(256)>C(64)>C(3)

C --> ConvTranspose2d + ReLU

**Results of Classification Accuracy:**

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | alpha | beta | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ----- | ---- | --------- | ------ | ---------- |
| 96.65 | Complete | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | True |
| 96.46 | Complete | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 0.0 | SGD | 40 | True |
| 96.79 | Complete | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | False |


*Same for resent, setting beta=0 might not be the best so tried same experiment.*

Results for this:

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | No Decoder | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ---------- | --------- | ------ | ---------- |
| 96.69 | Complete | resnet | 224\*224 | C(512)>C(256)>C(128)>C(64)>C(3) | True| SGD | 40 | True |


As we can see that using the ResNet50 as the encoder and classifier we can achieve really high accuracy(the best I have ever seen on CIFAR10 dataset) for the Classification task but when we try to reconstruct the images the results does not seem really great even for Resnet. I am thinking similar logic would be behind this too as the VGG16.


### MobileNetV2 as Encoder + Classifier:

Results and Conclusion to be added soon!!!


### SqueezeNetV1_1 as Encoder + Classifier:

Results and Conclusion to be added soon!!!



## Using Constrained dataset.

For this, given the time constraint, I will only use the archs which achieves best results in the above experiments. So I will only work with ModelV4(v4, from experiment 4) and ResNet50 (from the standard architectures experiment). *Please understand the time constraint I have and well as limited time for working on this project did not allow me to explore more.*

Constraint: Only can use 50% images for 3 classes(bird, deer and truck) and can use any number for other classes.

Now, in my understanding, CIFAR10 is really simple dataset and so even having 50% images/data would not impact much of the performance and so I went ahead and started a simple training on the constrained dataset using the above 2 model with the hyper-parameters and settings that achieved best results.

**Result for Classification Accuracy**

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 82.27 | Constrained | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | True |

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | alpha | beta | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ----- | ---- | --------- | ------ | ---------- |
| 96.11 | Complete | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | True |


From the above table we can see that ResNet still achieve really high(almost similar to complete dataset) performance even with 50% of the data, so improving this using any technique would be basically improving the model performance with using less data which seems quite difficult to do but I will still perform the experiments on it. In case of smaller network we can see difference of roughly of 3-4% which is significant. So I will try to improve this from this base accuracy of 82.27%. 


### Data size based approaches:

**Undersample**

In this we bring all classes's images to almost equal number. We can randomly choose similar number of images as much we have in classes with lower number of data. E.g. for this case we can choose 50% dataset for roughly all classes(other than the 3). This is reasonable to do when we feel we have enough number of images for all classes including the classes with lower number of images or to put in simple words when we have abundance of data :). I am not sure how well this will work for this case, but it is worth a try.

**Result for Classification Accuracy**

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | alpha | beta | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ----- | ---- | --------- | ------ | ---------- |
| 95.49 | Constrained_US | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | True |


| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 80.19 | Constrained_US | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | True |


**Conclusion:**

- Undersample results are poorer than the results we got just training on the directly the given constrained dataset.
- So having less images leads to poor performance atleast for this Dataset and Network combination.
- Accuracy decreased by 0.6% and 2% approx for both ResNet and Simple NN respectively. 


**Oversample**

In this we can try to have more data for the classes which has less number of data.
Simplest way to do it the repetition of data. But we can also use some sophisticated approaches such as SMOTE(Synthetic Minority Over Sampling Technique). For this experiment I just copy 2500 images one more time for the classes with less images.

**Result for Classification Accuracy**

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | alpha | beta | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ----- | ---- | --------- | ------ | ---------- |
| 96.44 | Constrained_OS | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | True |


| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 83.29 | Constrained_OS | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | True |


**Conclusion:**

- Oversample results are better than the results we got just training on the directly the given constrained dataset.
- So having more images leads to superior performance at least for this Dataset and Network combination.
- We saw little more than 1% increase in the performance of smaller network(V4) using oversample data.(This is still lower than the original complete data but improves from the constrained dataset) 


### Loss Function modification based approach:

In our baseline methods, we add the loss for modeling the problem of Jigsaw Puzzle Solving and I would like to see if exploiting the structure available in the problem actually helps us in getting the best accuracy.

In this, I divide the image into rectangular grids, assign the different image parts some id and jumble the image parts to get a final jumbled image. The task is to reconstruct the original image from the jumbled image and try to minimize the reconstruction error.


```
loss = alpha*classifier_loss + beta*autoencoder_loss + gamma*l2_reg_loss + delta*jigsaw_recon_loss
```

**Result for Classification Accuracy**

| Accuracy(%) | Data | Version | Input Size | Arch Decoder | alpha | beta | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ----- | ---- | --------- | ------ | ---------- |
| --.-- | Constrained | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | True |


| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 82.03 | Constrained | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | True |


**Conclusion:**
- Adding the additional loss function also just got us close to the original image for simple Neural Network i.e. V4.
- I still need to run and add results for ResNet, but because ResNet already achieve really high accuracy on the constrained data so I am not too confident about getting improved results.
- I have experience of using this type of loss for semi-supervised tasks and I have seen improvement on the task that I worked with my current company when I worked on one of the client's data but unfortunately did not see any improvement on simple supervised task with CIFAR10 dataset.


### Network Modification based approach:

In this, we modify the network architecture so that we can have better learning for the features coming from the data with low numbers. Because higher features will be more biased towards having the features from the classes with more number of images so we will pass the lower features using a single convolution layer and concatenate them with final higher features and then run the classifier on the concatenated features. By doing this, final features will be mixture of both higher as well lower features which will help in learning features for both low and high number of images classes.


| Accuracy(%) | Data | Version | Input Size | Arch Decoder | alpha | beta | Optimizer | Epochs | End-to-End |
| ----------- | ---- | ------- | ---------- | ------------ | ----- | ---- | --------- | ------ | ---------- |
| 96.07 | Constrained | resnet | 224\*224 | C(1024)>C(512)>C(256)>C(64)>C(3) | 1.0 | 1.0 | SGD | 40 | True |



| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 82.79 | Constrained | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | True |


**Conclusion:**
- We can a very small improvement in the simple Neural Network but the ResNet performs exactly the same.
- So this approach can be really helpful in terms of more complex data as I have seen by myself in my past work.


**One important thing I would like to point out that all these approaches(Data, Loss and Network Modification) are complementary to each other and so if we combine 2 or more approaches they might give even a better accuracy. I will try these experiments in coming days.**



## Adding more complexity in dataset to verify Hypothesis.

For doing this, I added a Gaussian Blur over the CIFAR10 dataset which make data more difficult to learn features. This in a way increases the complexity of the dataset and could be simple and great setup for checking the Hypothesis i.e. autoencoder helps in learning better features and so in better performance for the classification task when data is hard to learn. Because I feel CIFAR10 is pretty easy data we did not see any significant difference above when we train only the supervised model vs combination of both.

**Results of Classification Accuracy on Blurred dataset:**

This table is for the network using both Autoencoder and the Classifier in model arch.

| Accuracy(%) | Data | Version | Input Size | Arch Autoencoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ---------------- | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 60.16 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 1.0 | 0.001 | SGD | 40 | True |


This table is for the network using only Encoder and the Classifier in model arch.

| Accuracy(%) | Data | Version | Input Size | Arch Encoder | Arch Classifier | alpha | beta | gamma | Optimizer | Epochs | E2E |
| ----------- | ---- | ------- | ---------- | ------------ | --------------- | ----- | ---- | ----- | --------- | ------ | --- |
| 55.37 | Complete | v4 | 32\*32 | C'(32)>C(32) | C'(32)>C(64)>C(64)>L(512)>L(128)>L(10) | 1.0 | 0.0 | 0.001 | SGD | 40 | True |


In this, we can clearly see the difference between a simple supervised model vs an autoencoder with a classifier. The difference is roughly 5-6% which we were not able see before without adding the complexity in the dataset.

**Conclusion:**

- We can see a significant difference between autoencoder + classifier vs a simple encoder + classifier architecture.
- Also proves that for simpler dataset(CIFAR10) we do not necessarily needs to use autoencoder and simple supervised type architecture might give similar performance.
- This experiment strongly verify the hypothesis as well :)
- Also my point that CIFAR10 is a very simple dataset. :)



# Thank you :)  





