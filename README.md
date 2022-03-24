# NBHRnet Summary
The digital revolution of noncontact physiological signal monitoring in clinical and home health care is underway, and deep learning techniques are incredibly popular. Camera-based physiological signal monitoring for adults has made considerable progress in recent years. However, most of existing methods and datasets are developed for adult subjects, and until now, there has been no neonatal public database that is collected for developing deep learning method. Thus, in this paper, we introduce a large-scale newborn baby database, named NBHR (newborn baby heart rate estimation database), to fill the abovementioned knowledge gap. A total of 9.6 h of clinical videos (1130 videos totaling 921 GB) and reference vital signs are recorded from 257 infants at 0–6 days old. The facial videos and corresponding synchronized physiological signals, including photoplethysmograph information, heart rate, and oxygen saturation level, are recorded in our database. This large-scale database could be used to develop deep learning methods to estimate heart rate or oxygen saturation levels. Furthermore, a multitask deep learning method, called NBHRnet, is proposed to estimate heart rate based on the NBHR database, and the model is succinct that it can be deployed on a computer without GPUs. The experimental results indicate that NBHRnet yields competitive performance in predicting infant heart rate, with a mean absolute error of 3.97 bpm and a mean absolute percentage error of 3.28%; additionally, it can estimate heart rate almost instantaneously (2 s/60 frames). Our datasets are freely publicly available by 


![image](https://user-images.githubusercontent.com/11665683/159866843-acddb87a-23f4-43f8-a229-154283dbc8d5.png)


# Installation

1. Tensorflow-1.14
2. Keras-2.2.5

# Paper
Huang B, Chen W, Lin C L, et al. A neonatal dataset and benchmark for non-contact neonatal heart rate monitoring based on spatio-temporal neural networks[J]. Engineering Applications of Artificial Intelligence, 2021, 106: 104447. https://doi.org/10.1016/j.engappai.2021.104447

