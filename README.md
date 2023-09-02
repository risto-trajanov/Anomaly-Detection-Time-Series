# Anomaly Detection Using Deep-Learning Models on Sensor Data

## Dataset

The dataset used in this study is the Pump Sensor Data, which is a time-series dataset collected from a sensor attached to a pump in an industrial setting. It consists of 52,170 rows and 52 columns, with each row representing a time-step and each column representing a sensor reading. The dataset was obtained from Kaggle and can be found [here](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data).

## Analysis

### Data Preprocessing

Before the dataset could be used for training the models, preprocessing was required to ensure the quality of the data. The following steps were performed:

1. Removal of rows or columns with missing values (NaN).
2. Interpolation to fill in any remaining missing values.
3. Min-max scaling to ensure that each feature has the same range.

### Models Used

Two models were used in this study for anomaly detection:

1. **Autoencoders (Pytorch):**
   - Model architecture: Encoder (60 * 49 > 25 > Relu > 15) and Decoder (15 > Relu > 25 > 60 * 49).
   - Training parameters: 10 epochs, Mean Squared Error loss, Adam Optimizer (learning rate: 0.001).
   - Total trainable parameters: 13.3 K.

2. **Deep Neural Network (TensorFlow):**
   - Model architecture: Two hidden layers (200 and 40 neurons), single output neuron with a ReLU activation function.
   - Training parameters: 150 epochs, Stochastic Gradient Descent optimizer, Mean Square Error loss.
   - Total trainable parameters: 600 K.

## Results

The Autoencoder model did not perform well in detecting anomalies in the sensor data. Although the model showed promising results during training, it did not generalize well on the test data. This may be due to the lack of negative samples used in training the model.

In contrast, the deep learning model yielded highly promising results in detecting anomalies in the sensor data. By predicting whether a machine is going to break down in the next 20 minutes, the model achieved zero false negatives, indicating that it can accurately predict potential failures.

The confusion matrix obtained from the model also supports its efficacy in detecting anomalies with high precision and recall values.

## Conclusion and Future Work

In conclusion, we have explored and compared two approaches to anomaly detection in sensor data from pumps. Our deep learning model was successful in detecting all anomalies and provided a reliable classification of potential equipment failure in the next 20 minutes. On the other hand, our autoencoder model did not meet our expectations, mainly due to the lack of negative samples during training.

For future work, we suggest exploring different types of deep learning models, addressing the imbalance in the dataset, and investigating unsupervised anomaly detection techniques. Additionally, real-time monitoring of pump sensor data in an industrial setting can help evaluate the effectiveness of our approach in detecting equipment failure and preventing unplanned downtime.

## References

- [Link to the Kaggle dataset](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data).
- [Pytorch Lightning](https://www.pytorchlightning.ai/index.html).
- [TensorFlow](https://www.tensorflow.org/).
- Malhotra, P., Vig, L., Shroff, G., Agarwal, P. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
