# Mask Detection with EfficientNetB0

This project utilizes the EfficientNetB0 model for accurate mask detection in images.

![Uploading thumb_6ba06cc2-d1fb-43ea-a30a-44683210b954 (1).png…]()

## Data
The data be found [here](https://zindi.africa/competitions/spot-the-mask/data)


## Installation
`
Install required libraries:

`pip install -r requirements.txt`

Download the EfficientNetB0 weights:

`wget https://github.com/google/efficientnet/releases/download/v0.0/efficientnetb0_weights_tf_dim_ordering_tf_kernels_notop.h5`

Clone this repository:
`git clone https://github.com/Yohanes213/Spot-the-mask.git`

## Usage
1. Download the dataset containing masked and unmasked individuals.
2. Modify the train_labels.csv file with the corresponding filenames and labels (0 for mask, 1 for no mask).
3. Run the train.py script to train the model:
`python train.py`

## Model
Model can found [here](https://drive.google.com/file/d/11DKmLbmXOuxurH48F1HVhtDqKEAokBli/view?usp=sharing)

## Results
The trained model achieves an accuracy of 97.3% on the test set and a loss of 0.080. The confusion matrix shows:

Predicted Label	Mask	No Mask
- Mask	134	4
- No Mask	3	121
  
These results demonstrate the model's effectiveness in identifying people with and without masks.

## Contributing
Feel free to contribute to this project by submitting pull requests with improvements, bug fixes, or new features.
