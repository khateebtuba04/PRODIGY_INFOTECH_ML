# NutriScan AI - Food Classification & Calorie Estimation

NutriScan AI is a Streamlit-based web application that uses Deep Learning to recognize food items from images and estimate their calorie content. It aims to help users track their dietary intake and make informed food choices.

## Features
- **Food Recognition**: Identifies 10 different food categories using a Convolutional Neural Network (CNN).
- **Calorie Estimation**: Provides calorie information and serving sizes for detected food items.
- **Premium UI**: Features a modern, responsive design with glassmorphism effects.

## Supported Food Classes
- Apple Pie
- Baby Back Ribs
- Baklava
- Beef Carpaccio
- Beef Tartare
- Beet Salad
- Beignets
- Bibimbap
- Bread Pudding
- Breakfast Burrito

## Project Structure
- `app.py`: Main Streamlit application file.
- `src/`: Contains source code for model training and utilities.
  - `model.py`: CNN model architecture.
  - `train.py`: Script to train the model.
  - `preprocess.py`: Data preprocessing and augmentation.
  - `calorie_utils.py`: Dictionary mapping food classes to calorie data.
- `download_data.py`: Script to download the dataset.
## Requirements
- streamlit
- tensorflow
- numpy
- Pillow

## Installation & Usage

1. **Clone the repository** (or download files):
   ```bash
   git clone <your-repo-url>
   cd repo-name
   ```

2. **Install Dependencies**:
   You can install them individually:
   ```bash
   pip install streamlit tensorflow numpy Pillow
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Training the Model
To retrain the model on your own dataset:
1. Run `download_data.py` to get the dataset.
2. Run the training script:
   ```bash
   python src/train.py --epochs 10
   ```

## License
MIT
