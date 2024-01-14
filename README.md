# Sandbox
# AI Models Sandbox

Welcome to the AI Models Sandbox, a Django web app showcasing different AI models. This repository allows you to explore and interact with various machine learning models for different tasks. Below is a brief overview of the currently featured models:

## Models Overview

### 1. Digits
- **Description:** MNIST-based model for recognizing handwritten digits.
- **Usage:** Works best with dark text on a light background.
- **Test Accuracy:** 99.5%

### 2. Sentiment
- **Description:** Sentiment analysis model based on the IMDb dataset.
- **Usage:** Determines whether a review is positive or negative.
- **Test Accuracy:** 86.1%
- **Note:** The model needs to be downloaded separately or trained on your own machine due to its large file size.

### 3. Summary
- **Description:** Seq2Seq model for generating headlines from articles.
- **Dataset:** CNN-DailyMail
- **Status:** Work in progress

## Getting Started

Follow the instructions below to set up and run the Django web app with the included AI models:

### Prerequisites
- Make sure you have Python installed on your machine.
- Install required Python packages by running:
  ```bash
  pip install -r requirements.txt
  ```

  ### Running the Django App
1. Clone this repository:
   ```bash
   git clone https://github.com/JosipVucic/Sandbox.git
   cd your-repo
   ```

2. Migrate the database:
   ```bash
   python manage.py migrate
   ```

3. Run the development server:
   ```bash
   python manage.py runserver
   ```

4. Open your browser and navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to explore the AI Models Sandbox.

## License
This project is licensed under the [MIT License](LICENSE).
