# Emotion Detection from Facial Expressions

This project uses a pre-trained Keras model to detect emotions from a live webcam feed. It identifies faces and predicts emotions such as angry, disgust, fear, happy, neutral, sad, and surprise.

## Prerequisites

*   Python 3.x
*   pip (Python package installer)
*   A virtual environment manager (e.g., `venv`) is recommended.

## Setup and Installation

1.  **Clone the repository (if applicable) or download the project files.**

2.  **Navigate to the project directory:**
    ```bash
    cd path\to\your\project\readface\readface
    ```

3.  **Create and activate a virtual environment:**
    *   Create a virtual environment (e.g., named `venv`):
        ```bash
        python -m venv venv
        ```
    *   Activate the virtual environment:
        *   On Windows (Command Prompt/PowerShell):
            ```bash
            .\venv\Scripts\activate
            ```
        *   On macOS/Linux (bash/zsh):
            ```bash
            source venv/bin/activate
            ```

4.  **Install the required Python packages:**
    Make sure you have the `requirements.txt` file in your project directory.
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary libraries, including OpenCV, TensorFlow, and NumPy.

## Running the Emotion Detector

There are two ways to run the emotion detection program:

### Option 1: Using the Python script directly

1.  Ensure your virtual environment is activated.
2.  Run the `test_emotion_model.py` script:
    ```bash
    python test_emotion_model.py
    ```
    This will open a window showing your webcam feed with detected faces and their predicted emotions. Press 'q' to quit the application.

### Option 2: Using the batch script (Windows only)

1.  Navigate to the project directory `c:\Users\ggeta\Documents\readface\readface`.
2.  Double-click the `build_and_run.bat` file, or run it from the command prompt:
    ```bash
    .\build_and_run.bat
    ```
    This script will automatically activate the virtual environment (assuming it's in a `venv` folder), install requirements, and then run the `test_emotion_model.py` script.

## Project Structure

*   `test_emotion_model.py`: The main script to run the live emotion detection.
*   `train_emotion_model.py`: (Assumed) Script for training the emotion detection model.
*   `requirements.txt`: A list of Python packages required for the project.
*   `build_and_run.bat`: A batch script for easy setup and execution on Windows.
*   `model/`: Directory containing the pre-trained emotion detection models (`.h5` files).
    *   `emotion_model.h5`: The default model used by `test_emotion_model.py`.

## Notes

*   The application requires access to a webcam.
*   The accuracy of emotion detection depends on the quality of the pre-trained model and environmental factors like lighting and face visibility.
