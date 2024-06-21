# Receipt Scanner and Invoice Generator

This project is a Python application that captures images of receipts using a webcam, processes the image to extract text, and generates an invoice PDF from the extracted information. It uses various libraries for different functionalities including OpenCV for image capture, SpaCy for named entity recognition, ReportLab for PDF generation, and Loguru for logging.

## Features

- Capture receipt images using the webcam.
- Preprocess the captured images for better text extraction.
- Extract text from the receipt images using OpenAI's Vision API.
- Parse the extracted text to identify items and their prices.
- Generate a structured PDF invoice from the parsed receipt data.

## Requirements

- Python 3.6+
- OpenCV
- SpaCy
- ReportLab
- Loguru
- Halo
- dotenv
- Requests

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/bantoinese83/smart-invoice-gen.git
    cd receipt-scanner
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Download the SpaCy model:

    ```sh
    python -m spacy download en_core_web_sm
    ```

5. Create a `.env` file in the root directory and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Ensure your webcam is connected.

2. Run the application:

    ```sh
    python main.py
    ```

3. Follow the on-screen instructions to capture a receipt image using your webcam or use a test image provided in the `images` directory.

4. The application will process the image, extract text, and generate an invoice PDF in the `invoices` directory.

## Code Overview

### Environment Variables

Load environment variables from the `.env` file:

