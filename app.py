import base64
import os
import cv2
import requests
from halo import Halo
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from loguru import logger
from dotenv import load_dotenv
import re
import spacy

# Load environment variables from .env file
load_dotenv()

# Load SpaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

# Constants
USE_CAMERA = os.getenv('USE_CAMERA', 'True') == 'True'
TEST_RECEIPT = "images/IMG_0053.jpg"
IMAGE_DIR = 'images'
INVOICE_DIR = 'invoices'
IMAGE_PATH = os.path.join(IMAGE_DIR, 'captured_receipt.jpg') if USE_CAMERA else TEST_RECEIPT
OUTPUT_PDF_PATH = os.path.join(INVOICE_DIR, 'invoice.pdf')

# Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(INVOICE_DIR, exist_ok=True)


class ReceiptScanner:
    @staticmethod
    def capture_image_from_camera():
        """Capture an image from the default camera and save it to a file."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open webcam")

            spinner = Halo(text='Capturing image from camera', spinner='dots')
            spinner.start()

            while True:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to capture image")
                cv2.imshow('Camera', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    cv2.imwrite(IMAGE_PATH, frame)
                    spinner.succeed('Image captured successfully')
                    break
                elif key == ord('q'):
                    spinner.warn('Quitting without capturing an image')
                    raise Exception("Quitting without capturing an image")

            cap.release()
            cv2.destroyAllWindows()
            return IMAGE_PATH
        except Exception as e:
            logger.error(f"Error capturing image from camera: {e}")
            raise

    @staticmethod
    def preprocess_image(img):
        """Apply advanced preprocessing techniques to the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return gray

    @staticmethod
    def encode_image(image_path):
        """Encode the image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def extract_text_from_image(image_path):
        """Extract text from an image using OpenAI Vision."""
        try:
            # Check if the file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""

            # Encode the image
            base64_image = ReceiptScanner.encode_image(image_path)

            # OpenAI API request
            api_key = os.getenv('OPENAI_API_KEY')
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": "Please analyze the attached receipt image and extract the items purchased, "
                                     "their prices, and the total amount etc. Provide this information in a structured "
                                     "markdown format. Additionally, include any helpful tax write-off information "
                                     "related to the items."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

                        ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_data = response.json()

            # Extract and return the text from the response
            text = response_data['choices'][0]['message']['content']
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""

    @staticmethod
    def extract_entities(text):
        """Extract named entities from the receipt text."""
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    @staticmethod
    def parse_receipt_structure(text):
        """Parse structured data from receipt text."""
        try:
            # Split the text into lines
            lines = text.split("\n")
            lines = [line.strip() for line in lines if line.strip()]

            # Extract the total amount
            total = None
            for line in lines:
                if "total" in line.lower():
                    total = re.findall(r"\d+\.\d+", line)
                    if total:
                        total = float(total[0])
                    break

            # Extract the items and amounts
            items = []
            for line in lines:
                if re.search(r"\d+\.\d+", line):
                    amount = re.findall(r"\d+\.\d+", line)
                    if amount:
                        amount = float(amount[0])
                        desc = line.replace(str(amount), "").strip()
                        items.append((desc, amount))

            return items, total
        except Exception as e:
            logger.error(f"Error parsing receipt structure: {e}")
            return [], None


class InvoiceGenerator:
    @staticmethod
    def generate_invoice(lines, total, output_path, extracted_text):
        """Generate a PDF invoice from a list of lines and a total."""
        try:
            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter

            c.setFont("Helvetica", 24)
            c.drawString(200, 750, "Invoice")

            # Add the extracted text to the PDF
            c.setFont("Helvetica", 12)
            textobject = c.beginText()
            textobject.setTextOrigin(50, 700)
            textobject.textLines(extracted_text)
            c.drawText(textobject)

            # Calculate the y-coordinate for the start of the invoice details
            # based on the number of lines in the extracted text
            num_lines = len(extracted_text.split("\n"))
            y = 700 - num_lines * 14  # Adjust the multiplier as needed

            c.setFont("Helvetica", 12)
            c.drawString(50, y, "Item Description")
            c.drawString(400, y, "Amount")

            y -= 20
            for line in lines:
                desc, amount = line
                c.drawString(50, y, desc)
                c.drawString(400, y, f"${amount}")
                y -= 20

            if total:
                c.drawString(50, y - 20, f"Total: ${total}")

            c.save()
        except Exception as e:
            logger.error(f"Error generating invoice: {e}")
            raise


def main():
    try:
        scanner = ReceiptScanner()
        if USE_CAMERA:
            image_path = scanner.capture_image_from_camera()
        else:
            image_path = IMAGE_PATH

        if image_path:
            text = scanner.extract_text_from_image(image_path)
            if text:
                # Debugging: print the extracted text
                logger.info(f"Extracted Text:\n{text}")

                # Extract entities from the text
                entities = scanner.extract_entities(text)
                logger.info(f"Entities: {entities}")

                lines, total = scanner.parse_receipt_structure(text)
                if lines:
                    InvoiceGenerator().generate_invoice(lines, total, OUTPUT_PDF_PATH, text)
                    logger.info(f"Invoice generated successfully: {OUTPUT_PDF_PATH}")
                else:
                    logger.warning("No lines found in the receipt.")
            else:
                logger.warning("No text extracted from the image.")
        else:
            logger.error("Failed to capture image.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

