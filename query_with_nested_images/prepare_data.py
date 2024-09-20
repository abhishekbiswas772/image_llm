import os
import re
import requests
import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()

class Part(Base):
    __tablename__ = 'parts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pnc = Column(String, nullable=False)
    image_path = Column(String, nullable=True)

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

def extract_text_from_images(image_paths):
    subscription_key = os.getenv("subscription_key")
    ocr_url = os.getenv("endpoint") + "/vision/v3.2/read/analyze"

    results = {}

    for image_path in image_paths:
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-Type': 'application/octet-stream'
        }

        with open(image_path, 'rb') as image_file:
            response = requests.post(ocr_url, headers=headers, data=image_file)
            response.raise_for_status()

        operation_location = response.headers['Operation-Location']
        while True:
            result_response = requests.get(operation_location, headers={'Ocp-Apim-Subscription-Key': subscription_key})
            result_response.raise_for_status()
            result = result_response.json()

            if result['status'] == 'succeeded':
                break

        lines = result['analyzeResult']['readResults'][0]['lines']
        detected_text = [clean_text(line['text']) for line in lines if clean_text(line['text'])]
        results[image_path] = detected_text

    return results

def load_and_clean_dataframe(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(["Part No", "Part Description"], axis=1)
    return df

def get_ocr_data(image_paths):
    return extract_text_from_images(image_paths)

def create_value_to_image_path_map(ocr_data):
    value_to_image_path = {}
    for image_path, words in ocr_data.items():
        for word in words:
            value_to_image_path[word] = image_path
    return value_to_image_path

def update_dataframe_with_image_paths(df, value_to_image_path):
    df['image_path'] = df['PNC'].map(value_to_image_path)
    return df

def save_dataframe_to_sql(df, db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    for _, row in df.iterrows():
        part = Part(pnc=str(row['PNC']), image_path=str(row['image_path']))
        session.add(part)
    session.commit()
    session.close()

def create_sql_database(image_dir, csv_file, db_url):
    try:
        df = load_and_clean_dataframe(csv_file)
        ocr_data = get_ocr_data([os.path.join(image_dir, image) for image in os.listdir(image_dir)
                                 if image.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))])
        value_to_image_path = create_value_to_image_path_map(ocr_data)
        df = update_dataframe_with_image_paths(df, value_to_image_path)
        save_dataframe_to_sql(df, db_url)
        print("Database creation and data insertion completed.")
    except Exception as e:
        print(f"Error: {e}")
        print("Unable to store data to the database.")


