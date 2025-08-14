import os
from pdf2image import convert_from_path

def convert_pdfs_to_pngs(pdf_folder, output_folder, poppler_path):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all PDF files in the pdf_folder
    for file_name in os.listdir(pdf_folder):
        if file_name.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, file_name)
            print(f"Converting {file_name}...")

            # Convert pdf to images
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

            # Save each page as PNG
            for i, page in enumerate(pages):
                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_page_{i+1}.png")
                page.save(output_file, 'PNG')

pdf_folder = r'C:\Desktop\MMV2\TrainingData\Single Lined New PDFs'
output_folder = r'C:\Desktop\MMV2\TrainingData\Single Lined New PNGs'
poppler_path = r'C:\Users\braxt\measure-magician\poppler-24.08.0\Library\bin'  # adjust this to your actual poppler bin path

convert_pdfs_to_pngs(pdf_folder, output_folder, poppler_path)
