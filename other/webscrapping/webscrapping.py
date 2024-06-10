import fitz  # PyMuPDF
import re
import pandas as pd

def extract_superintendent_info(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Define regular expressions to find relevant information
    district_name_re = re.compile(r'(District|SD|School District)\s*[\d\w\s]*')
    superintendent_re = re.compile(r'Superintendent:?\s*([\w\s]+)')
    email_re = re.compile(r'Email:?\s*([\w\.-]+@[\w\.-]+)')
    phone_re = re.compile(r'Phone:?\s*([\d-]+)')
    
    # List to store extracted data
    data = []
    
    # Iterate through each page in the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Split text into lines for easier processing
        lines = text.split('\n')
        
        # Temporary storage for district information
        district_info = {"District": "", "Superintendent": "", "Email": "", "Phone": ""}
        
        for line in lines:
            district_match = district_name_re.search(line)
            superintendent_match = superintendent_re.search(line)
            email_match = email_re.search(line)
            phone_match = phone_re.search(line)
            
            if district_match:
                district_info["District"] = district_match.group(0)
            if superintendent_match:
                district_info["Superintendent"] = superintendent_match.group(1)
            if email_match:
                district_info["Email"] = email_match.group(1)
            if phone_match:
                district_info["Phone"] = phone_match.group(1)
            
            # Check if we have all necessary information
            if all(district_info.values()):
                data.append(district_info.copy())
                district_info = {"District": "", "Superintendent": "", "Email": "", "Phone": ""}
    
    return data

def save_to_excel(data, output_path):
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

if __name__ == "__main__":
    pdf_path = "origon.pdf"
    output_path = "superintendent_list.xlsx"
    
    # Extract information
    data = extract_superintendent_info(pdf_path)
    
    # Save to Excel
    save_to_excel(data, output_path)

    print(f"Superintendent information saved to {output_path}")
