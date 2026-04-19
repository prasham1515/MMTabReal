import os
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
import pandas as pd

load_dotenv()

html_path = r"C:\Users\prash\OneDrive\Desktop\data"

no_html_file_found = []
no_questions_file_found = []

for folder in os.listdir(html_path):
    folder_path = os.path.join(html_path, folder)
    if not os.path.isdir(folder_path):
        continue

    html_files = [f for f in os.listdir(folder_path) if f.endswith(".html")]
    if not html_files:
        print("No HTML file found in", folder_path)
        no_html_file_found.append(folder)
        continue

    html_file_path = os.path.join(folder_path, html_files[0])

    # Parse HTML
    with open(html_file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    tables = soup.find_all("table")
    if not tables:
        print(f"No tables found in HTML file: {html_file_path}")
        continue

    for i, table in enumerate(tables):
        rows = table.find_all("tr")
        table_data = []
        for row in rows:
            cols = row.find_all(["td", "th"])
            row_data = []
            for col in cols:
                # Extract text
                text = col.get_text(strip=True)

                # Check for image
                img_tag = col.find("img")
                if img_tag and img_tag.has_attr("src"):
                    img_src = img_tag["src"]
                    cell_content = f"{text} \t {img_src}" if text else img_src
                else:
                    cell_content = text

                row_data.append(cell_content)
            table_data.append(row_data)

        if not table_data:
            print(f"Empty table found in {html_file_path}")
            continue

        # Convert to DataFrame for easy cleaning
        df = pd.DataFrame(table_data)

        # Normalize empty strings to NaN
        df.replace('', pd.NA, inplace=True)

        # Drop any rows or columns that are entirely empty
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        # Replace remaining NaN (i.e. originally empty cells in non-empty rows/cols) with "-"
        df.fillna('-', inplace=True)

        # Build CSV filename (one per HTML, you can adjust for multiple tables if you like)
        csv_file_name = f"{os.path.splitext(html_files[0])[0]}.csv"
        csv_file_path = os.path.join(folder_path, csv_file_name)

        print(f"Saving cleaned CSV to: {csv_file_path}")
        df.to_csv(csv_file_path, index=False, header=False, encoding='utf-8')
        print(f"Saved CSV: {csv_file_path}")

print("No HTML file found in", no_html_file_found)
print("No questions file found in", no_questions_file_found)
print("Done")
