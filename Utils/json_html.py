import json

def json_to_html(json_file, output_file1, output_file2):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    datasets = [data[:21], data[21:42]]  
    output_files = [output_file1, output_file2]
    
    for idx, dataset in enumerate(datasets):
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>National Parks</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 90%; border: 1px solid #ccc; padding: 20px; border-radius: 10px; 
                             box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); }
                img { max-width: 100%; height: auto; border-radius: 10px; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f4f4f4; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>National Parks</h1>
                <table>
                    <tr>
                        <th>Name</th>
                        <th>Image</th>
                        <th>Date Established</th>
                        <th>Area</th>
                        <th>Visitors (2022)</th>
                        <th>Description</th>
                    </tr>
        """
        
        for park in dataset:
            img_src= "./"+park.get('Image', '#')
            html_content += f"""
            <tr>
                <td>{park.get('Name', 'N/A')}</td>
                
                <td><img src= {img_src} alt="{park.get('Name', 'Image')}"></td>
                
                <td>{park.get('Date established as park[12]', 'N/A')}</td>
                <td>{park.get('Area (2023)[8]', 'N/A')}</td>
                <td>{park.get('Recreation visitors (2022)[11]', 'N/A')}</td>
                <td>{park.get('Description', 'N/A')}</td>
            </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_files[idx], "w", encoding="utf-8") as file:
            file.write(html_content)

# Convert JSON to HTML
data_file = "national_parks_table_1.json"
json_to_html(data_file, "national_parks_1.html", "national_parks_2.html")
