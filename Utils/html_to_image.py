from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# 📌 Input directory containing subfolders with HTML files
input_dir = r"C:\Users\prash\OneDrive\Desktop\data"

# 📌 Output directory for storing images (maintains the same subfolder structure)
output_base_dir = r"C:\Users\prash\OneDrive\Desktop\image-data"
os.makedirs(output_base_dir, exist_ok=True)

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless=new")  # Improved headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Function to process each HTML file
def process_html_file(html_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the output folder if it doesn't exist
    output_base = os.path.join(output_dir, os.path.splitext(os.path.basename(html_file))[0])

    # Start WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Convert file path to proper URL format
    file_url = "file:///" + html_file.replace("\\", "/")

    # Load the HTML file
    driver.get(file_url)
    time.sleep(3)  # Ensure full rendering

    # Expand all scrollable areas to ensure the full table is captured
    driver.execute_script("""
        let elements = document.querySelectorAll("table, div, body");
        for (let el of elements) {
            el.style.overflow = 'visible';
            el.style.maxWidth = 'none';
            el.style.maxHeight = 'none';
        }
        window.scrollTo(0, 0);  // Scroll to top before taking screenshot
    """)

    # Get the full document dimensions
    full_width = driver.execute_script("return document.body.scrollWidth")
    full_height = driver.execute_script("return document.body.scrollHeight")

    # Resize the browser to fit the entire content
    driver.set_window_size(full_width + 100, full_height + 200)  # Add padding to avoid cropping
    time.sleep(2)  # Allow adjustment

    # Take the full-page screenshot
    screenshot_path = output_base + "_full.png"
    driver.save_screenshot(screenshot_path)  # Capture everything

    driver.quit()

    print(f"✅ Full table screenshot saved: {screenshot_path}")

# Loop through all subfolders and process the HTML file inside each
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)

    if os.path.isdir(subfolder_path):  # Only process directories
        html_file = None

        # Find the first HTML file in the subfolder
        for file in os.listdir(subfolder_path):
            if file.endswith(".html"):
                html_file = os.path.join(subfolder_path, file)
                break  # Process only one HTML file per subfolder

        if html_file:
            output_subfolder = os.path.join(output_base_dir, subfolder)  # Keep the same subfolder structure
            process_html_file(html_file, output_subfolder)
