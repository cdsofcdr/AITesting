import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from skimage.metrics import structural_similarity as ssim
import base64

# Define paths
screenshot_folder = "screenshots"
initial_image_path = os.path.join(screenshot_folder, "initial_screenshot.jpg")
final_image_path = os.path.join(screenshot_folder, "final_screenshot.jpg")
report_path = os.path.join(screenshot_folder, "report.html")

# Function to take a screenshot
def take_screenshot(file_path):
    print(f"Taking screenshot and saving to {file_path}")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get("https://cdsofcdr.github.io/AITesting/")
    time.sleep(5)  # Wait for the page to load completely
    driver.save_screenshot(file_path)
    driver.quit()
    print(f"Screenshot saved to {file_path}")

# Function to load images
def load_images(image_path1, image_path2):
    print(f"Loading images from {image_path1} and {image_path2}")
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    if image1 is None or image2 is None:
        print("One of the images did not load correctly.")
    else:
        print("Images loaded successfully.")
    return image1, image2

# Function to compute SSIM and absolute difference
def compute_differences(image1, image2):
    print("Computing SSIM and absolute differences")
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # SSIM
    score, diff_ssim = ssim(gray_image1, gray_image2, full=True)
    diff_ssim = (diff_ssim * 255).astype("uint8")
    print(f"SSIM score: {score}")
    
    # Absolute difference
    diff_abs = cv2.absdiff(image1, image2)
    
    return score, diff_ssim, diff_abs

# Function to highlight differences
def highlight_differences(image1, image2, diff_ssim, diff_abs):
    print("Highlighting differences")
    # Thresholding for SSIM differences
    _, thresh_ssim = cv2.threshold(diff_ssim, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Thresholding for absolute differences
    gray_diff_abs = cv2.cvtColor(diff_abs, cv2.COLOR_BGR2GRAY)
    _, thresh_abs = cv2.threshold(gray_diff_abs, 25, 255, cv2.THRESH_BINARY)
    
    # Combine the differences
    combined_thresh = cv2.bitwise_or(thresh_ssim, thresh_abs)
    
    contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(image1.shape, dtype='uint8')
    filled_image = image2.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 40:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(mask, [contour], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_image, [contour], 0, (0, 255, 0), -1)
    
    return image1, image2, mask, filled_image

# Function to plot images and save them to a file
def plot_images(image1, image2, diff_ssim, diff_abs, image1_highlighted, image2_highlighted, mask):
    print("Plotting images")
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 4, 1)
    plt.title("Original Image 1")
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 4, 2)
    plt.title("Original Image 2")
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 4, 3)
    plt.title("SSIM Difference")
    plt.imshow(diff_ssim, cmap='gray')

    plt.subplot(2, 4, 4)
    plt.title("Absolute Difference")
    plt.imshow(cv2.cvtColor(diff_abs, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 4, 5)
    plt.title("Highlighted Differences Image 1")
    plt.imshow(cv2.cvtColor(image1_highlighted, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 4, 6)
    plt.title("Highlighted Differences Image 2")
    plt.imshow(cv2.cvtColor(image2_highlighted, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 4, 7)
    plt.title("Mask of Differences")
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    comparison_image_path = os.path.join(screenshot_folder, "comparison.png")
    plt.savefig(comparison_image_path)
    plt.close()
    print(f"Images plotted and saved to {comparison_image_path}")
    return comparison_image_path

# Function to generate HTML report
def generate_html_report(image1, image2, diff_ssim, diff_abs, image1_highlighted, image2_highlighted, mask, report_path, comparison_image_path):
    print("Generating HTML report")
    
    def image_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    html_content = f"""
    <html>
    <head>
        <title>Visual Test Report</title>
    </head>
    <body>
        <h1>Visual Test Report</h1>
        <h2>Original Image 1</h2>
        <img src="data:image/jpeg;base64,{image_to_base64(initial_image_path)}" />
        <h2>Original Image 2</h2>
        <img src="data:image/jpeg;base64,{image_to_base64(final_image_path)}" />
        <h2>SSIM Difference</h2>
        <img src="data:image/jpeg;base64,{cv2.imencode('.jpg', diff_ssim)[1].tobytes().hex()}" />
        <h2>Absolute Difference</h2>
        <img src="data:image/jpeg;base64,{cv2.imencode('.jpg', diff_abs)[1].tobytes().hex()}" />
        <h2>Highlighted Differences Image 1</h2>
        <img src="data:image/jpeg;base64,{cv2.imencode('.jpg', image1_highlighted)[1].tobytes().hex()}" />
        <h2>Highlighted Differences Image 2</h2>
        <img src="data:image/jpeg;base64,{cv2.imencode('.jpg', image2_highlighted)[1].tobytes().hex()}" />
        <h2>Mask of Differences</h2>
        <img src="data:image/jpeg;base64,{cv2.imencode('.jpg', mask)[1].tobytes().hex()}" />
        <h2>Comparison Plot</h2>
        <img src="data:image/png;base64,{image_to_base64(comparison_image_path)}" />
    </body>
    </html>
    """
    with open(report_path, "w") as f:
        f.write(html_content)
    print(f"HTML report generated and saved to {report_path}")

# Main function
def main():
    print("Starting main function")
    # Create the screenshots folder if it doesn't exist
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    # Check the state of the folder and take action accordingly
    if not os.path.exists(initial_image_path):
        print("Creating initial screenshot...")
        take_screenshot(initial_image_path)
    elif not os.path.exists(final_image_path):
        print("Creating final screenshot...")
        take_screenshot(final_image_path)
    else:
        print("Both initial and final screenshots exist. Updating screenshots...")
        os.remove(initial_image_path)
        os.rename(final_image_path, initial_image_path)
        take_screenshot(final_image_path)

        # Load and compare images
        image1, image2 = load_images(initial_image_path, final_image_path)
        score, diff_ssim, diff_abs = compute_differences(image1, image2)
        print(f"SSIM Score: {score}")

        if score >= 0.95:
            print("Test Passed: No significant visual differences detected.")
        else:
            print("Test Failed: Significant visual differences detected.")
            image1_highlighted, image2_highlighted, mask, filled_image = highlight_differences(image1, image2, diff_ssim, diff_abs)
            comparison_image_path = plot_images(image1, image2, diff_ssim, diff_abs, image1_highlighted, image2_highlighted, mask)
            generate_html_report(image1, image2, diff_ssim, diff_abs, image1_highlighted, image2_highlighted, mask, report_path, comparison_image_path)
            exit(1)  # Exit with a non-zero status to indicate test failure

if __name__ == "__main__":
    main()
    print("Main function completed")