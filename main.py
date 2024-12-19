import cv2
import numpy as np
import math
import imutils 
import tkinter as tk
from tkinter import filedialog

def preprocess_image(input):
    res = cv2.GaussianBlur(input, (3, 3), 1);
    res = cv2.medianBlur(res, 5);
    kernel = np.ones((5,5),np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Original Image", input)
    cv2.imshow("Preprocessed Image", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res

def hough_detection(res):
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.blur(res, (5, 5))
    kernel = np.ones((5, 5), np.uint8)
    res = cv2.erode(res, kernel, iterations=1)
    cv2.imshow("Edge Detection Preparation", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cdst = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    res = cv2.Canny(res, 15, 40, apertureSize=3)
    cv2.imshow("Canny Edge Detection", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = cv2.HoughLines(res, 1, np.pi / 180, 80)

    result_lines = []
    line_groups = []

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            grouped = False
            for group in line_groups:
                avg_theta_group = group['average_theta']
                avg_rho_group = group['average_rho']
                
                if abs(avg_theta_group - theta) < 0.1 and abs(avg_rho_group - rho) < 100:
                    group['lines'].append(lines[i][0])
                    # Recalculate average theta and rho for this group after adding the new line
                    group['average_theta'] = np.mean([line[1] for line in group['lines']])
                    group['average_rho'] = np.mean([line[0] for line in group['lines']])
                    grouped = True
                    break

            if not grouped:
                line_groups.append({
                    'lines': [lines[i][0]],
                    'average_theta': theta,
                    'average_rho': rho
                })

        for idx, group in enumerate(line_groups):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # Random color

            for line in group['lines']:
                rho = line[0]
                theta = line[1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                # cv2.line(cdst, pt1, pt2, color, 3, cv2.LINE_AA)

            avg_theta = group['average_theta']
            avg_rho = group['average_rho']
            a = math.cos(avg_theta)
            b = math.sin(avg_theta)
            x0 = a * avg_rho
            y0 = b * avg_rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 255, 0), 12, cv2.LINE_AA)  # Distinct color for average line (bright green)

            result_lines.append({
                'average_theta': group['average_theta'],
                'average_rho': group['average_rho']
            })

        cv2.imshow("Detected Lines (Grouped with Colors and Average Lines)", cdst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result_lines

def rotate_and_crop(input, theta):
    strip_height = int(input.shape[1] * 0.3)
    
    center_y = input.shape[0] // 2
    
    y_start = center_y - strip_height // 2
    y_end = center_y + strip_height // 2
    
    cropped_image = input[y_start:y_end, :]
    cv2.imshow("Rotated and Cropped Image", cropped_image)
    return cropped_image


def find_resistor_body(image):
    pre_bil = cv2.bilateralFilter(image, 5, 80, 80)
    hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)

color_ranges = {
    "black": ((0, 0, 0), (180, 255, 80)),   
    "brown": ((0, 50, 50), (7, 255, 200)),
    "orange": ((0, 16, 191), (9, 255, 255)),
    "yellow": ((24, 50, 50), (35, 255, 255)),
    "green": ((35, 50, 50), (85, 255, 255)),
    "blue": ((85, 50, 50), (130, 255, 255)),
    "violet": ((130, 50, 50), (160, 255, 255)),
    "white": ((0, 0, 200), (180, 30, 255))  
}


def find_resistor_bands(image):
    image = cv2.medianBlur(image, 15)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red bands
    mask1 = cv2.inRange(hsv_image, (0, 255, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv_image, (170, 182, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    cv2.imshow("Image", image)
    cv2.imshow("Red mask", red_mask)
    # End of Red Bands

    masks = {}
    masks["red"] = red_mask
    for color, (lower, upper) in color_ranges.items():
        masks[color] = cv2.inRange(hsv_image, lower, upper)
    
    contours_found = []

    # RED MASK
    contours, _ = cv2.findContours(masks["red"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 1000): 
            x, _, _, _ = cv2.boundingRect(contour)
            contours_found.append({'color': "red", 'area': area, 'x': x})
    # END OF RED MASK

    for color, (lower, upper) in color_ranges.items():
        cv2.imshow("Image", image)
        cv2.imshow(color, masks[color])
        contours, _ = cv2.findContours(masks[color], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if (area > 2000): 
                x, _, _, _ = cv2.boundingRect(contour)
                contours_found.append({'color': color, 'area': area, 'x': x})
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contours_found

def calculate_resistor(bands_found):

    sorted_data = sorted(bands_found, key=lambda item: item['x'])
    
    color_to_band1 = {"black": 0, "brown": 10, "red": 20, "orange": 30, "yellow": 40,
                      "green": 50, "blue": 60, "violet": 70, "grey": 80, "white": 90}
    color_to_band2 = {"black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
                      "green": 5, "blue": 6, "violet": 7, "grey": 8, "white": 9}
    color_to_multiplier = {"black": 1, "brown": 10, "red": 100, "orange": 1000,
                           "yellow": 10000, "green": 100000, "blue": 1000000}

    sorted_data = sorted(sorted_data, key=lambda item: item['x'])

    band1 = color_to_band1.get(sorted_data[0]['color'], 0)
    band2 = color_to_band2.get(sorted_data[1]['color'], 0)
    multiplier = color_to_multiplier.get(sorted_data[2]['color'], 1)

    result = (band1 + band2) * multiplier
    print("Resistor value: ", result)

    return result

def select_file_and_calculate():
    root = tk.Tk()
    root.withdraw()  

    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if not file_path:
        print("No file selected. Exiting.")
        return

    image = cv2.imread(file_path)
    
    if image is None:
        print("Error loading image!")
        return

    cv2.imshow("Selected Resistor Image", image)

    key = cv2.waitKey(0)
    
    if key == 27:  
        print("Exiting...")
    else:  
        input = preprocess_image(image)
        bands_found = find_resistor_bands(input)
        result = calculate_resistor(bands_found)
        print(f"Calculated Resistor Value: {result}")

def main():
    select_file_and_calculate()

if __name__ == "__main__":
    main()