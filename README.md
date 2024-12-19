# Automated Resistor Value Detection Using HSV-Based Color Segmentation and Edge Detection

A system for automating resistor value detection using computer vision techniques, including Gaussian blurring, Canny edge detection, and HSV-based color segmentation. The program simplifies resistor value identification by reducing human error and providing efficient, accurate results via a graphical user interface (GUI).

## Features
- **Automated Detection**: Identifies resistor values from images without manual computation.
- **Image Preprocessing**: Reduces noise using Gaussian and Median blurring.
- **Edge Detection**: Employs Canny edge detection and Hough Transform for isolating resistor boundaries.
- **HSV Color Segmentation**: Detects resistor bands by mapping colors to HSV ranges.
- **GUI**: Allows users to upload resistor images and view calculated values.

## Applications
- **Manufacturing**: Automates resistor inspection and classification during production.
- **Maintenance**: Reduces human error and labor costs in identifying resistor values.
- **Education**: Assists students in learning resistor decoding without manual charts or multimeters.

## System Overview
1. **Input**: Upload an image of the resistor.
2. **Preprocessing**: Apply noise reduction and edge detection.
3. **Color Segmentation**: Use HSV space to identify resistor bands.
4. **Value Computation**: Map detected colors to the resistor color code chart.
5. **Output**: Display the computed resistor value.

## Techniques Used
- **Gaussian and Median Blurring**: Remove noise while preserving edges.
- **Canny Edge Detection**: Identify resistor boundaries.
- **HSV Conversion**: Distinguish color bands with minimal lighting interference.
- **Color Masking**: Detect specific color ranges for band isolation.
- **Contours**: Identify and sort color bands for computation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pierre-parel/arvd.git
   ```
2. Install required python packages:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the application:
    ```sh
    python main.py
    ```
