Classroom Focus Analysis System v1.0
====================================

Overview
--------

The Classroom Focus Analysis System is a GUI application built with Python and Tkinter that analyzes student focus levels in classroom settings using computer vision and machine learning techniques. The system can process video feeds, images, or live camera input to evaluate student behavior and emotions, then visualize the results through various charts.
Features

--------

1. **Multi-Media Support**:
   * Process static images
   * Analyze video files
   * Capture live camera feed
2. **Analysis Capabilities**:
   * Behavior detection (raising hand, reading, writing)
   * Emotion recognition (sad, angry, fear, happy, neutral)
   * Focus score calculation
3. **Visualization Tools**:
   * Behavior-Emotion time series plot
   * Correlation matrix heatmap
   * Behavior-Emotion-Focus plot
   * Joint distribution plot
   * "Show All Plots" feature for comprehensive visualization
4. **Data Management**:
   * Save analysis results
   * Export visualization charts

### Dataset

The **Classroom Focus Analysis System** requires a dataset to perform behavior detection and emotion recognition. You can download the necessary datasets from the following links:

1. **Behavior Detection Dataset**: This dataset contains labeled data for detecting student behaviors like raising hands, reading, and writing.  
   [GitHub - Whiffe/SCB-dataset: Student Classroom Behavior dataset](https://github.com/Whiffe/SCB-dataset.git)

2. **Emotion Recognition Dataset**: This dataset is used for emotion classification (e.g., sad, angry, fear, happy, neutral).  
   [Visual Geometry Group - 牛津大学](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

Requirements
------------

To run this application, you'll need the following Python packages:
    tkinteropencv-pythonPillowpandasmatplotlibseabornnumpy
Installation

------------

1. Clone the repository:
    bashgit clone https://github.com/yangfsds-rgb/sy-tudent.git

2. Install required packages:
    bashpip install -r requirements.txt

3. Run the application:
    bashpython main_qt.py

Usage
-----

1. **Model Loading**:
   * Click "Load YOLOv9 Model" to load the detection model
2. **Media Source**:
   * Choose between "Open Image", "Open Video", or "Open Camera"
3. **Analysis**:
   * Start analysis with "Start Analysis" button
   * Stop analysis with "Stop Analysis" button
4. **Visualization**:
   * Use the visualization buttons to view different charts
   * Click "Show All Plots" to display all available visualizations
5. **Results**:
   * View analysis results in the text panel
   * Save results using "Save Results" button

Interface Layout
----------------

The application is divided into two main panels:

1. **Left Panel (Control Panel)**:
   * Model loading controls
   * Media source selection
   * Analysis controls
   * Results display
   * Visualization options
2. **Right Panel (Display Panel)**:
   * Video/image display area
   * Chart visualization frames

Contributing
------------

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a pull request

License
-------

This project is licensed under the MIT License. See the [LICENSE](https://yiyan.baidu.com/chat/LICENSE) file for details.
Acknowledgments

---------------

* Built with Python and Tkinter
* Leverages OpenCV for computer vision tasks
* Uses Matplotlib and Seaborn for data visualization
* Inspired by classroom behavior analysis research

Contact
-------

For questions or feedback, please contact the project maintainer at [mailto:your.email@example.com].

* * *

**Note**: This README assumes you have a `main.py` file that contains the application code shown in the provided code snippet. You may need to adjust the installation and usage instructions based on your actual project structure.
