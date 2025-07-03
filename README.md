# Highway Inspection and Maintenance Using AI

An AI-powered system designed for real-time inspection and maintenance monitoring of highways. This project aims to detect and segment damaged road signs, lane lines, and crash barriers, helping road authorities to identify and prioritize repair and maintenance activities.

## 🚀 Features

- Real-time object detection and segmentation
- Classification of road elements as **good** or **damaged**
- Handles road signs, lane markings, and crash barriers
- High accuracy (97%) on test data
- Supports video stream input for live monitoring
- Integrated with Streamlit for interactive dashboards
- Power BI integration for reporting and visualization

## 🛠️ Tech Stack

- Python
- OpenCV
- TensorFlow
- PyTorch
- YOLO (You Only Look Once) for object detection
- COCO JSON annotations with polygons for segmentation
- Streamlit (for deployment and user interface)
- Power BI (for data visualization)

## 📂 Dataset

- 6,000 images of road infrastructure
- Annotated using COCO JSON format
- Classes include:
  - Good lane line
  - Damaged lane line
  - Good crash barrier
  - Damaged crash barrier
  - Good road sign
  - Damaged road sign

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dayavallepu/highway-inspection-and-Maintenance.git
   cd highway-inspection-and-Maintenance
2. Install dependencies:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py

🧪 How it Works
Captures image or video frame
Runs YOLO-based detection model
Segments detected objects using polygon masks
Classifies objects as good or damaged
Streams results to the user interface

🚧 Future Work
Extend support to more road elements (e.g., potholes, guardrails)
Add predictive maintenance scheduling
Integrate IoT sensor data for enhanced anomaly detection
Deploy on embedded edge devices

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📜 License
This project is licensed under the MIT License. See LICENSE for details.

👤 Author
Dayakar Vallepu
