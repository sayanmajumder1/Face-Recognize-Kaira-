

---

# **🖥️ Face Recognize Kaira**  

> **An advanced face recognition system for real-time authentication and attendance tracking.**  

![Face Recognize Kaira](https://raw.githubusercontent.com/sayanmajumder1/Face-Recognize-Kaira-/master/s4.png)  

---

## 📖 **About Face Recognize Kaira**  

**Face Recognize Kaira** is a **real-time face recognition** system designed for **secure authentication and attendance logging**. Using **OpenCV, SVM, and deep learning techniques**, the system captures images, trains a model, and recognizes faces efficiently.  

🔹 **Real-time Face Recognition** using OpenCV & Machine Learning  
🔹 **Attendance Logging** in Excel format  
🔹 **User-Friendly UI** for better usability  
🔹 **Sci-Fi Themed Desktop Interface**  
🔹 **Secure Model Training and Recognition**  

---

## 📷 **Screenshots**  

| Capture Images | Training Model | Recognition Mode |
|---------------|---------------|------------------|
| ![Capture](https://raw.githubusercontent.com/sayanmajumder1/Face-Recognize-Kaira-/master/s1.png) | ![Training](https://raw.githubusercontent.com/sayanmajumder1/Face-Recognize-Kaira-/master/s2.png) | ![Recognition](https://raw.githubusercontent.com/sayanmajumder1/Face-Recognize-Kaira-/master/s3.png) |

---

## 🛠️ **Project Structure**  

```
Face-Recognize-Kaira/
│── dataset/                # Stores captured images categorized by person
│── models/                 # Trained model (face_recognition_model.pkl)
│── records/                # Attendance logs (Excel)
│── resources/              # Secret keys and additional resources
│── scripts/
│   ├── capture_images.py   # Script to capture and store images
│   ├── train_model.py      # Training script using SVM
│   ├── recognize_face.py   # Face recognition & attendance logging
│── ui/desktop/             # Desktop UI files
│── setup/                  # Installation setup scripts
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
```

---

## 🚀 **Features**  

✅ **Capture & Store Faces** – Add new users with multiple images  
✅ **Train & Optimize** – Uses **SVM** for accurate predictions  
✅ **Live Face Recognition** – Detect & authenticate users in real-time  
✅ **Excel-based Attendance** – Logs user details with timestamps  
✅ **Desktop GUI with Sci-Fi Theme** – Interactive and futuristic interface  

---

## 📥 **Installation Guide**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/sayanmajumder1/Face-Recognize-Kaira.git
cd Face-Recognize-Kaira
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Capture Images**  
Run the script to capture face images:  
```bash
python scripts/capture_images.py
```

### **4️⃣ Train the Model**  
Once images are captured, train the recognition model:  
```bash
python scripts/train_model.py
```

### **5️⃣ Start Face Recognition**  
Run the recognition script to authenticate users:  
```bash
python scripts/recognize_face.py
```

---

## 🖥️ **Technologies Used**  

- **Python 3.x**  
- **OpenCV** (Image Processing & Face Detection)  
- **scikit-learn (SVM)** (Face Classification)  
- **Pandas & NumPy** (Data Handling)  
- **Tkinter/PyQt** (Desktop UI)  

---

## 📊 **Attendance Logging**  

- Logs attendance in **Excel format** (`records/attendence_YYYY-MM-DD.xlsx`).  
- Records **User Name, Date, and Time** automatically.  

---

## 🎨 **UI Enhancement**  

- The **desktop version** has a **sci-fi themed UI** for an interactive experience.  
- Future updates will include **dark mode, animations, and futuristic elements**.  

---

## 🛠️ **Future Improvements**  

🚀 **Deep Learning Model** – Upgrade from **SVM** to **CNN-based face recognition**  
🔒 **Secure Access** – Implement **encryption & authentication** for better security  
📡 **Cloud Integration** – Store logs & models in the cloud for accessibility  
📱 **Mobile App Support** – Develop an **Android/iOS** version  

---

## 🤝 **Contributing**  

We welcome contributions! To contribute:  

1. **Fork** the repository  
2. **Clone** your fork  
3. **Create a new branch**  
4. **Commit your changes**  
5. **Push to GitHub & create a Pull Request**  

---

## 📬 **Contact & Support**  

📧 **Email:** sayanmajumder566@gmail.com  
🐱 **GitHub:** [sayanmajumder1](https://github.com/sayanmajumder1)  
🌐 **Website:** [My WebSite(https://sayanmajumder1.github.io/Sayanmajumder-Portfolio/)]  

📢 *If you find this project helpful, give it a ⭐ on GitHub!*  

