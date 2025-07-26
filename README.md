# 🧠 Eye Images Globe 🔍🌐

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/Mediapipe-Hand%20%26%20Face%20Mesh-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-RealTime-red)
![License](https://img.shields.io/badge/license-MIT-green)

> A hand-controlled, 3D floating eye images globe powered by real-time face mesh & gesture detection. Built with OpenCV + MediaPipe. Looks like a sci-fi neural scanner.

---

## 💡 What is this?

This project is an **interactive 3D eye visualization system**:

- Scans your face with a Green laser line 🟢
- Captures **30 right-eye images** dynamically 📸
- Maps them into a 3D spherical grid 🌐
- Lets you **rotate the globe using your bare hands** ✋

---

## 🧪 Demo

📽️ **Watch it in action**: [Instagram Reel](https://www.instagram.com/reel/DMftWVzySnK/)

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/izharashiq/Eye-Images-Globe.git
```

```bash
cd MatrixEyeGlobe
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run it!

```bash
python Image_Globe.py
```

---

## 🛠️ Built With

- **Python 3**
- **OpenCV**
- **MediaPipe FaceMesh + Hand Tracking**
- **Numpy**

---

## 🖥️ How it Works

1. **Phase 1**: Face scan begins with a green scan line.
2. **Phase 2**: If a face is detected, it captures 30 right-eye images over time.
3. **Phase 3**: Images float in 3D space, rotating based on your hand's orientation.
4. **Control**: Use your hand to rotate. Make a fist to pause rotation.

---

## 📸 Captures

All eye images are saved in the `eye_images/` folder as `.jpg` files locally.

---

### 📄 License

MIT License - see [LICENSE](https://github.com/izharashiq/Eye-Images-Globe/blob/main/LICENSE) file for details.

Free to use, modify, distribute.

### 👨‍💻 Author

Created by: [Github Profile](https://www.github.com/izharashiq)

🤝 Follow on [Instagram](https://www.instagram.com/i_izhar9?igsh=OWZ2MTZvbW9pbXE1)