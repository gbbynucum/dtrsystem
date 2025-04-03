import sys
import os
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QWidget, QInputDialog, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


# python "c:\Users\Geb\Documents\VSC Projects\dtrsystem\myproject\myapp\mtcnn.py"

# Import FaceNet and MTCNN from facenet-pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1

DATASET_DIR = "dataset"

# Set up device and models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define a small MLP classifier for the embeddings
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Attendance System")
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            print(f"Created dataset folder at {DATASET_DIR}")
            
        # Load existing user data from disk (if any)
        self.user_embeddings = self.load_dataset()
        self.mlp_clf = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.threshold = 0.8       # Confidence threshold for softmax probability
        self.camera_active = False
        self.cap = None            # OpenCV video capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Define a transformation pipeline for face crops
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.setup_ui()
        self.update_classifier()   # Update classifier using loaded dataset

    def setup_ui(self):
        self.addUserButton = QPushButton("Add User")
        self.toggleCameraButton = QPushButton("Toggle Camera")
        self.retrainButton = QPushButton("Retrain Model")
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.addUserButton)
        layout.addWidget(self.toggleCameraButton)
        layout.addWidget(self.retrainButton)
        layout.addWidget(self.video_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.addUserButton.clicked.connect(self.add_user)
        self.toggleCameraButton.clicked.connect(self.toggle_camera)
        self.retrainButton.clicked.connect(self.retrain_model)

    def load_dataset(self):
        """
        Load the dataset from disk. For each subfolder (user) in the dataset folder,
        open each image, detect the face, compute the embedding, and add it to a dictionary.
        """
        embeddings_dict = {}
        if not os.path.exists(DATASET_DIR):
            return embeddings_dict
        for user in os.listdir(DATASET_DIR):
            user_dir = os.path.join(DATASET_DIR, user)
            if os.path.isdir(user_dir):
                user_embeddings = []
                for file in os.listdir(user_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(user_dir, file)
                        try:
                            img = Image.open(file_path).convert('RGB')
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                            continue
                        face_tensor = mtcnn(img)
                        if face_tensor is not None:
                            face_tensor = face_tensor.unsqueeze(0).to(device)
                            with torch.no_grad():
                                embedding = resnet(face_tensor)
                            embedding = embedding.cpu().numpy()[0]
                            user_embeddings.append(embedding)
                        else:
                            print(f"No face detected in {file_path}, skipping.")
                if user_embeddings:
                    embeddings_dict[user] = user_embeddings
        return embeddings_dict

    def add_user(self):
        name, ok = QInputDialog.getText(self, "Add User", "Enter the name of the user:")
        if ok and name:
            user_dir = os.path.join(DATASET_DIR, name)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
                print(f"Created user folder for {name} at {user_dir}")
            
            file_paths, _ = QFileDialog.getOpenFileNames(self, "Select 10 images", "", "Images (*.png *.jpg *.jpeg)")
            if len(file_paths) < 10:
                QMessageBox.warning(self, "Insufficient Images", "Please select at least 10 images.")
                return

            embeddings = []
            count = 0
            for file_path in file_paths:
                if count >= 10:
                    break
                try:
                    img = Image.open(file_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = resnet(face_tensor)
                    embedding = embedding.cpu().numpy()[0]
                    embeddings.append(embedding)
                    count += 1
                    ext = os.path.splitext(file_path)[1]
                    dest_path = os.path.join(user_dir, f"image_{count}{ext}")
                    shutil.copy(file_path, dest_path)
                    print(f"Copied {file_path} to {dest_path}")
                else:
                    print(f"No face detected in {file_path}, skipping.")
            
            if count < 10:
                QMessageBox.warning(self, "Insufficient Valid Images",
                                    "Could not detect faces in 10 images. Please try again.")
                return

            # Update in-memory data and classifier
            self.user_embeddings[name] = embeddings
            QMessageBox.information(self, "User Added", f"User {name} added successfully!")
            self.update_classifier()

    def update_classifier(self):
        # Gather embeddings and labels
        X = []
        y = []
        for user, emb_list in self.user_embeddings.items():
            for emb in emb_list:
                X.append(emb)
                y.append(user)
        if not X:
            self.mlp_clf = None
            print("No embeddings found. Classifier not updated.")
            return
        
        # Create label mapping
        unique_labels = sorted(list(set(y)))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        num_classes = len(unique_labels)
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        y_indices = [self.label_to_index[label] for label in y]
        y_tensor = torch.tensor(y_indices, dtype=torch.long).to(device)
        
        # Define the MLP classifier
        input_dim = X_tensor.shape[1]  # embedding dimension
        self.mlp_clf = MLPClassifier(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.mlp_clf.parameters(), lr=0.001)
        
        # Train the MLP classifier
        epochs = 100
        self.mlp_clf.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.mlp_clf(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        self.mlp_clf.eval()
        print("MLP classifier updated with {} samples.".format(len(X)))

    def retrain_model(self):
        # Reload the dataset from disk so that even previously saved data is used
        dataset_embeddings = self.load_dataset()
        if not dataset_embeddings:
            QMessageBox.information(self, "Retrain Model", "No user data found. Please add users first.")
            return
        self.user_embeddings = dataset_embeddings
        self.update_classifier()
        QMessageBox.information(self, "Retrain Model", "The model has been retrained with the current dataset.")

    def toggle_camera(self):
        if self.camera_active:
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_active = False
            self.toggleCameraButton.setText("Toggle Camera")
            self.video_label.setText("Camera Feed")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Camera Error", "Cannot open camera")
                return
            self.camera_active = True
            self.toggleCameraButton.setText("Stop Camera")
            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Get bounding boxes for faces in the image
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_crop = pil_img.crop((x1, y1, x2, y2))
                face_tensor = self.transform(face_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = resnet(face_tensor)
                embedding = embedding.cpu().numpy()[0]
                name = "Unknown"
                # Use the MLP classifier to predict the user
                if self.mlp_clf is not None:
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = self.mlp_clf(embedding_tensor)
                    prob = torch.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(prob, dim=1)
                    if confidence.item() >= self.threshold:
                        name = self.index_to_label[pred_idx.item()]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
