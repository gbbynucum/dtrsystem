# myapp/mtcnn_logic.py
import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from .models import Employee
from django.conf import settings

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# MLP Classifier for employee classification
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

class FaceRecognizer:
    def __init__(self):
        self.user_embeddings = {}
        self.mlp_clf = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.threshold = 0.8  # Confidence threshold for recognition
        self.transform = transforms.Compose([  # Define transform here
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.load_dataset()  # Load embeddings on init
        self.update_classifier()  # Train classifier on init

    def load_dataset(self):
        """Load employee images from the database and generate embeddings."""
        self.user_embeddings.clear()
        employees = Employee.objects.all()
        for employee in employees:
            user_embeddings = []
            for i in range(1, 11):
                image_field = f"image{i}"
                image = getattr(employee, image_field, None)
                if image and image.name:
                    try:
                        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
                        if not os.path.exists(image_path):
                            print(f"Image not found: {image_path}")
                            continue
                        img = Image.open(image_path).convert('RGB')
                        face_tensor = mtcnn(img)
                        if face_tensor is not None:
                            face_tensor = face_tensor.unsqueeze(0).to(device)
                            with torch.no_grad():
                                embedding = resnet(face_tensor).cpu().numpy()[0]
                            user_embeddings.append(embedding)
                        else:
                            print(f"No face detected in {image_path}")
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
            if user_embeddings:
                self.user_embeddings[employee.employee_id] = user_embeddings
            else:
                print(f"No valid embeddings for employee {employee.employee_id}")

    def update_classifier(self):
        """Train an MLP classifier based on loaded embeddings."""
        X = []
        y = []
        for employee_id, emb_list in self.user_embeddings.items():
            for emb in emb_list:
                X.append(emb)
                y.append(employee_id)
        
        if not X:
            print("No embeddings available to train classifier.")
            self.mlp_clf = None
            return
        
        unique_labels = sorted(list(set(y)))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        num_classes = len(unique_labels)
        
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        y_indices = [self.label_to_index[label] for label in y]
        y_tensor = torch.tensor(y_indices, dtype=torch.long).to(device)
        
        input_dim = X_tensor.shape[1]  # Should be 512 (FaceNet embedding size)
        self.mlp_clf = MLPClassifier(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.mlp_clf.parameters(), lr=0.001)
        
        epochs = 100
        self.mlp_clf.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.mlp_clf(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        self.mlp_clf.eval()
        print(f"Classifier trained with {num_classes} classes.")

    def recognize_face(self, frame):
        """Process a single frame and return recognized employee data with bounding box or None."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            boxes, probs = mtcnn.detect(pil_img)
            print(f"Detected boxes: {boxes}, Probabilities: {probs}")
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                face_crop = pil_img.crop((x1, y1, x2, y2))
                face_tensor = self.transform(face_crop).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy()[0]
                
                if self.mlp_clf is not None:
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = self.mlp_clf(embedding_tensor)
                    prob = torch.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(prob, dim=1)
                    print(f"Confidence: {confidence.item()}, Predicted index: {pred_idx.item()}")
                    
                    if confidence.item() >= self.threshold:
                        employee_id = self.index_to_label[pred_idx.item()]
                        try:
                            employee = Employee.objects.get(employee_id=employee_id)
                            result = {
                                'name': employee.name,
                                'department': employee.department,
                                'employee_id': employee.employee_id,
                                'confidence': confidence.item(),
                                'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                            }
                            print(f"Recognized employee: {result}")
                            return result
                        except Employee.DoesNotExist:
                            print(f"Employee with ID {employee_id} not found.")
                            return None
                    else:
                        print(f"Confidence {confidence.item()} below threshold {self.threshold}")
                else:
                    print("No classifier available for recognition.")
            else:
                print("No faces detected in frame.")
        except Exception as e:
            print(f"Error in recognize_face: {e}")
        return None

# Initialize the recognizer globally
recognizer = FaceRecognizer()