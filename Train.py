import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- Settings ----------
img_size = (100, 100)
class_labels = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
train_dir = r'C:\Users\Jana\Desktop\Fruits Classification\train'
test_dir  = r'C:\Users\Jana\Desktop\Fruits Classification\test'
epochs = 30

# ---------- Load and Preprocess Data ----------
datagen = ImageDataGenerator(rescale=1./255)

def extract_data_from_generator(generator):
    X, y = [], []
    for batch_x, batch_y in generator:
        for i in range(len(batch_x)):
            img = batch_x[i]
            X.append(img.flatten())
            y.append(np.argmax(batch_y[i]))
        if len(X) >= generator.samples:
            break
    return np.array(X), np.array(y)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=32, class_mode='categorical', shuffle=True
)

test_generator = datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=32, class_mode='categorical', shuffle=False
)

X_train, y_train = extract_data_from_generator(train_generator)
X_test, y_test = extract_data_from_generator(test_generator)

# ---------- Feature Scaling ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- Train SVM ----------
svm = SGDClassifier(loss='hinge', max_iter=1, tol=None, random_state=42)
classes = np.unique(y_train)

for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    svm.partial_fit(X_train_shuffled, y_train_shuffled, classes=classes)
    print(f"Epoch {epoch + 1}/{epochs} completed")

# ---------- Save Model ----------
joblib.dump(svm, 'svm_fruit_classifier.pkl')
joblib.dump(scaler, 'svm_scaler.pkl')
print("Model and scaler saved.")

# ---------- Predict & Evaluate ----------
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

# Print metrics
print(f"\n--- Evaluation Metrics ---")
print(f"Accuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")

# Save metrics report
report = classification_report(y_test, y_pred, target_names=class_labels)
with open("metrics_report.txt", "w") as f:
    f.write(f"Accuracy:  {accuracy:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1_score:.4f}\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(report)

print("Metrics report saved to metrics_report.txt")

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
plt.yticks(np.arange(len(class_labels)), class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
