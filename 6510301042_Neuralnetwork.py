import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import Dense

# สร้างข้อมูล Data set A และ B
n_samples = 200
std_dev = 0.5  # ลดค่า cluster_std เพื่อลดความซ้อนทับของข้อมูล

# สร้างข้อมูลให้ห่างจากกันมากขึ้น
X, y = make_blobs(n_samples=n_samples, centers=[[-1, -1], [1, 1]], cluster_std=std_dev, random_state=42)

# แบ่งข้อมูล train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize ข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้าง Neural Network Model (โมเดลเรียบง่าย)
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),  # Hidden layer 1
    Dense(1, activation='sigmoid')  # Binary Classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model (เพิ่ม Epoch)
model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

# Plot Decision Boundary
def plot_decision_boundary(X, y, model, scaler):
    # กำหนดช่วงของ x, y ตามที่ต้องการ
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    
    # Create a meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Scale meshgrid ก่อนทำการพยากรณ์
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scaled_grid_points = scaler.transform(grid_points)
    
    # Predict for scaled meshgrid points
    # Z = model.predict(scaled_grid_points)
    # Z = (Z > 0.5).astype(int).reshape(xx.shape)  # ใช้ 0.5 เป็นเกณฑ์การตัดสินใจปกติ
  # แก้ไขการตัดสินใจเพื่อให้เส้นอยู่ฝั่งตรงข้าม
    Z = model.predict(scaled_grid_points)
    Z = (Z > 0.4).astype(int).reshape(xx.shape)  # การตัดสินใจ

# สลับด้านการตัดสินใจ
    Z = 1 - Z  # สลับผลการทำนายเพื่อให้เส้นแบ่งอยู่ในฝั่งตรงข้าม

    # สลับด้านการตัดสินใจ
    Z = 1 - Z  # สลับผลการทำนายเพื่อให้เส้นแบ่งอยู่ตรงข้าม

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.7, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap='coolwarm')
    
    # เส้นแบ่งกึ่งกลาง
    plt.contour(xx, xx, Z, levels=[0.5], colors='k', linewidths=1.5)
   
    # เปิดการแสดงตารางพร้อมปรับความอ่อน
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)

    # Add labels and legend
    plt.title("Decision Plane")
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class 1'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 2')
    ], loc='lower right')

    plt.show()

# Plot decision boundary
plot_decision_boundary(X, y, model, scaler)
