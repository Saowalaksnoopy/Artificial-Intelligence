import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# สร้าง DataFrame 
data = {
    "Temperature": [29, 28, 34, 31, 25],
    "Orders": [77, 62, 93, 84, 59]
}
df = dd.from_pandas(pd.DataFrame(data), npartitions=1)

# คำนวณค่าเฉลี่ยของ X และ Y
x_mean = df["Temperature"].mean().compute()
y_mean = df["Orders"].mean().compute()

# คำนวณค่า Sxx และ Sxy
df["x_diff"] = df["Temperature"] - x_mean
df["y_diff"] = df["Orders"] - y_mean
df["x_diff_sq"] = df["x_diff"] ** 2
df["xy_diff"] = df["x_diff"] * df["y_diff"]

Sxx = df["x_diff_sq"].sum().compute()
Sxy = df["xy_diff"].sum().compute()

# คำนวณค่าสมการเส้นตรง
a = Sxy / Sxx
b = y_mean - a * x_mean

# สร้างกราฟ
plt.figure(figsize=(8, 6))

# จุดข้อมูลต้นฉบับ
x = np.array(data["Temperature"])
y = np.array(data["Orders"])
plt.scatter(x, y, color="blue", label="Data Points")

# สมการเส้นตรง
x_line = np.linspace(min(x), max(x), 100)
y_line = a * x_line + b
plt.plot(x_line, y_line, color="red", label=f"y = {a:.3f}x{b:.3f}")

# เพิ่มรายละเอียดกราฟ
plt.title("Linear Regression")
plt.legend()
plt.grid(True)

# แสดงกราฟ
plt.show()
