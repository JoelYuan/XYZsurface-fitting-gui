# 3D Surface Fitting GUI

A Python desktop GUI application for fitting 3D scatter data (x, y, z) from Excel files using polynomial regression with dynamic clustering. The app provides interactive 3D visualization and displays polynomial formulas for each region, suitable for PLC or other programs.

## Features
- Load Excel files with x, y, z data (x: row 1, y: column A, z: matrix starting at B2).
- Perform surface fitting with dynamic clustering (1-10 regions) using DBSCAN and polynomial regression.
- Display interactive 3D scatter and surface plots using Plotly.
- Show polynomial formulas, x/y ranges, and errors in a copyable text box.
- Optimized for PLC applications with explicit polynomial formulas.

## Installation

pip install -r requirements.txt

2.Run the application:
bash

python src/main.py
Usage
1.Launch the app (

python src/main.py).
2.Click "选择Excel文件" to load an Excel file (see 

sample_data/data.xlsx for format).
3.Click "执行拟合" to run fitting.
4.View the interactive 3D plot in your browser.
5.Copy formulas and ranges from the right text box for PLC or other use.
Excel File Format
- Row 1 (B1 onward)

: x coordinates.
- Column A (A2 downward)

: y coordinates.
- Matrix (B2 onward)

: z values.
Example (sample_data/data.xlsx):

   |  A  |  B  |  C  |  D  |
---+-----+-----+-----+-----+
1  |     | 1.0 | 2.0 | 3.0 |
2  | 1.0 | 0.1 | 0.2 | 0.3 |
3  | 2.0 | 0.4 | 0.5 | 0.6 |
4  | 3.0 | 0.7 | 0.8 | 0.9 |
Dependencies
- Python 3.8+
- Libraries: numpy, pandas, scipy, scikit-learn, plotly, openpyxl


创建一个简单的Excel文件，结构如`README.md`中的示例，供用户测试。

---

使用说明

1. **运行程序**：
   - 确保安装依赖：`pip install -r requirements.txt`
   - 运行：`python src/main.py`
   - 界面显示，包含“选择Excel文件”和“执行拟合”按钮。

2. **操作流程**：
   - 点击“选择Excel文件”，选择包含x, y, z数据的Excel文件。
   - 点击“执行拟合”，程序运行三角化、聚类和拟合。
   - 结果：
     - **3D图**：在浏览器中打开Plotly交互式3D图，显示散点和分色曲面。
     - **文本框**：右侧显示簇数、每个区域的公式、x/y范围、训练/交叉验证误差，文本可复制。
   - 状态栏显示加载或拟合状态。

3. **PLC应用**：
   - 从文本框复制公式和范围，直接用于PLC代码。
   - 示例PLC逻辑：
     ```c
     if (x >= 1.0 && x <= 5.0 && y >= 1.0 && y <= 4.0) {
         z = 0.123 + 0.456*x + 0.789*y - 0.234*x*y + 0.567*x*x + 0.891*y*y;
     }
     ```

---