import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
from plotly.io import to_html
from threading import Thread
import webbrowser
import os

class SurfaceFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Surface Fitting GUI (Grid-based)")
        self.root.geometry("1200x800")
        
        # 布局框架
        self.frame_top = ttk.Frame(self.root)
        self.frame_top.pack(pady=10, fill="x")
        
        self.frame_plot = ttk.Frame(self.root)
        self.frame_plot.pack(pady=10, fill="both", expand=True)
        
        self.frame_text = ttk.Frame(self.root)
        self.frame_text.pack(pady=10, side="right", fill="y")
        
        # 顶部控件
        self.btn_load = ttk.Button(self.frame_top, text="选择Excel文件", command=self.load_file)
        self.btn_load.pack(side="left", padx=5)
        
        self.btn_fit = ttk.Button(self.frame_top, text="执行拟合", command=self.run_fitting, state="disabled")
        self.btn_fit.pack(side="left", padx=5)
        
        self.label_status = ttk.Label(self.frame_top, text="请加载Excel文件")
        self.label_status.pack(side="left", padx=5)
        
        # 文本框（右侧）
        self.text_output = scrolledtext.ScrolledText(self.frame_text, width=50, height=40, wrap=tk.WORD)
        self.text_output.pack(padx=10)
        self.text_output.insert(tk.END, "拟合结果将显示在这里...\n")
        self.text_output.config(state="normal")
        
        # 存储数据
        self.X = None
        self.Y = None
        self.Z = None
        self.file_path = None
        
    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if self.file_path:
            try:
                self.X, self.Y, self.Z = self.read_excel_data(self.file_path)
                self.label_status.config(text=f"已加载文件: {os.path.basename(self.file_path)}")
                self.btn_fit.config(state="normal")
            except Exception as e:
                self.label_status.config(text=f"加载失败: {str(e)}")
                self.btn_fit.config(state="disabled")
    
    def read_excel_data(self, file_path, sheet_name=0):
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        x = df.iloc[0, 1:].values.astype(float)
        y = df.iloc[1:, 0].values.astype(float)
        z = df.iloc[1:, 1:].values.astype(float)
        X, Y = np.meshgrid(x, y)
        return X, Y, z
    
    def compute_grid_features(self, Z, grid_size=(5, 5)):
        # Z: (m, n) matrix
        m, n = Z.shape
        gx, gy = grid_size
        features = []
        grid_labels = np.zeros((m, n), dtype=int)  # 初始子网格标签
        grid_idx = 0
        
        # 分割为gx×gy子网格
        for i in range(0, m, m//gx):
            for j in range(0, n, n//gy):
                i_end = min(i + m//gx, m)
                j_end = min(j + n//gy, n)
                sub_grid = Z[i:i_end, j:j_end]
                if sub_grid.size > 0:
                    # 计算特征
                    mean_z = np.mean(sub_grid)
                    var_z = np.var(sub_grid)
                    grad_x = np.mean(np.diff(sub_grid, axis=1)) if sub_grid.shape[1] > 1 else 0
                    grad_y = np.mean(np.diff(sub_grid, axis=0)) if sub_grid.shape[0] > 1 else 0
                    features.append([mean_z, var_z, grad_x, grad_y])
                    grid_labels[i:i_end, j:j_end] = grid_idx
                    grid_idx += 1
        
        features = np.array(features)
        scaler = StandardScaler()
        features = scaler.fit_transform(features) if features.size > 0 else features
        return features, grid_labels, gx, gy
    
    def cluster_regions(self, Z, points, max_clusters=10):
        features, grid_labels, gx, gy = self.compute_grid_features(Z)
        best_labels = grid_labels.ravel()
        best_n_clusters = len(np.unique(grid_labels))
        best_score = float('inf')
        
        # 尝试不同簇数
        for k in range(1, min(max_clusters + 1, len(features) + 1)):
            if len(features) >= k:
                kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
                cluster_labels = kmeans.labels_
                # 映射子网格标签到点
                point_labels = np.zeros(len(points), dtype=int)
                for idx, label in enumerate(grid_labels.ravel()):
                    point_labels[idx] = cluster_labels[label]
                
                error = 0
                n_clusters = len(np.unique(cluster_labels))
                for cluster in range(n_clusters):
                    cluster_points = points[point_labels == cluster]
                    if len(cluster_points) > 6:
                        error += self.fit_surface(cluster_points, degree=2)[1]
                if error < best_score and n_clusters <= max_clusters:
                    best_score = error
                    best_n_clusters = n_clusters
                    best_labels = point_labels
        
        return best_n_clusters, best_labels
    
    def fit_surface(self, points, degree=2):
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]
        X_train = np.vstack((X, Y)).T
        polyreg = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        cv_scores = cross_val_score(polyreg, X_train, Z, cv=3, scoring='neg_mean_squared_error')
        mse_cv = -cv_scores.mean()
        polyreg.fit(X_train, Z)
        Z_pred = polyreg.predict(X_train)
        mse = np.mean((Z_pred - Z) ** 2)
        poly = polyreg.named_steps['polynomialfeatures']
        coef = polyreg.named_steps['ridge'].coef_
        intercept = polyreg.named_steps['ridge'].intercept_
        terms = poly.powers_
        formula = f"z = {intercept:.3f}"
        for i, (power_x, power_y) in enumerate(terms):
            if abs(coef[i]) > 1e-6:
                term = f"{'+' if coef[i] > 0 else ''}{coef[i]:.3f}*x^{power_x}*y^{power_y}"
                formula += term.replace("^0", "").replace("*x^0*y^0", "").replace("*x^0", "").replace("*y^0", "")
        return formula, mse, mse_cv, polyreg
    
    def visualize(self, X, Y, Z, labels, formulas, points, ranges):
        fig = go.Figure()
        unique_labels = np.unique(labels[labels != -1])
        colors = [f'rgb({int(255*i/len(unique_labels))}, {int(255*(1-i/len(unique_labels)))}, 100)' for i in range(len(unique_labels))]
        
        for i, cluster in enumerate(unique_labels):
            mask = labels == cluster
            cluster_points = points[mask]
            fig.add_trace(go.Scatter3d(
                x=cluster_points[:, 0], y=cluster_points[:, 1], z=cluster_points[:, 2],
                mode='markers', marker=dict(size=5, color=colors[i], opacity=0.6),
                name=f'Cluster {i+1}'
            ))
            if len(cluster_points) > 6:
                formula, _, _, polyreg = self.fit_surface(cluster_points, degree=2)
                x_range, y_range = ranges[i]
                X_grid = np.linspace(x_range[0], x_range[1], 20)
                Y_grid = np.linspace(y_range[0], y_range[1], 20)
                X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
                Z_grid = polyreg.predict(np.vstack((X_grid.ravel(), Y_grid.ravel())).T).reshape(X_grid.shape)
                fig.add_trace(go.Surface(
                    x=X_grid, y=Y_grid, z=Z_grid, opacity=0.3, colorscale=[[0, colors[i]], [1, colors[i]]],
                    showscale=False, name=f'Surface {i+1}'
                ))
        
        fig.update_layout(
            title="3D Surface Fitting with Grid-based Clustering",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='auto'),
            showlegend=True
        )
        
        html_file = "plot.html"
        fig.write_html(html_file)
        webbrowser.open(f"file://{os.path.abspath(html_file)}")
    
    def run_fitting(self):
        self.label_status.config(text="正在拟合...")
        self.btn_fit.config(state="disabled")
        self.text_output.delete(1.0, tk.END)
        
        def fitting_thread():
            try:
                points = np.vstack((self.X.ravel(), self.Y.ravel(), self.Z.ravel())).T
                n_clusters, labels = self.cluster_regions(self.Z, points)
                self.text_output.insert(tk.END, f"Optimal number of clusters: {n_clusters}\n\n")
                
                formulas = []
                ranges = []
                for cluster in range(n_clusters):
                    cluster_points = points[labels == cluster]
                    if len(cluster_points) > 6:
                        formula, mse, mse_cv, _ = self.fit_surface(cluster_points, degree=2)
                        x_range = (cluster_points[:, 0].min(), cluster_points[:, 0].max())
                        y_range = (cluster_points[:, 1].min(), cluster_points[:, 1].max())
                        formulas.append(formula)
                        ranges.append((x_range, y_range))
                        self.text_output.insert(tk.END, f"Cluster {cluster + 1}:\n")
                        self.text_output.insert(tk.END, f"Formula: {formula}\n")
                        self.text_output.insert(tk.END, f"X range: {x_range}, Y range: {y_range}\n")
                        self.text_output.insert(tk.END, f"Mean Squared Error (train): {mse:.6f}\n")
                        self.text_output.insert(tk.END, f"Mean Squared Error (cross-val): {mse_cv:.6f}\n\n")
                
                self.visualize(self.X, self.Y, self.Z, labels, formulas, points, ranges)
                
                self.label_status.config(text="拟合完成")
                self.btn_fit.config(state="normal")
            except Exception as e:
                self.label_status.config(text=f"拟合失败: {str(e)}")
                self.btn_fit.config(state="normal")
        
        Thread(target=fitting_thread).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = SurfaceFittingApp(root)
    root.mainloop()