import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# --- Complexity Analysis (分析) ---
#
# Algorithm: K-Nearest Neighbors (KNN) Regressor
# N = number of training points (训练点数)
# D = number of dimensions/features (维度/特征数, 在此为 3)
# k = number of neighbors (邻居数)
# M = number of points to predict (预测点数)
#
# --- Training Phase (训练阶段) ---
# Time Complexity: O(1) or O(N*D*logN)
#   - KNN is a "lazy learner". In the brute-force approach, training is just O(1)
#     as it only stores the data.
#   - If a data structure like a KD-Tree is built to speed up prediction,
#     the training time complexity is O(N*D*logN). scikit-learn does this.
#   - KNN 是一个“懒惰学习者”。对于暴力法，训练仅是 O(1) 时间，因为它只存储数据。
#   - 如果为了加速预测而构建了像 KD-Tree 这样的数据结构，训练时间复杂度为 O(N*D*logN)。scikit-learn 会这样做。
#
# Space Complexity: O(N*D)
#   - The model needs to store the entire training dataset.
#   - 模型需要存储整个训练数据集。
#
# --- Prediction Phase (预测阶段) ---
# Time Complexity: O(M * k * logN) on average with KD-Tree
#   - For each of the M prediction points, the algorithm searches for its k nearest
#     neighbors in the training data. With a KD-Tree, this search is very efficient.
#   - 对于 M 个预测点中的每一个，算法都会在训练数据中搜索其 k 个最近的邻居。
#     使用 KD-Tree，这个搜索过程非常高效。
#
# Space Complexity: O(D)
#   - Space is needed to store the query points during prediction.
#   - 预测期间需要空间来存储查询点。
# =============================================================================


def generate_point_cloud_data(n_points=2000):
    """
    Generates a synthetic point cloud dataset from different shapes (a cube and a sphere).
    For each point, its coordinates (x,y,z) are the features, and its exact distance
    to the nearest boundary of its shape is the label.

    (中文: 生成一个包含不同形状（立方体和球体）的合成点云数据集。
     对于每个点，其坐标(x,y,z)是特征，而它到其形状最近边界的精确距离是标签。)
    """

    # --- Generate points in a cube (生成立方体内的点) ---
    cube_points = np.random.rand(n_points // 2, 3) * 2 - 1  # Cube from -1 to 1
    # Calculate distance to the 6 faces for each point
    dist_x = 1 - np.abs(cube_points[:, 0])
    dist_y = 1 - np.abs(cube_points[:, 1])
    dist_z = 1 - np.abs(cube_points[:, 2])
    # The true distance is the minimum of the distances to the faces
    cube_distances = np.min([dist_x, dist_y, dist_z], axis=0)

    # --- Generate points in a sphere (生成球体内的点) ---
    # To get uniform distribution, generate in a cube and reject points outside sphere
    sphere_points = []
    sphere_distances = []
    radius = 1.0
    while len(sphere_points) < n_points // 2:
        p = np.random.rand(1, 3) * 2 - 1  # Point in a [-1, 1] cube
        dist_from_center = np.linalg.norm(p)
        if dist_from_center <= radius:
            sphere_points.append(p[0])
            sphere_distances.append(radius - dist_from_center)

    sphere_points = np.array(sphere_points)
    sphere_distances = np.array(sphere_distances)

    # Combine the data from both shapes
    X = np.vstack([cube_points, sphere_points])
    y = np.concatenate([cube_distances, sphere_distances])

    return X, y


def run_prediction_test():
    """
    Main function to run the entire ML workflow:
    1. Generate data
    2. Split into training and testing sets
    3. Train a KNN Regressor model
    4. Make predictions
    5. Evaluate and print the results

    (中文: 主函数，运行整个机器学习流程：
     1. 生成数据
     2. 划分训练集和测试集
     3. 训练一个 KNN 回归模型
     4. 进行预测
     5. 评估并打印结果)
    """
    print("--- 1. Generating synthetic point cloud data... ---")
    X, y = generate_point_cloud_data(n_points=5000)
    print(f"Generated {X.shape[0]} total points.")

    print("\n--- 2. Splitting data into training and testing sets... ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    print("\n--- 3. Training the K-Nearest Neighbors Regressor model... ---")
    # We choose k=7 as a reasonable number of neighbors to consider
    knn_model = KNeighborsRegressor(n_neighbors=7)
    knn_model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n--- 4. Making predictions on the test set... ---")
    y_pred = knn_model.predict(X_test)
    print("Prediction complete.")

    print("\n--- 5. Evaluating model performance... ---")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print("A lower MSE indicates a better fit. An MSE close to 0 means the model's predictions are very accurate.")

    # --- Visualization (for local execution) ---
    # This part creates a 3D plot to visually compare true vs. predicted distances.
    # Note: In some environments, this plot might not show up automatically.
    print("\n--- 6. Visualizing results (optional)... ---")
    fig = plt.figure(figsize=(16, 8))

    # Plot 1: True Distances
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='viridis', s=10)
    ax1.set_title('Test Points Colored by True Distance')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    fig.colorbar(sc1, ax=ax1, label='Distance to Boundary')

    # Plot 2: Predicted Distances
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', s=10)
    ax2.set_title('Test Points Colored by Predicted Distance')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    fig.colorbar(sc2, ax=ax2, label='Distance to Boundary')

    plt.suptitle("Model Prediction vs. True Labels", fontsize=16)

    # To make this runnable in any session without GUI issues, we save the figure.
    output_filename = "prediction_visualization.png"
    plt.savefig(output_filename)
    print(f"Visualization saved to '{output_filename}'.")
    # In a local environment with a GUI, you could use plt.show()
    # plt.show()


# Run the entire process when the script is executed
if __name__ == "__main__":
    run_prediction_test()
