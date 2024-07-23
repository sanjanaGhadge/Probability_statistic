import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# Function to accept height and weight values from the user
def get_height_weight():
    height = float(input("Enter the height (in cm): "))
    weight = float(input("Enter the weight (in kg): "))
    return height, weight
# Function to calculate the joint density function using KDE
def calculate_joint_density(heights, weights):
    data = np.vstack((heights, weights))
    return gaussian_kde(data)
# Function to calculate the marginal density function for height
def calculate_marginal_height_density(joint_density, height_range):
    return [joint_density.evaluate([h, 0])[0] for h in height_range]
# Function to calculate the marginal density function for weight
def calculate_marginal_weight_density(joint_density, weight_range):
    return [joint_density.evaluate([0, w])[0] for w in weight_range]
# Main function
def main():
    # Get the number of data points from the user
    num_data_points = int(input("Enter the number of data points: "))
    # Initialize lists to store height and weight data
    heights = []
    weights = []
    # Accept height and weight values from the user
    for i in range(num_data_points):
        print(f"\nEnter data point {i+1}:")
        height, weight = get_height_weight()
        heights.append(height)
        weights.append(weight)

    # Calculate the covariance matrix
    cov_matrix = np.cov(heights, weights)
    # Define range for plotting
    height_range = np.linspace(min(heights), max(heights), 100)
    weight_range = np.linspace(min(weights), max(weights), 100)
    # Calculate joint density function using KDE
    joint_density = calculate_joint_density(heights, weights)
    # Calculate marginal density functions
    marginal_height_density = calculate_marginal_height_density(joint_density, height_range)
    marginal_weight_density = calculate_marginal_weight_density(joint_density, weight_range)
    # Plot joint density function
    X, Y = np.meshgrid(height_range, weight_range)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(joint_density(positions).T, X.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.title('Joint Density Function of Height and Weight')
    plt.grid(True)
    # Plot marginal density functions
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(height_range, marginal_height_density, color='skyblue')
    plt.xlabel('Height (cm)')
    plt.ylabel('Density')
    plt.title('Marginal Density Function for Height')
    plt.subplot(1, 2, 2)
    plt.plot(weight_range, marginal_weight_density, color='salmon')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Density')
    plt.title('Marginal Density Function for Weight')
    plt.tight_layout()
    plt.show()
    print("\nCovariance Matrix:")
    print(cov_matrix)
if __name__ == "__main__":
    main()