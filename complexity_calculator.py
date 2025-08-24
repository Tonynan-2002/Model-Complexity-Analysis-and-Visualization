import matplotlib.pyplot as plt

def self_attention_complexity(L, D, h):
    """Calculates the complexity of the Self Attention module."""
    return 8 * L * D**2 + 4 * L**2 * D + 4 * h * L**2

def add_norm_1_complexity(L, D):
    """Calculates the complexity of the first Add & Norm module."""
    return 7 * L * D

def ffn_complexity(L, D, k):
    """Calculates the complexity of the FFN module."""
    return 4 * k * L * D**2 + k * L * D

def add_norm_2_complexity(L, D):
    """Calculates the complexity of the second Add & Norm module."""
    return 7 * L * D

def get_complexity_breakdown(L, D, k, h):
    """
    Calculates the computational complexity for each part of the model.

    Args:
        L (int): Sequence length.
        D (int): Hidden dimension.
        k (int): Intermediate dimension multiplier for FFN.
        h (int): Number of attention heads.

    Returns:
        dict: A dictionary containing the complexity of each module.
    """
    complexities = {
        "Self Attention": self_attention_complexity(L, D, h),
        "Add & Norm 1": add_norm_1_complexity(L, D),
        "FFN": ffn_complexity(L, D, k),
        "Add & Norm 2": add_norm_2_complexity(L, D),
    }
    return complexities

def plot_complexity_breakdown(complexities):
    """
    Plots the complexity breakdown as a pie chart.

    Args:
        complexities (dict): A dictionary with module names as keys and complexities as values.
    """
    labels = complexities.keys()
    sizes = complexities.values()
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Model Complexity Breakdown")
    plt.show()

def main(L, D, k, h):
    """
    Main function to calculate and print the complexity breakdown and total complexity.

    Args:
        L (int): Sequence length.
        D (int): Hidden dimension.
        k (int): Intermediate dimension multiplier for FFN.
        h (int): Number of attention heads.
    """
    
    # Get the breakdown of complexities
    breakdown = get_complexity_breakdown(L, D, k, h)
    
    # Calculate the total complexity
    total = sum(breakdown.values())
    
    # Print the results
    print("Complexity Breakdown:")
    for module, complexity in breakdown.items():
        print(f"- {module}: {complexity}")
        
    print(f"\nTotal Complexity: {total}")
    
    # Plot the breakdown
    plot_complexity_breakdown(breakdown)

# --- Example Usage ---
if __name__ == "__main__":
    # You can change these values to test with different model configurations
    L_val = 512  # Example Sequence Length
    D_val = 768  # Example Hidden Dimension
    k_val = 4    # Example FFN Multiplier
    h_val = 12   # Example Number of Heads
    
    main(L_val, D_val, k_val, h_val)