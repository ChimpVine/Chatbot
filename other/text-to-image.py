import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def plot_math_function(description, x_range=(-10, 10)):
    # Parse the function from the description using SymPy
    x = sp.symbols('x')
    try:
        function = sp.sympify(description)
    except sp.SympifyError:
        print("Invalid mathematical expression. Please try again.")
        return
    
    # Convert the SymPy function to a numerical function using lambdify
    func = sp.lambdify(x, function, 'numpy')
    
    # Generate x values
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = func(x_vals)
    
    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f'$y = {sp.latex(function)}$')
    plt.title('Graph of the function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()

# Main program
if __name__ == "__main__":
    description = input("Enter a mathematical function : ")
    plot_math_function(description)
