import numpy as np

def newton_rahmpson_method(f, df, x0, tol1=1e-6, tol2=1e-6, max_iter=100):
    
    """
    
    Newton_method that solves f(x)= 0 for both scalar and vector valued functions
    
    -Parameters-
    
    f : function 
        The function f(x) whose root needs to be found
    df: Jacobian of f
        The Jacobian matrix of f(x) with respect to x, used for vector-valued functions
    x0 : initial guess for the solution
        Initial guess of the root
    tol1 : tolerance for the change in the guess of the root
        Tolerance for the relative change in x at which the algorithm is stopped
    tol2 : tolerance for the size of the residuum
        Tolerance for the magnitude of f(x) at which the algorithm is stopped
    max_iter : Maximum number of iterations 
        Maximum number of iterations for which the algorithm runs
    
    -Returns-
    
    x : float or numpy array 
        Estimate of the root of f(x)
    it : int
        Number of iterations used to find the root
    
    """
    
    x = x0  # Initial guess for the root
    it = 0  # Counter for number of iterations
    
    # Loop over maximum iterations to find the root
    for i in range(max_iter):
        
        # Evaluate the function f(x) and its Jacobian at current estimate x
        f_x = f(x)
        
        # Check if the function is scalar-valued or vector-valued
        if np.isscalar(f_x):
            
            # Scalar-valued function
            
            # Check if the residuum is smaller than the tolerance
            if abs(f_x) < tol2:
                return x, i+1
            
            # Compute the change in the guess of the root using Newton's method
            dx = f_x / df(x)
            
            # Update the guess of the root
            x = x - dx
            
            # Check stopping criteria
            if abs(dx) < tol1 and abs(f_x) < tol2:
                return x, i+1
            
        else:
            
            # Vector-valued function
            
            # Compute the Jacobian matrix of f(x) at current estimate x
            jac = df(x)
            
            # Compute the change in the guess of the root using Newton's method
            dx = np.linalg.solve(jac, -f_x)
            
            # Update the guess of the root
            x = x + dx
            
            # Increment the iteration counter
            it = it + 1
            
            # Check stopping criteria
            if np.linalg.norm(dx) < tol1 and np.linalg.norm(f_x) < tol2:
                return x, it+1
        
    # Return the final estimate of the root and the number of iterations used
    return x, it+1
