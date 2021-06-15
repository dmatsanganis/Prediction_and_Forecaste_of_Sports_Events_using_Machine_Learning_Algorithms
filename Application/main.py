import numpy as np
import Database_Configuration as dt
from KFold_CrossValidation import KFold_CrossValidation_Function 
from sklearn.model_selection import KFold

# A While-loop statement.
while True:

    # Dialog Menu to select action.
    print("\nSelect the action you want: ")
    dialog = input("a) Least Mean Square Algorithm.\nb) Least Square Algorithm.\nc) Multi-Layer Neural Network.\nd) Exit the program.\n\nSelect either a,b,c or d option: ").lower()

    # If user's response belongs to the 
    # aforementioned responses range. 
    if dialog in ["a", "b", "c", "d"]:

        # Response's code for a-option, Least Mean Squares Algorithm.
        if dialog == "a":

            # Initialize important variables, in order to utilize 
            # the Least Mean Squares Algorithm.
            import LeastMeanSquares as lms

            # Betting Odd's Table.
            x = np.array(dt.K)
            # Output - Label's Table.
            y = np.array(dt.Y)

            # Call the K-Fold Algorithm (10 Folds), 
            # with category equals to 1. 
            KFold_CrossValidation_Function(x, y, 1)

        # Response's code for b-option, Least Squares Algorithm.
        elif dialog == "b":

            # Initialize important variables, in order to utilize 
            # the Least Squares Algorithm.
            import LeastSquares as ls

            # Betting Odd's Table.
            x = np.array(dt.K)
            # Output - Label's Table.
            y = np.array(dt.Y)

            # Call the K-Fold Algorithm (10 Folds), 
            # with category equals to 2.  
            KFold_CrossValidation_Function(x, y, 2)
      
        # Response's code for c-option, Multi-Layer Neural Network.
        elif dialog == "c":

            # Initialize important variables, in order to utilize 
            # the Multi-Layer Neural Network.
            import MultiLayerNN as mnn

            # Getting the Multi-Layer Neural 
            # Network's data for the tables.
            x,y=dt.MNN_data()
            
            # Betting Odd's Table.
            x = np.array(x)
            # Output - Label's Table.
            y = np.array(y)

            # Call the K-Fold Algorithm (10 Folds), 
            # with category equals to 3. 
            KFold_CrossValidation_Function(x, y, 3)

        # Response's code for d-option (exit the program).
        elif dialog == "d":
            input("\nPress Enter to exit...")
            break

        else:
            # Console's message regarding Invalid Input Error.
            print("\nInvalid input, please try again.")    
    else:
        # Console's message regarding Invalid Input Error.
        print("\nInvalid input, please try again.") 