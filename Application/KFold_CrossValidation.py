import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Create the K-Fold Cross Validation Function (here 10-Fold).
def KFold_CrossValidation_Function(x, y, category):

    # The data set is split into a K number of sections/folds, 
    # here K = 10, in order to have the 10-Fold Cross Validation.
    kf = KFold(n_splits = 10)
    kf.get_n_splits(x)

    # Initialize the array with the 4-Betting sites.
    bet_sites = ['B365','BW','IW','LB']

    # Initialize accuracy_total array.
    accuracy_total = []
    accuracy = 0

    # Initialize the iteration_counter variable,
    # which keeps tracking of the iterations 
    # and their number.
    iteration_counter = 1

    # A For-Loop Statement, in order to get each of the 10 folds/partitions 
    # and present its results, depending on the category - user's choice.
    for train_index, test_index in kf.split(x):

        # Console's Message, regarding the iteration number, appears. 
        print('\nProcess: Iteration No: ' + str(iteration_counter))

        # Console's Message, regarding the training process, appears. 
        print('Please wait. The Training Process has been started!')

        # Training and Testing of the Betting Odd's Table.
        x_train, x_test = x[train_index], x[test_index]

        # Training and Testing of the Output - Label's Table.
        y_train, y_test = y[train_index], y[test_index]
        
        # If user select the first (a) option, 
        # the Least Mean Squares Algorithm.
        if(category == 1):

            # Initialize the theta array.
            theta=[1,1,1,1]

            # Initialize important variables, in order to utilize 
            # the Least Mean Squares Algorithm.
            import LeastMeanSquares as lms
            matrix_weights, performance, output = lms.init(theta, x_train, y_train)

            # Finds the Predicted Matches' Accuracy by the Algorithm.
            accuracy = lms.test(matrix_weights[np.argmin(performance)], x_test, y_test, output)

            # A For-Loop, in order to find the Most Accurate Betting site.
            for k in bet_sites:
                if (bet_sites.index(k) == np.argmin(performance)):
                    most_accurate_betting_site = k
                    break
            
            plot_colors = ['green','blue','red','brown']
            plt.bar(['B365','IW','BW','LB'], performance, color = plot_colors, width = 0.3)
            plt.title('Results for Iteration No: ' + str(iteration_counter), fontsize=14)
            plt.xlabel('Betting Sites', fontsize=14)
            plt.ylabel("Prediction's Accuracy", fontsize=14)
            plt.ylim([0.65, 0.78])
            plt.grid(False)                            
            plt.show()

            # If iteration_counter equals to 10.
            if (iteration_counter == 10):
                accuracy_total.append(accuracy)

                # On the 10th iteration print the 
                # total accuracy plot. 
                # Create the plot.
                plt.figure(figsize = (8, 6), dpi=80)
                plt.subplot(1, 1, 1)
                plt.plot([1,2,3,4,5,6,7,8,9,10], accuracy_total, color='orange', linewidth=4.0, linestyle='solid')
                plt.title('Least Mean Square Algorithm - Total Accuracy Results', fontsize=14)
                plt.xlabel('Iterations', fontsize=14)
                plt.ylabel("Model's Accuracy", fontsize=14)
                plt.legend(["Model's Accuracy"], loc = 'upper left')
                plt.ylim(60, 85)
                plt.grid(False)                            
                plt.show()

            # iteration_counter != 10.
            else: 
                accuracy_total.append(accuracy)

            # Console's Message, regarding the Most Accurate Betting Site appears.
            print('Most Accurate Betting Site: ' + str(most_accurate_betting_site))

            # Console's Message, regarding Algorithm Weights appears.
            print('Weights: ' + str(matrix_weights[np.argmin(performance)]))

            # Console's Message, regarding the iteration number, appears. 
            print('Completed: Iteration No: ' + str(iteration_counter))

        # Else If user select the second (b) option, 
        # the Least Squares Algorithm.   
        elif(category==2):

            # Initialize the theta array.
            theta=[1,1,1,1]
            
            # Initialize important variables, in order to utilize 
            # the Least Squares Algorithm.
            import LeastSquares as ls
            accuracy = ls.main(x_train,y_train,x_test,y_test)
            performance = ls.init(theta, x_train, y_train)
            
            # A For-Loop, in order to find the Most Accurate Betting site.
            for k in bet_sites:
                if (bet_sites.index(k) == np.argmin(performance)):
                    most_accurate_betting_site = k
                    break

            # If iteration_counter equals to 10.
            if (iteration_counter == 10):
                accuracy_total.append(accuracy)

                # On the 10th iteration print the 
                # total accuracy plot. 
                # Create the plot.
                plt.figure(figsize = (8, 6), dpi=80)
                plt.subplot(1, 1, 1)
                plt.plot([1,2,3,4,5,6,7,8,9,10], accuracy_total, color='cyan', linewidth=4.0, linestyle='solid')
                plt.title('Least Square Algorithm - Total Accuracy Results', fontsize=14)
                plt.xlabel('Iterations', fontsize=14)
                plt.ylabel("Model's Accuracy", fontsize=14)
                plt.legend(["Model's Accuracy"], loc = 'upper left')
                plt.ylim(40, 65)
                plt.grid(False)                            
                plt.show()

            # iteration_counter != 10.    
            else:
                accuracy_total.append(accuracy)
        
            # Console's Message, regarding the Most Accurate Betting Site appears.
            print('Most Accurate Betting Site: ' + str(most_accurate_betting_site))

            # Console's Message, regarding the iteration number, appears. 
            print('Completed: Iteration No: ' + str(iteration_counter))
            
        # Else If user select the third (c) option, 
        # the Multi-Layer Neural Network.  
        else:

            # Initialize important variables, in order to utilize 
            # the Multi-Layer Neural Network.
            import MultiLayerNN as mnn
            mnn.init(x_train,y_train,x_test,y_test)

        # Increase the iteration_counter, in order to get the next iteration.    
        iteration_counter += 1