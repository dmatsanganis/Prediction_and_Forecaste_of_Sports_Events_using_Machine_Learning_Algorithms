import math
import numpy as np

# Apply the hypothesis function (Wt*x).
def hypothesis(x, theta):

    h = np.transpose(np.array(theta)).dot(np.array(x))
    return h

# Find the cost function.    
def costFunction(theta, x, y):

    # Initialize variables.
    factor = 1 / len(x)
    sum = 0

    # A For-Loop Statement, in order 
    # to calculate the sum.
    for i in range(0,len(x)):
        sum += math.pow((hypothesis(x[i], theta) - y[i]), 2)    
    return factor * sum

# Implement the w(n)+2*p*Error*x
def learnThetaSingle(theta, x, y, alpha):
    return theta + 2*alpha * (y - hypothesis(x, theta)) * x

# Learn the rate function.
def learnTheta(theta, x, y):

    # Initialize variables.
    alpha = 1/len(x)
    temp_theta = theta
    bet = []
    current = []

    # Getting all Betting Sites.
    get_all_bet_sites = [0 for i in range(0,len(x))]

    # A For-Loop Statement, in which we create a table with all 
    # the best and the "1" at the end of it.
    # The "1" its to help for the cost calculation process.
    for i in range(0,len(x)):

        current = x[i][1], x[i][2], x[i][3],1
        current = np.array(current)
        get_all_bet_sites[i] = current

    # Initialize variables.
    matrix_weights = []
    costs =  []
    performance = []
    minimum = 1

    # Set the iterations to 10 (loops), 
    # in order to see if the cost decrease.
    iterations = 10 
    
    # Initialize Betting Site variable, j.
    j = 0

    # A While-Loop Statement, 
    # to check all 4 Betting Sites.
    while(j<4):

        # Initialize variable-counter.
        i = 0

        # A While-Loop Statement, 
        # as long as we have data (len(x)). 
        while(i<len(x)):
            
            # Put all the Odds into a variable and add the "1" at the end of it.
            bet = x[i][j*3+1], x[i][j*3+2], x[i][j*3+3], 1
            bet = np.array(bet)

            # Learn the new Weights.
            temp_theta = learnThetaSingle(temp_theta, bet, y[i], alpha)
            i+=1
        
        # Calculate the Cost, foreach Betting Site.
        cost = costFunction(temp_theta, get_all_bet_sites, y)
        costs.append(cost) 

        # In Case that the Cost increases the iteration stops 
        # and the program stores the current table, 
        # in order to get the smaller value.

        # Help for easier calculations.
        current = round(cost, 6)

        # If Statement, for the case that the Cost Decrease,
        # then we dont switch the Betting Site.
        if(current < minimum):
            minimum = current

        # Else, in case  that the Cost Increases the program will
        # save and store the current data (Weights, Costs etc.).   
        else:

            # Change iteration value, in order to see 
            # if the cost will decrease in the 
            # next (only the next one) iteration.
            iterations-=1

            # If iteration are over.
            if(iterations == 0):

                matrix_weights.append(temp_theta)
                temp_theta = [1,1,1,1]
                j+=1

                # Try/Catch Statement.
                try:
                    for i in range(0,len(x)):
                        get_all_bet_sites[i] = x[i][j*3+1], x[i][j*3+2], x[i][j*3+3],1
                    performance.append(costs[np.argmin(costs)])
                except:
                    performance.append(costs[np.argmin(costs)])

                # Setting variables.
                iterations = 10
                minimum = 1
                performance.append(costs[np.argmin(costs)])
                costs = []

    # Return the Weights'Table  from all 
    # the Betting Sites, with their costs.  
    return matrix_weights,performance

# Init Function.
def init(theta,x,y):

    # Create 3 Discriminant Functions, 
    # then we will ecide which one to keep.
    # The y (labels) foreach function.
    output = [[1,-1,-1],[-1,1,-1],[-1,-1,1]]

    # Initialize variables.
    new_matrix_weights = []
    new_performance = []
    new_y = [0 for i in range(0,len(y))]

    # A For-Loop Statement in order to change the tables,
    # when the function changes.
    for j in output:
        for i in range(0,len(y)):

            if(y[i][1] == '1'):
                new_y[i] = j[0]

            elif(y[i][1] == '0'):
                new_y[i] = j[1]

            else:
                new_y[i] = j[2]
        
        # The Least Mean Square Algorith will execute and check
        # all the Betting Sites at the same time.
        matrix_weights,performance = learnTheta(theta, x, new_y)

        # The new Weights after the Algorithm execution foreach Betting Site.
        new_matrix_weights.append(matrix_weights)

        # The Cost after the Algorithm execution foreach Betting Site.
        new_performance.append(performance)

    # Initialize Variables.
    minimum_performance = []
    final_weights = []
    final_performance = []
    right_output = []

    # A For-Loop for the 4 Betting Sites.
    for i in range(0,4):

        # Calculating the performance, foreach Betting Site.
        minimum_performance = new_performance[0][i], new_performance[1][i], new_performance[2][i]
        index = np.argmin(minimum_performance)
        right_output.append(index)

        # The Final Weight (based on smaller Cost, foreach Betting Site).
        final_weights.append(new_matrix_weights[index][i])
        # The Final Costs (the smallest ones).
        final_performance.append(new_performance[index][i])

    # The output that the program will use (keep) is the one with
    # the more correctly predicted matches.
    final_output = output[right_output[np.argmin(final_performance)]]
    
    return final_weights, final_performance, final_output

# Testing Function.
def test(theta,x,y,output_matrix):

    # Create a Table with the best output, that the program found.
    new_y = [0 for i in range(0,len(y))]

    # A For-Loop Statement, as long as 
    # the Output's Table has data availiable.
    for i in range(0,len(y)):

        if(y[i][1] == '1'):
            new_y[i] = output_matrix[0]

        elif(y[i][1] == '0'):
            new_y[i] = output_matrix[1]

        else:
            new_y[i] = output_matrix[2]

    # Initialize Wrongs Counter.
    wrongs_counter = 0
    # Initialize Rights Counter.
    rights_counter = 0

    # An Inner For-Loop fpr the 4 Betting Sites.
    for j in range(0,4):
        for i in range(0,len(x)):

            # Put all the Bets into a variable and add the "1" at the end of it.
            bet = x[i][j*3+1], x[i][j*3+2], x[i][j*3+3], 1

            # If the result is acceptable by the hypothetical function that we want, the count it as right.
            if(np.sign(hypothesis(theta,bet)) == new_y[i]):
                rights_counter+=1

            # Else If the result is not acceptable by the hypothetical function that we want, count it as wrong.
            else:
                wrongs_counter+=1
    
    # Create accuracy for the best Betting Site.
    accuracy = rights_counter/(wrongs_counter + rights_counter) * 100

    # Console's Message, regarding Algorithm Cost appears.
    # print('Cost: ' + str(performance[np.argmin(performance)]))

    # Console's Message, regarding the Wrong - Predicted Matches appears.
    print('Wrong - Predicted Matches: '+ str(wrongs_counter))
    # Console's Message, regarding the Right - Predicted Matches appears.
    print('Rights - Predicted Matches: '+ str(rights_counter))
    # Console's Message, regarding the Predicted Matches' Accuracy stat appears.
    print("Predicted Matches Accuracy: " + str(accuracy) + " %")

    # Returns the Right and Wrong - Predicted Matches and accuracy.
    return accuracy