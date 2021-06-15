import math
import numpy as np
from random import shuffle, randint
from numpy.linalg import inv, solve, matrix_rank


# Train Function.
def train(x, y):

    # Initialize shape's variables.
    D = x.shape[1] + 1
    K = y.shape[1]

    # Initialize two matrices.
    sum1 = np.zeros((D, D))
    sum2 = np.zeros((D, K))

    # Initialize variable.
    i = 0

    # xi*xi and xi*yi, is based to the
    # Probability Density Function for "w".
    # A For - Loop Statement
    for x_i in x:

        # Add array with "1", to the vector.
        x_i = np.append(1, x_i)
        # Getting y array.
        y_i = y[i]

        # xi * xi_hat - simplyfied version of
        # the Function (P.P J(w) -> sxesi 3.43 apo to biblio).
        sum1 += np.outer(x_i, x_i)
        # xi * yi_hat - simplyfied version of
        # the Function (P.P J(w) -> sxesi 3.43 apo to biblio).
        sum2 += np.outer(x_i, y_i)

        # Increase the counter, i.
        i += 1

    # Weights' vector: Multiply the sum2 with the inversed sum1.
    # (sxesi 3.45 apo to biblio).
    fn = np.dot(inv(sum1), sum2)

    return fn

# Predict Function.
def predict(W, x):

    # Add the array of "1" ones,
    # in order to do the operation.
    x = np.append(1, x)

    # Solve the W_hat*x .
    values = list(np.dot(W.T, x))

    # Finds maximum value.
    winners = [i for i, x in enumerate(values) if x == max(values)]

    # Choose randomly if the result is "0" or "1" (the y value).
    index = randint(0, len(winners)-1)

    # Setting the new variable winner and the value of "i" and "x",
    # which are selected for the prediction.
    winner = winners[index]

    # Initialize a Zero Matrix.
    y = [0 for x in values]
    # Select value.
    y[winner] = 1

    return y

# Operation Function fixLabels
# (regarding outputs - labels of y).
def fixLabels(y):

    # Initialize an empty array, newY.
    newY = []

    # A For - Loop, for length of the Labels' (y's) Data.
    # Each list has the length of
    # the class with the bigger value.
    for i in range(len(y)):

        # If Home Team wins, put 1 to the
        # Betting Site's Odd for the home win
        # and 0 to draw and away win.
        if(y[i][1] == '1'):
            newY.append([1, 0, 0])

        # Else If the match ends as draw, put 1 to the
        # Betting Site's Odd for draw and
        # 0 to home and away wins.
        elif(y[i][1] == '0'):
            newY.append([0, 1, 0])

        # Else If Away Team wins, put 1 to the
        # Betting Site's Odd for the away win
        # and 0 to home win and draw.
        else:
            newY.append([0, 0, 1])

    return np.matrix(newY)

# Testing Function.
def test(a, b, c, d):

    # Create Weights' vector.
    W = train(a, b)

    # Create Training Fields (Synola Ekpaideusis).
    x = c
    y = d

    # Initialize variables.
    total = y.shape[0]
    i = 0
    hits = 0
    no_hits = 0

    # A For-Loop Statement that if "i" from the
    # "prediction" is equal to the value from
    # the "y", then the Accuracy is increased.
    # (If we predict correctly).
    for i in range(total):

        prediction = predict(W, x[i])
        actual = list(y[i].A1)

        # If accuracy hits correctly.
        if prediction == actual:
            hits += 1
            
        # Else the prediction is wrong.
        else:
            no_hits += 1

    # Then divide with its length and find the total Accuracy.
    accuracy = hits/(hits + no_hits) * 100

    # Console's Message, regarding the Predicted Matches' Accuracy stat appears.
    print("Wrong - Predicted Matches: " + str(no_hits))
    # Console's Message, regarding the Predicted Matches' Accuracy stat appears.
    print("Rights - Predicted Matches: " + str(hits))
    # Console's Message, regarding the Predicted Matches' Accuracy stat appears.
    print("Predicted Matches Accuracy: " + str(accuracy) + " %")

    return accuracy
    
# Main Least Square Function.
def main(x_train, y_train, x_test, y_test):

    # Console's Message, regarding the training process, appears.
    # print('Please wait. The Training Process has been started!')

    # Initialize Variables.
    data = x_train[:, 1:]
    classes = y_train

    # Convert input data to matrices, via numpy.
    x = np.matrix(data)
    # Reform y for the next operations.
    y = fixLabels(classes)
    # Convert input data to matrices, via numpy.
    c = np.matrix(x_test[:, 1:])
    # Reform y for the next operations.
    d = fixLabels(y_test)

    # Data Shuffle for better precision.
    # Initialize, a temp table, in which
    # we put both x abd y.
    z = []

    # Getting size.
    # size-1, because the
    # For-Loop starts ftom 0.
    size = x.shape[0] - 1

    # A For-Loop Statement, regarding size.
    for i in range(size):

        z.append((x[i], y[i]))

    # A For-Loop Statement, regarding size.
    for i in range(size):
        x[i] = z[i][0]
        y[i] = z[i][1]

    # Append the Training and Testing Data.
    accuracy = test(x, y, c, d)

    return accuracy

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
    for i in range(0, len(x)):
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
    get_all_bet_sites = [0 for i in range(0, len(x))]

    # A For-Loop Statement, in which we create a table with all
    # the best and the "1" at the end of it.
    # The "1" its to help for the cost calculation process.
    for i in range(0, len(x)):

        current = x[i][1], x[i][2], x[i][3], 1
        current = np.array(current)
        get_all_bet_sites[i] = current

    # Initialize variables.
    costs = []
    performance = []
    minimum = 1

    # Set the iterations to 10 (loops),
    # in order to see if the cost decrease.
    iterations = 10

    # Initialize Betting Site variable, j.
    j = 0

    # A While-Loop Statement,
    # to check all 4 Betting Sites.
    while(j < 4):

        # Initialize variable-counter.
        i = 0

        # A While-Loop Statement,
        # as long as we have data (len(x)).
        while(i < len(x)):

            # Put all the Odds into a variable and add the "1" at the end of it.
            bet = x[i][j*3+1], x[i][j*3+2], x[i][j*3+3], 1
            bet = np.array(bet)

            # Learn the new Weights.
            temp_theta = learnThetaSingle(temp_theta, bet, y[i], alpha)
            i += 1

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
            iterations -= 1

            # If iteration are over.
            if(iterations == 0):

                temp_theta = [1, 1, 1, 1]
                j += 1

                # Try/Catch Statement.
                try:
                    for i in range(0, len(x)):
                        get_all_bet_sites[i] = x[i][j*3 +
                                                    1], x[i][j*3+2], x[i][j*3+3], 1
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
    return performance

# Init Function.
def init(theta, x, y):

    # Create 3 Discriminant Functions,
    # then we will ecide which one to keep.
    # The y (labels) foreach function.
    output = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]

    # Initialize variables.
    new_performance = []
    new_y = [0 for i in range(0, len(y))]

    # A For-Loop Statement in order to change the tables,
    # when the function changes.
    for j in output:
        for i in range(0, len(y)):

            if(y[i][1] == '1'):
                new_y[i] = j[0]

            elif(y[i][1] == '0'):
                new_y[i] = j[1]

            else:
                new_y[i] = j[2]

        # The Least Square Algorith will execute and check
        # all the Betting Sites at the same time.
        performance = learnTheta(theta, x, new_y)

        # The Cost after the Algorithm execution foreach Betting Site.
        new_performance.append(performance)

    # Initialize Variables.
    minimum_performance = []
    final_performance = []

    # A For-Loop for the 4 Betting Sites.
    for i in range(0, 4):

        # Calculating the performance, foreach Betting Site.
        minimum_performance = new_performance[0][i], new_performance[1][i], new_performance[2][i]
        index = np.argmin(minimum_performance)

        # The Final Costs (the smallest ones).
        final_performance.append(new_performance[index][i])

    return final_performance