import sqlite3
import numpy as np
from datetime import datetime

# A function, which retrieves the data from the database.
def retrieve_data(ColumnNames,Table):
    list_col=[]
    conn = sqlite3.connect('database.sqlite')
    
    # SQL query to retrive properly the data.
    sqlquery="SELECT "
    for x in ColumnNames:
        if(x==ColumnNames[len(ColumnNames)-1]):
            sqlquery=sqlquery+x
        else:
            sqlquery=sqlquery+x+","  
 
    cursor = conn.execute(sqlquery+" FROM "+Table)
    rows = cursor.fetchall()
    
    for row in rows:
        list_col.append(row)
    
    return list_col

# Setting up custom arrays for specific data from th sqlite database.
goals = ['match_api_id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'date']

odds = ['match_api_id', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD','IWA', 'LBH', 'LBD', 'LBA']

odds_labels = ['team_api_id', 'buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing',
               'chanceCreationCrossing', 'chanceCreationShooting', 'defencePressure', 'defenceAggression', 'defenceTeamWidth', 'date']

# Odds' Table.
K = retrieve_data(odds, "Match")

# Retrieves the odds for 5000 matches (0 to 5000).
K = K[:5000]

# Initialize an empty array, A 
# to select the odds with 'none' value.
A = []

# An Inner For-loop to retrieve the odds and parse them into
# the aforementioned empty array, A.
for i in K:
    for j in i:
        # An if statement to reject the 'none' 
        # provided odds (A is the no-value odds array). 
        if(j is None):
            A.append(i)
            break

# A For-loop to remove the no-value 
# odds from the K array
for i in A:
    K.remove(i) 

# Delete A     
del A

# Print the K array - testing purposes.
# print(K)

# The table with the Goal Difference, in order 
# to find out which is the result of the match.
G = retrieve_data(goals,"Match")

# Retrieves the goals for 5000 matches (0 to 5000).
G = G[:5000]

results=[]

#------------------------------
# g[0]: match id.             |
# g[1]: home team.            |
# g[2]: away team.            |
# g[3]: home team's goals.    |
# g[4]: away team's goals.    |  
# g[5]: match's date.         |
#------------------------------

# A For-loop to get the match's result via Goal Difference.
for g in G:
    # Find Goal Difference as home - away team's goals.
    GD = g[3] - g[4]

    # If GD > 0, the home team goals are more than 
    # the away's, so home team wins.
    if(GD > 0):
        results.append([g[0],g[1],g[2],1,g[5]])

    # Else if GD = 0, the home team goals equals 
    # the away's, so match ends as a draw.    
    elif(GD == 0):
        results.append([g[0],g[1],g[2],0,g[5]])

    # Else (If GD < 0), the home team goals are  
    # less than the away's, so away team wins.    
    else:
        results.append([g[0],g[1],g[2],2,g[5]])

# Print the results array - testing purposes.
# print(results)

# An empty array Y to ensure that the 
# two-aforementioned table's data, are equal.
Y=[]

# An Inner For - loop to ensure that 
# the two table's data are even.
for i in K:
    for j in results:
        if(i[0]==j[0]):
            Y.append([i[0],j[3],j[1],j[2],j[4]])
            break

# Print the Y array - testing purposes.
# print(Y)        


# A Function for MLNN (3rd Subject), in order to 
# concarate the data from F and K tables, when 
# the option is selected.
def MNN_data():

    # Odds Labels' Table.
    F = retrieve_data(odds_labels,"Team_Attributes")
    teams_att = np.array(F)
    training_data = K
    target_data = np.array(Y)

    # Get the date.
    min_dt = datetime.strptime(teams_att[np.argmin(teams_att[:,9])][9], '%Y-%m-%d %H:%M:%S')

    # Console's Message.    
    print('Please wait. Setting up and Fixing some Data!')

    # Initialize array, trained_data_array for the trained data of the MLNN.
    trained_data_array = []

    # Initialize array, classified_results_array for the classified results 
    # and data to 3 specific classes [0,1,2]=[1,0,0],[0,1,0],[0,0,1].
    classified_results_array = []

    # Initialize a counter variable.
    counter = 0

    # Inner For-loops and if statements, in order to create the final two tables.
    for i in range(0,len(target_data)):
        dt2 = datetime.strptime(target_data[i][4], '%Y-%m-%d %H:%M:%S')
        counter = 0

        for j in range(0,len(teams_att)):
            if(min_dt.year > dt2.year or counter > 1):
                break

            # Get the date.    
            dt = datetime.strptime(teams_att[j][9], '%Y-%m-%d %H:%M:%S')

            if((teams_att[j][0] == target_data[i][2] or teams_att[j][0] == target_data[i][3]) and dt2.year == dt.year):
                counter += 1
                training_data[i] = np.concatenate(([training_data[i], teams_att[j][1:9]]), axis=None)

        # Depending the match's result (home, away win or draw), create a new 
        # table - array, in order to classify the data to 3 classes 
        # [0,1,2]=[1,0,0],[0,1,0],[0,0,1 (one-hot encode method).
        if(len(training_data[i]) > 21):

            # Append data to the trained_data_array. 
            trained_data_array.append(training_data[i][1:])
            
            # Home team win.
            if(target_data[i][1] == '1'):
                classified_results_array.append([1,0,0])
            # Draw.    
            elif(target_data[i][1] == '0'):
                classified_results_array.append([0,1,0])
            # Away team win.    
            else:
                classified_results_array.append([0,0,1])

    # Returns the tables, in order to get handled from the main.py .
    return trained_data_array, classified_results_array