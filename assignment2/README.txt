ASSIGNMENT 2
=======

Code is located at https://github.com/emthomas/CS7641/tree/master/assignment2

# Requirements
* jython
* java
* ABAGAIL

# To Run
Execute the `run.sh` in the jython folder. The contents is located below. 

  ```
  #!/bin/bash
  # edit the classpath to to the location of your ABAGAIL jar file
  #
  export CLASSPATH=../ABAGAIL.jar:$CLASSPATH

  # four peaks
  echo "four peaks"
  jython fourpeaks.py

  # count ones
  echo "count ones"
  jython countones.py

  # continuous peaks
  echo "continuous peaks"
  jython continuouspeaks.py

  # knapsack
  echo "Running knapsack"
  jython knapsack.py

  # abalone test
  echo "Running abalone test"
  jython abalone_test.py

  # traveling salesman
  echo "Running traveling salesman test"
  jython travelingsalesman.py
  ```
