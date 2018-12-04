import numpy as np

##
#  Author : Jesus Antonio Valadez Flores
#  Date : 03/12/2018
#  Version : 1.0
#  Description : A simple Neural network emulating a XOR function
##

## Initialize a input x data

Set_X = np.array([[0,0], [0,1], [1,0], [1,1]])
print ("INPUT_X\n", Set_X)

## Initialize expected results from x data

Result_data = np.array([[0,1,1,0]]).T
print ("Expected_Result:\n", Result_data)

##Tabla de verdad
print ("Result:\n", np.append(Set_X, Result_data, axis = 1) )

## Activation function

def sigmoid(x, activationfunc = False):
    return ( x*(1-x) ) if (activationfunc) else ( 1 /( 1 + np.exp(-x) ))


##    if activationfunc:
##        return x*(1-x)
##    return 1/(1+np.exp(-x))

## Initialize seed in random

np.random.seed(0)

## Matrix between layer 0 and 1

Intermediate_0_1 = 2 * np.random.random((2,3)) - 1

Intermediate_1_2 = 2 * np.random.random((3,1)) - 1

## Learning function

for i in range(20000):
    ## Ponemos los datos de prueba en la entrada de la primera capa
    l0 = Set_X
    ## Obtenemos el producto punto de los inputs y los pesos,
    ## para después computar el resultado en la función de activación
    l1 = sigmoid(np.dot(l0, Intermediate_0_1))

    l2 = sigmoid(np.dot(l1, Intermediate_1_2))

    ## Calculamos el error de la capa final y la derivada multiplicada por el error
    l2_error = Result_data - l2
    l2_delta = l2_error * sigmoid(l2, True)
    

    ## Ahora computamos el error y derivada de la capa 1
    l1_error = np.dot(l2_delta, Intermediate_1_2.T)
    l1_delta = l1_error * sigmoid(l1, True)

    Intermediate_0_1 += np.dot(l0.T, l1_delta)

    Intermediate_1_2 += np.dot(l1.T, l2_delta)

    if(i % 1000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))) )

print (l2)
    
    

