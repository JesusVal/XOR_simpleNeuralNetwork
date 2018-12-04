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

Intermediate_0_1 = 2 * np.random.random((2,1)) - 1

## Learning function

for i in range(20000):
    ## Ponemos los datos de prueba en la entrada de la primera capa
    l0 = Set_X
    ## Obtenemos el producto punto de los inputs y los pesos,
    ## para después computar el resultado en la función de activación
    l1 = sigmoid(np.dot(l0, Intermediate_0_1))

    ## Calculamos el error de la capa 1
    l1_error = Result_data - l1

    ## Derivada pultiplicada por el error
    l1_delta = l1_error * sigmoid(l1, True)

    Intermediate_0_1 += np.dot(l0.T, l1_delta)

    if(i % 1000) == 0:
        print ("Error:" + str(np.mean(np.abs(l1_error))) )

print (l1)
    
    

