# %%
import numpy as np

input_nodes = np.array([[0,0],[1,0],[0,1],[1,1]]).reshape(4,2)
output_nodes = np.array([0,1,1,0]).reshape(4,1)


# %%
class MultiLayer_Perceptron:
    import numpy as np
    
    def __init__(stuff, input_nodes, output_nodes, hidden_nodes,hidden_nodes2):
        stuff.in_nodes = input_nodes
        stuff.out_nodes = output_nodes
        stuff.hid_nodes = hidden_nodes
        stuff.hid_nodes2 = hidden_nodes2
        stuff.learning_rate = .1
        
        stuff.W_in2hid = stuff.np.random.uniform(-1,1,stuff.hid_nodes*(len(stuff.in_nodes.T)+1)).reshape(len(stuff.in_nodes.T)+1,stuff.hid_nodes)
        stuff.W_hid2hid = stuff.np.random.uniform(-1,1,stuff.hid_nodes2*(stuff.hid_nodes+1)).reshape(stuff.hid_nodes+1,stuff.hid_nodes2)
        stuff.W_hid2out = stuff.np.random.uniform(-1,1,(stuff.hid_nodes2+1)*stuff.out_nodes).reshape((stuff.hid_nodes2+1),output_nodes)
    
        stuff.Yhat = None
        stuff.Y1 = None
        stuff.Y2 = None
        
#/* ------------------------------------------------------------------------------------------------------------------------ */    
    def addBias(stuff,xs):
        return stuff.np.concatenate((np.ones(len(xs)).reshape(-1,1), xs), axis=1)
        
    def sigmoid(stuff,nput):
        return (1/(1+(stuff.np.exp(-nput))))
    
    def initial_weight(stuff,row,col):
        return stuff.np.random.uniform(-1,1,row*col).reshape(row,col)
    
    def get_me(stuff):
        return print("W_in2hid:\n\n\n",stuff.W_in2hid,"\n\n\n\nnW_hid2hid:\n\n\n",stuff.W_hid2hid,"\n\n\n\nnW_hid2out:\n\n\n",stuff.W_hid2out)

    def piece_of(stuff,obj):
        return obj*(1-obj)
    
    def score(stuff,Y,X):
        print("\n   Error:     ",round(stuff.np.sum((Y-X)**2),5))
        print("Accuracy:       ",round(1-stuff.np.mean(abs(stuff.np.round(X-Y))),5)*100)          
        return

#/* ------------------------------------------------------------------------------------------------------------------------ */        
#      METHOD: Feedforward 
#/* ------------------------------------------------------------------------------------------------------------------------ */    
    def feedforward(stuff,xs):
        
        # Check the input and change to array if not
        if type(xs) != type(stuff.np.array([])):
            xs = np.array(xs)
#        else:
            #print(\"Input has been checked for type and is an array.\")

        # input to hidden layer
        #       add the bias
        xs_wBias = stuff.addBias(xs)
        
        #       multi node's with weights
        hidden_nodes = stuff.sigmoid(stuff.np.matmul(xs_wBias,stuff.W_in2hid))
        stuff.Y2 = hidden_nodes
        
        # hidden layer to hidden layer
        #       add the bias
        hidden_nodes = stuff.addBias(hidden_nodes)
        
        #       multi node's with weights
        hidden_nodes = stuff.sigmoid(stuff.np.matmul(hidden_nodes,stuff.W_hid2hid))
        stuff.Y1 = hidden_nodes
        
        # hidden layer to output
        #       add the bias
        hidden_nodes = stuff.addBias(hidden_nodes)
        #       multi node's with weights
        hidden_nodes = stuff.sigmoid(stuff.np.matmul(hidden_nodes,stuff.W_hid2out))
        stuff.Yhat = hidden_nodes
        
        return hidden_nodes

#/* ------------------------------------------------------------------------------------------------------------------------ */         
#      METHOD: Backpropagation 
#/* ------------------------------------------------------------------------------------------------------------------------ */       
    def backprop(stuff,guess,actual,xs):
        
        # WEIGHT UPDATE: output to hidden layer
        #        Calculate the error related to the guess
        ERR_deltaG = (guess - actual)*(guess*(1-guess))
        
        #        Calculate the change in the weight
        deltaW_Y1 = 2*(np.matmul(ERR_deltaG.T,stuff.addBias(stuff.Y1)))
        hodl = 2*(np.matmul(ERR_deltaG,stuff.W_hid2out[1:].T))*stuff.piece_of(stuff.Y1)
    
        #        Calculate the update for weight: _hid2out
        stuff.W_hid2out = stuff.W_hid2out - deltaW_Y1.T*stuff.learning_rate # transposed?
        
        
        
        # WEIGHT UPDATE: hidden layer to hidden layer
        deltaW_Y2 = 2*(np.matmul(hodl.T,stuff.addBias(stuff.Y2)))
        hodl = 2*(np.matmul(hodl,stuff.W_hid2hid[1:].T))*stuff.piece_of(stuff.Y2)
    
        #        Calculate the update for weight: _hid2hid
        stuff.W_hid2hid = stuff.W_hid2hid - deltaW_Y2.T*stuff.learning_rate # transposed?
        
        
        
        # WEIGHT UPDATE: hidden layer to data layer
        deltaW_xs = 2*(np.matmul(hodl.T,stuff.addBias(xs)))
    
        #        Calculate the update for weight: _in2hid
        stuff.W_in2hid = stuff.W_in2hid - deltaW_xs.T*stuff.learning_rate # transposed?

                     
#        prnt_Ys = print(\"Y2\n\n\n",stuff.Y2, \"\n\n\n\nnY1\n\n\n",stuff.Y1, \"\n\n\n\nnYhat\n\n\n",stuff.Yhat)
#        prnt_W_Y1 = print(\"\n\n\n\nndeltaW_hid2out\n\n\n",deltaW_Y1.T,\"\n\n\n\nnupdate_W_hid2out\n\n\n",update_W_hid2out)
#        prnt_W_Y2 = print(\"\n\n\n\nndeltaW_hid2hid\n\n\n",deltaW_Y2.T,\"\n\n\n\nnupdate_W_hid2hid\n\n\n",update_W_hid2hid)
#        prnt_W_xs = print(\"\n\n\n\nndeltaW_in2hid\n\n\n",deltaW_xs.T,\"\n\n\n\nnupdate_W_in2hid\n\n\n",update_W_in2hid)
        return


# %%
test = MultiLayer_Perceptron(input_nodes,1,3,2)

print("The guess before the backprop step: \n",test.feedforward(input_nodes))

test.backprop(test.feedforward(input_nodes),output_nodes,input_nodes)

print("\nThe guess after the backprop step: \n",test.feedforward(input_nodes))

# %%
iters = 3100

for i in range(iters):
    gg = test.feedforward(input_nodes)
    test.backprop(gg,output_nodes,input_nodes)
    
print("Guess after", iters, "iterations\n",gg)

if round(gg[0][0],1) != 0:
    for i in range(iters):
        gg = test.feedforward(input_nodes)
        test.backprop(gg,output_nodes,input_nodes) 
    iters = 6200
    print("Guess after", iters, "iterations\n",gg)

if round(gg[3][0],1) != 0:
    for i in range(iters):
        gg = test.feedforward(input_nodes)
        test.backprop(gg,output_nodes,input_nodes) 
    iters = 9300    
    print("Guess after", iters, "iterations\n",gg)

# %%
RUN_TEST_INPUT = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2)
RUN_TEST_OUTPUT = np.array([0,1,1,0]).reshape(4,1)

print('In:\n',RUN_TEST_INPUT,'\n\nOut:\n',RUN_TEST_OUTPUT)
# %%

# %%
