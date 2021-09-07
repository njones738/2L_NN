# %%
class MultiLayer_Perceptron:
    import numpy as np
    from sklearn.utils import shuffle
    import seaborn as sns
    from matplotlib import pyplot as plt

    
    def __init__(stuff, input_nodes, output_nodes, hidden_nodes=None,hidden_nodes2=None):  # Other parameters related 
        stuff.in_nodes = input_nodes                                                       # to modeling are in the fit
        stuff.out_nodes = output_nodes                                                     # method.
        stuff.hid_nodes = hidden_nodes
        stuff.hid_nodes2 = hidden_nodes2
                
        if stuff.hid_nodes == None:         # IF the first hidden node parameter is None, THEN continue to line 15. ELSE line 28.
            if stuff.hid_nodes2 == None:    # IF the second hidden node parameter is None, THEN continue to line 16. ELSE line 20.
                stuff.layrs = 0  
                stuff.W_hid2out = stuff.np.random.uniform(-1,1,
                                                         stuff.out_nodes*(len(stuff.in_nodes.T)+1)
                                                        ).reshape((len(stuff.in_nodes.T)+1),stuff.out_nodes)
            elif stuff.hid_nodes2 != None:  # IF the second hidden node parameter is NOT None, THEN continue to line 21.
                stuff.layrs = 1
                stuff.W_hid2hid = stuff.np.random.uniform(-1,1,
                                                          stuff.hid_nodes2*(len(stuff.in_nodes.T)+1)
                                                         ).reshape(len(stuff.in_nodes.T)+1,stuff.hid_nodes2)
                stuff.W_hid2out = stuff.np.random.uniform(-1,1,
                                                          stuff.out_nodes*(stuff.hid_nodes2+1)
                                                         ).reshape((stuff.hid_nodes2+1),stuff.out_nodes)
        if stuff.hid_nodes != None:         # IF the first hidden node parameter is NOT None, THEN continue to line 29.
            if stuff.hid_nodes2 != None:    # IF the second hidden node parameter is NOT None, THEN continue to line 30. ELSE line 40.
                stuff.layrs = 2
                stuff.W_in2hid = stuff.np.random.uniform(-1,1,
                                                         stuff.hid_nodes*(len(stuff.in_nodes.T)+1)
                                                        ).reshape((len(stuff.in_nodes.T)+1),stuff.hid_nodes)
                stuff.W_hid2hid = stuff.np.random.uniform(-1,1,
                                                          stuff.hid_nodes2*(stuff.hid_nodes+1)
                                                         ).reshape((stuff.hid_nodes+1),stuff.hid_nodes2)
                stuff.W_hid2out = stuff.np.random.uniform(-1,1,
                                                          stuff.out_nodes*(stuff.hid_nodes2+1)
                                                         ).reshape((stuff.hid_nodes2+1),stuff.out_nodes)
            elif stuff.hid_nodes2 == None:  # IF the first hidden node parameter is None, THEN continue to line 41.
                stuff.layrs = 1  
                stuff.W_hid2hid = stuff.np.random.uniform(-1,1,
                                                          stuff.hid_nodes*(len(stuff.in_nodes.T)+1)
                                                         ).reshape(len(stuff.in_nodes.T)+1,stuff.hid_nodes)
                stuff.W_hid2out = stuff.np.random.uniform(-1,1,
                                                          stuff.out_nodes*(stuff.hid_nodes+1)
                                                         ).reshape((stuff.hid_nodes+1),stuff.out_nodes)
    
#/* ------------------------------------------------------------------------------------------------------------------------ */    
    def addBias(stuff,xs):
        return stuff.np.concatenate((np.ones(len(xs)).reshape(-1,1), xs), axis=1)
        
    def sigmoid(stuff,nput):
        return (1/(1+(stuff.np.exp(-nput))))
    
    def piece_of(stuff,obj):
        return obj*(1-obj)
    
    def _nextBatch2(stuff, y):
        from sklearn.utils import shuffle
        X, y = shuffle(stuff.in_nodes, y, random_state=None)
        for i in stuff.np.arange(0, X.shape[0], stuff.btch_sze):
            yield (X[i:i + stuff.btch_sze], y[i:i + stuff.btch_sze]) 

    def get_numOF_layers(stuff,p=0):
        stuff.get_weights(p)
        return stuff.layrs
    
#/* ------------------------------------------------------------------------------------------------------------------------ */         
#      METHOD: score 
#/* ------------------------------------------------------------------------------------------------------------------------ */       
    def score(stuff,Y,X):
        Y = Y.reshape(-1,1)
        print("\n   Error:     ",round(stuff.np.sum((Y-stuff.feedforward(X))**2),5))
        print("Accuracy:       ",round(1-stuff.np.mean(abs(stuff.np.round(stuff.feedforward(X)-Y))),5)*100)
        
#/* ------------------------------------------------------------------------------------------------------------------------ */        
#      METHOD: Feedforward & Backpropagation
#/* ------------------------------------------------------------------------------------------------------------------------ */    
    def feedforward(stuff,xs):

        if stuff.layrs == 0:
            stuff.Yhat = stuff.sigmoid(stuff.np.matmul(stuff.addBias(xs),stuff.W_hid2out))       # hidden layer to output        
        if stuff.layrs == 1:
            stuff.Y1 = stuff.sigmoid(stuff.np.matmul(stuff.addBias(xs),stuff.W_hid2hid))         # hidden layer to hidden layer
            stuff.Yhat = stuff.sigmoid(stuff.np.matmul(stuff.addBias(stuff.Y1),stuff.W_hid2out)) # hidden layer to output
        if stuff.layrs == 2:          
            stuff.Y2 = stuff.sigmoid(stuff.np.matmul(stuff.addBias(xs),stuff.W_in2hid))          # input to hidden layer
            stuff.Y1 = stuff.sigmoid(stuff.np.matmul(stuff.addBias(stuff.Y2),stuff.W_hid2hid))   # hidden layer to hidden layer
            stuff.Yhat = stuff.sigmoid(stuff.np.matmul(stuff.addBias(stuff.Y1),stuff.W_hid2out)) # hidden layer to output
        return stuff.Yhat
    
    def backprop(stuff,guess,actual,xs):
        
        ERR_deltaG = (guess - actual)*guess*(1-guess)                                           # Compute the initial error
        
# FOR zero layer
        if stuff.layrs == 0:
          # WEIGHT UPDATE: output to data layer
            deltaW_xs = 2*(stuff.np.matmul(ERR_deltaG.T,stuff.addBias(xs)))                     # Compute the gradient
            stuff.W_hid2out = stuff.W_hid2out - deltaW_xs.T*stuff.learning_rate                 # Update the weights: hid2out     
        
# FOR one layer
        if stuff.layrs == 1:
          # WEIGHT UPDATE: output to hidden layer
            deltaW_Y1 = 2*(stuff.np.matmul(ERR_deltaG.T,stuff.addBias(stuff.Y1)))               # Compute the gradient
            stuff.W_hid2out = stuff.W_hid2out - deltaW_Y1.T*stuff.learning_rate                 # Update the weights: hid2out     
            hodl = (stuff.np.matmul(ERR_deltaG,stuff.W_hid2out[1:].T))*stuff.piece_of(stuff.Y1) # Compute link: _hid2out

          # WEIGHT UPDATE: hidden layer to data layer
            deltaW_xs = (stuff.np.matmul(hodl.T,stuff.addBias(xs)))                             # Compute the gradient
            stuff.W_hid2hid = stuff.W_hid2hid - deltaW_xs.T*stuff.learning_rate                 # Update the weights: hid2hid
        
# FOR two layer
        if stuff.layrs == 2:
          # WEIGHT UPDATE: output to hidden layer
            deltaW_Y1 = 2*(stuff.np.matmul(ERR_deltaG.T,stuff.addBias(stuff.Y1)))               # Compute the gradient
            stuff.W_hid2out = stuff.W_hid2out - deltaW_Y1.T*stuff.learning_rate                 # Update the weights: hid2out     
            hodl = (stuff.np.matmul(ERR_deltaG,stuff.W_hid2out[1:].T))*stuff.piece_of(stuff.Y1) # Compute link: _hid2out
    
          # WEIGHT UPDATE: hidden layer to hidden layer
            deltaW_Y2 = (stuff.np.matmul(hodl.T,stuff.addBias(stuff.Y2)))                       # Compute the gradient
            stuff.W_hid2hid = stuff.W_hid2hid - deltaW_Y2.T*stuff.learning_rate                 # Update the weights: hid2hid
            hodl = (stuff.np.matmul(hodl,stuff.W_hid2hid[1:].T))*stuff.piece_of(stuff.Y2)       # Compute link: _hid2hid
       
          # WEIGHT UPDATE: hidden layer to data layer
            deltaW_xs = (stuff.np.matmul(hodl.T,stuff.addBias(xs)))                             # Compute the gradient
            stuff.W_in2hid = stuff.W_in2hid - deltaW_xs.T*stuff.learning_rate                   # Update the weights: in2hid

#/* ------------------------------------------------------------------------------------------------------------------------ */         
#      METHOD: Predict 
#/* ------------------------------------------------------------------------------------------------------------------------ */       
    def predict(stuff,xs):
        return stuff.feedforward(xs)
    
#/* ------------------------------------------------------------------------------------------------------------------------ */         
#      METHOD: save & get weights 
#/* ------------------------------------------------------------------------------------------------------------------------ */       
    def get_weights(stuff,prnt=0):
        if stuff.layrs == 0:
            if prnt == 1:
                print("W_hid2out:\n\n\n",stuff.W_hid2out)
            return stuff.W_hid2out
        elif stuff.layrs == 1:
            if prnt == 1:
                print("W_hid2hid:\n\n\n",stuff.W_hid2hid,
                      "\n\n\n\nW_hid2out:\n\n\n",stuff.W_hid2out)        
            return stuff.W_hid2hid, stuff.W_hid2out
        elif stuff.layrs == 2:
            if prnt == 1:
                print("W_in2hid:\n\n\n",stuff.W_in2hid,
                      "\n\n\n\nW_hid2hid:\n\n\n",stuff.W_hid2hid,
                      "\n\n\n\nW_hid2out:\n\n\n",stuff.W_hid2out)        
            return stuff.W_in2hid, stuff.W_hid2hid, stuff.W_hid2out
        
    def save_the_weight(stuff):
        stuff.np.savetxt('current_weights_in2hid.csv', stuff.W_in2hid, delimiter=',')
        stuff.np.savetxt('current_weights_hid2hid.csv', stuff.W_hid2hid, delimiter=',')
        stuff.np.savetxt('current_weights_hid2out.csv', stuff.W_hid2out, delimiter=',')
        return print("weights are saved",stuff.get_weights())    

#/* ------------------------------------------------------------------------------------------------------------------------ */         
#      METHOD: fit 
#/* ------------------------------------------------------------------------------------------------------------------------ */       
    def fit(stuff,Y,lr=.01,ep_ch=10,batch_size=8,prt_frq=1):
        stuff.learning_rate = lr
        stuff.btch_sze = batch_size
        stuff.print_freq = prt_frq
        stuff.ep_ch = ep_ch
        Y = Y.reshape(-1,1)
        
        for epoch in range(int(ep_ch/prt_frq)):
            for prt_freq in range(prt_frq):
                for x_batch, y_batch in stuff._nextBatch2(Y):
                    stuff.backprop(stuff.feedforward(x_batch),y_batch,x_batch)
            stuff.score(Y,stuff.in_nodes)
# %%
