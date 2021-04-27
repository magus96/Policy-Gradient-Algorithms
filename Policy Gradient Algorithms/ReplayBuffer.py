#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
class ReplayBuffer(object):
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size=max_size
        self.mem_counter=0
        self.state_memory=np.zeros((self.mem_size,*input_shape))
        self.new_state_memory=np.zeros((self.mem_size,*input_shape))
        self.action_memory=np.zeros((self.mem_size,*n_actions))
        self.reward_memory=np.zeros(self.mem_size)
        self.terminal_memory=np.zeros(self.mem_size,dtype=np.float32)
                                    
    def store_transition(self,state,action,reward,state_,done):
        index=self.mem_counter&self.mem_size
        self.state_memory[index]=state
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.new_state_memory[index]=state_
        self.terminal_memory[index]=1-done
        self.mem_counter+=1
        
    def sample_buffer(self,batch_size):
        max_mem=min(self.mem_counter,self.mem_size)
        batch_random=np.random.choice(max_mem,batch_size)
        states=self.state_memory[batch_random]
        actions=self.action_memory[batch_random]
        rewards=self.reward_memory[batch_random]
        states_=self.new_state_memory[batch_random]
        terminals=self.terminal_memory[batch_random]
        
        return states,actions,rewards,states_,terminals  
        


# In[ ]:




