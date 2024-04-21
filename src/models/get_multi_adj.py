
from asyncio import events
import numpy as np

def get_batch_adj(frame_prob, th, min_length,event_split=False,adj_mode='local'):
    
    bs_a_sb_list=[]  
    bs_v_sb_list=[]
    bs, t, _, num_classes = frame_prob.shape
    adj=np.zeros([bs,num_classes,2*t,2*t]) 

    for i in range(bs):
        adj_a, a_subgraph_list=get_adj(frame_prob[i,:,0,:], th=th, min_length=min_length, event_split=event_split, adj_mode=adj_mode) 
        adj_v, v_subgraph_list=get_adj(frame_prob[i,:,1,:], th=th, min_length=min_length, event_split=event_split, adj_mode=adj_mode)
        adj[i,:,:t,:t]=adj_a
        adj[i,:,t:,t:]=adj_v

        bs_a_sb_list.append(a_subgraph_list)
        bs_v_sb_list.append(v_subgraph_list)
        
    adj=adj.transpose(1,0,2,3)
    
    
    return adj, bs_a_sb_list, bs_v_sb_list

def get_adj(frame_prob, th=0.5, min_length=1, event_split=False, adj_mode='local'):
    
    t, num_classes=frame_prob.shape
    subgraph_list=[[] for _ in range(num_classes)]
    
    diag=np.ones(t)
    adj=np.diag(diag) 
    adj[1:,:-1]=adj[1:,:-1]+np.diag(diag[:-1]) 
    adj[:-1,1:]=adj[:-1,1:]+np.diag(diag[1:])  

    adj=np.expand_dims(adj,0)  
    adj = adj*np.ones((num_classes,1,1)) 

    if th == 'avg':
        avg_prob=np.mean(frame_prob,axis=0)
        th=np.mean(avg_prob)

    else:
        th=th 

    for i in range(num_classes): 
        if np.max(frame_prob[:,i])>=th:
            if adj_mode=='local':
                flag=0 
                for j in range(t):
                    if flag==0 and frame_prob[j,i]>=th:
                        start=j
                        flag=1
                    if flag==1:
                        if j>=t-1 and frame_prob[j,i]>=th:
                            end=t
                            flag=0 
                        elif frame_prob[j,i]<th:
                            end=j
                            flag=0

                        if flag==0 and end-start>=min_length:
                            subgraph_adj=local_adj(t, start, end)
                            
                            if event_split==True: 
                                adj[i]=np.where(adj[i]>subgraph_adj, adj[i], subgraph_adj) 
                            else: 
                                adj=np.where(adj>subgraph_adj, adj, subgraph_adj) 
                            
                            subgraph_list[i]+=list(range(start,end))

            elif adj_mode == 'global':

                garray = list(np.where(frame_prob[:,i]>th)[0]) 
                        
                subgraph_adj=global_adj(t, garray)
                            
                if event_split==True: 
                    adj[i]=np.where(adj[i]>subgraph_adj, adj[i], subgraph_adj) 
                else: 
                    adj=np.where(adj>subgraph_adj, adj, subgraph_adj) 
                
                subgraph_list[i] = garray
            
            else:
                raise NotImplementedError('illegal choice')
                
    return adj, subgraph_list

def global_adj(t, array):
    row=np.zeros([t,t])
    col=np.zeros([t,t])
    row[array,:]=1 
    col[:,array]=1
    local_adj=np.where(row>col,col,row) 
    return local_adj


def local_adj(t,start,end):

    row=np.zeros([t,t])
    col=np.zeros([t,t])
    row[start:end,:]=1 
    col[:,start:end]=1
    local_adj=np.where(row>col,col,row) 
    return local_adj


if __name__=='__main__':
    a=np.array([[1,2,3],
                [4,5,6]])

    print(np.max(a[0]))
    
