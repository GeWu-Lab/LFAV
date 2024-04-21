
import numpy as np

def get_batch_adj(frame_prob, th, min_length,cross_modal=False):
    
    bs_a_sb_list=[]  
    bs_v_sb_list=[]
    bs, t, _, _ = frame_prob.shape
    adj=np.zeros([bs,2*t,2*t]) 

    for i in range(bs):
        adj_a, a_subgraph_list=get_adj(frame_prob[i,:,0,:], th=th, min_length=min_length) 
        adj_v, v_subgraph_list=get_adj(frame_prob[i,:,1,:], th=th, min_length=min_length)
        adj[i,:t,:t]=adj_a
        adj[i,t:,t:]=adj_v

        if cross_modal==True: 
            
            adj[i,:t,t:]=np.diag(np.ones(t)) 
            adj[i,t:,:t]=np.diag(np.ones(t)) 

        bs_a_sb_list.append(a_subgraph_list)
        bs_v_sb_list.append(v_subgraph_list)
    return adj, bs_a_sb_list, bs_v_sb_list

def get_adj(frame_prob, th=0.5, min_length=1):
    
    t, num_classes=frame_prob.shape
    subgraph_list=[[] for _ in range(num_classes)]
    
    diag=np.ones(t)
    adj=np.diag(diag) 
    adj[1:,:-1]=adj[1:,:-1]+np.diag(diag[:-1]) 
    adj[:-1,1:]=adj[:-1,1:]+np.diag(diag[1:])  

    if th == 'avg':
        avg_prob=np.mean(frame_prob,axis=0)
        th=np.mean(avg_prob)
    elif th=='ada':
        avg_prob=np.mean(frame_prob,axis=0)
        sort_prob=np.sort(avg_prob)
        th=sort_prob[-4]  
    else:
        th=th 

    for i in range(num_classes): 
        if np.max(frame_prob[:,i])>=th:
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
                        adj=np.where(adj>subgraph_adj, adj, subgraph_adj) 
                        subgraph_list[i]+=list(range(start,end))
    return adj, subgraph_list


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
    
