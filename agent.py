import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
import random
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import spotipy


cs ="CS_SPOTIFY_API"
cid = "CID_SPOTIFY_ID"



client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=cs)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_playlist_tracks(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks





def take_action(playlist,a,s,t,auto):
    if auto:
        if playlist[a].startswith("t"+"_"+str(playlist[s].split("_")[1])):

            return 1,0
        else:
            return 0,0
    print("[*] playing "+str(playlist[a]))
    skip = int(input("skip or no?: "))
    return (not skip),0

    
    
    
    
    

def list_prive(playlist,list1,list2,mode="e"):
    
    r = []

    for j,i in enumerate(list1):
        if mode =="e":
            if  i not in list2:
                r.append(i)
        else:
            if j not in [playlist.index(i) for i in list2]:
                r.append(list1[j])
        
    return r
def get_max(playlist,q,tr):
    a=0
    t=-1000000
    for p,j in enumerate(q):
        if p not in [playlist.index(k) for k in tr]:
            if j >= t:
                a=p
                t = j
    return a


def Train(final,start,t,eps,alpha,gamma,auto=True,mode="q",c=1,mode2="none",playlist=None):
    
    if auto :
        playlist = ["t"+"_"+str(i)+"_"+str(j) for i in range(15) for j in range(4)]
        pref = {}
        random.shuffle(playlist)
        start  = playlist.index("t_1_1")
    
    
    n = len(playlist)
    Q = np.zeros([n,n])
    rs= []
    for episode in range(t):
        done = False
        r = 0
        j=1
        N = {}
        for i in range(n):
            N[str(i)] = 0
        
        state = start
        trace2= [playlist[start]]
        while   j <= final and (list_prive(playlist,playlist,trace2) != []):
        
            #print(N)
            
            
            if random.uniform(0,1) < eps:

                
                a = playlist.index(random.choice(list_prive(playlist,playlist,trace2)))
            else:

                a=get_max(playlist,Q[state],trace2)

            if mode2== "ucb":
                p=[]
                for k in Q[state]:
                    if N[str(a)] != 0:
                        p.append(t+c*math.sqrt(math.log(j)/N[str(a)]))
                    else:
                        p.append(10000000)
                #print(p)
                a = np.argmax(p)
                N[str(a)] +=1
            else:
                if auto:
                    reward ,done=take_action(playlist,a,state,trace2,True)
                else:
                    reward ,done=take_action(playlist,a,trace2,auto=False)    
                
            trace2.append(playlist[a])
            next_max = Q[a][get_max(playlist,Q[a],trace2)]
            old_value = Q[state,a]
            if mode =="q":
                new_value = old_value + alpha*(reward+next_max-old_value)
                
            if mode =="epsgreedy":

                new_value = old_value + alpha*(reward-old_value)

            Q[state,a] = new_value

            r += reward 
            state = a
            j+=1
        rs.append(r)
            
        
        #print(trace2)
        print(f"epsiode {episode+1} done .. ")
        




    th = [playlist[start]]
    j = 0
    while j<= final:
        a = get_max(playlist,Q[state],th)
        reward, done = take_action(playlist,a,state,th,auto)
        th.append(playlist[a])
        state = a 
        j+=1

    return rs,th













def main(mode):
    if mode=="-tr":
    #Testing
        # test params : 
        num_songs_to_train = 100
        song_to_start  =1
        num_episodes = 200
        epsilon= 0.2
        learning_rate = 0.1
        gamma = 0.5 # useless
        auto = True 
        learning_algorithm = "q"
        c= 1 # for ucb 
        mode2 = "none" # ucb for ucb

        rs_q,tr_q= Train(num_songs_to_train,song_to_start,num_episodes,epsilon,learning_rate,gamma,auto,learning_algorithm,c,mode2)
        rs_eps,tr_eps= Train(num_songs_to_train,song_to_start,num_episodes,epsilon,learning_rate,gamma,auto,'epsgreedy',c,mode2)
        print("===================================")
        print("[+] Final Results : ")
        print("[+] Q - learning  final trace :" ,tr_q)
        print("[+] Q - learning average reward : ",np.mean(rs_q)) 
        print("[+] eps greedy - learning  final trace :" ,tr_eps)
        print("[+] eps greedy  - learning average reward : ",np.mean(rs_eps)) 
        plt.plot(range(num_episodes),rs_q,label="q")
        plt.title("q  vs eps_greedy algo")
        plt.plot(range(num_episodes),rs_eps,label="epsgreedy")
        plt.legend()
        plt.show()

    if mode =="-f":
        num_songs_to_train = 100
        song_to_start  =1
        num_episodes = 200
        #epsilon= 0.2
        learning_rate = 0.1
        gamma = 0.5 # useless
        auto = True 
        learning_algorithm = "q"
        c= 1 # for ucb 
        mode2 = "none" # ucb for ucb
        var = input("this is the finetuning mode of alpha or epsilon, type e for epsilon or a for alpha: ")
        if var == "e":
            
            epsilon_ranges = np.linspace(0.01,0.7,20)
            r_epsilon = []
            for epsilon in epsilon_ranges:
                rs_eps,tr_eps= Train(num_songs_to_train,song_to_start,num_episodes,epsilon,learning_rate,gamma,auto,'epsgreedy',c,mode2)
                r_epsilon.append(np.mean(rs_eps))
            print("======================")
            print("[*] best epsilon : "+str(epsilon_ranges[np.argmax(r_epsilon)]))
            plt.plot(epsilon_ranges,r_epsilon)
            plt.title("epsilon vs mean reward")
            plt.show()


        elif var == "a":
            epsilon = 0.18 # best epsilon accordin to previous fine tuning
            alpha_ranges = np.linspace(0.001,0.2,20)
            r_alpha = []
            for alpha in alpha_ranges:
                rs_eps,tr_eps= Train(num_songs_to_train,song_to_start,num_episodes,epsilon,alpha ,gamma,auto,'epsgreedy',c,mode2)
                r_alpha.append(np.mean(rs_eps))
            print("======================")
            print("[*] best alpha : "+str(alpha_ranges[np.argmax(r_alpha)]))
            plt.plot(alpha_ranges,r_alpha)
            plt.title("alpha vs mean reward")
            plt.show()
            # best alpha is 0.2
        else:
            print("wrong input")

    if mode == "-test":
        url  = input("enter url : ")
        username = input("enter username: ")
        all_tracks = get_playlist_tracks(username, url)
        playlist = []
        for i in all_tracks:
            playlist.append(i['track']['name'])
        num_songs_to_train = 100
        song_to_start  =1
        num_episodes = 200
        epsilon= 0.2
        learning_rate = 0.1
        gamma = 0.5 # useless
        auto = False 
        learning_algorithm = "q"
        c= 1 # for ucb 
        mode2 = "none" # ucb for ucb
        rs_eps,tr_eps= Train(num_songs_to_train,song_to_start,num_episodes,epsilon,learning_rate,gamma,auto,'epsgreedy',c,mode2)
    else:
        print("wrong input..")


if __name__ == "__main__":
    
    print("""
    Made By zitoun\n
    -tr to see how the model trained more about that in readme.md\n
    -f to fine tune epsilon or alpha\n 
    -test to test the model\n
          
          
          
          
          """)
    mode = str(input("enter mode : "))
    main(mode)