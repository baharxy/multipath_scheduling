import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy
import pickle
import pdb
import csv
import pandas as pd
import timeit
import sys
from pmf import ProbabilisticMatrixFactorization




def update_delay_profiles(df, slot):
 
     delayDF = pd.DataFrame(numpy.nan, index= range(0,slot), columns=['wifi', 'lte'] )
     wifi_rcvd_delay_list = df[df['feedbackreceivedWiFi'] < df.iloc [slot]['WiFiSentTimes'] ].index.tolist()
     lte_rcvd_delay_list = df[df['feedbackreceivedLTE'] < df.iloc [slot]['LTESentTimes' ] ].index.tolist()
     delayDF.wifi [ wifi_rcvd_delay_list ] = df.feedbackreceivedWiFi [wifi_rcvd_delay_list]
     delayDF.lte [ lte_rcvd_delay_list  ] =  df.feedbackreceivedLTE [lte_rcvd_delay_list]  

     return delayDF

def update_list_ratings( dynamicdf ):

     u = []
     v = []
     ratings = []
     num_slots = dynamicdf.shape[0]
     num_paths = 2
     
     latent_dimension = 5
     dynamicdf_wifi_rated = dynamicdf.index[dynamicdf['wifi'].apply(numpy.isnan)]
     dynamicdf_lte_rated = dynamicdf.index[dynamicdf['lte'].apply(numpy.isnan)]
     num_ratings = len(dynamicdf_wifi_rated) + len(dynamicdf_lte_rated)
     # Generate the latent user and item vectors
     for i in range(num_slots):
        u.append(2 * numpy.random.randn(latent_dimension))
     for i in range(num_slots):
        v.append(2 * numpy.random.randn(latent_dimension))

     # Get num_ratings ratings per user.
     for i in range(num_slots):
       if not ( i in dynamicdf_wifi_rated) :
           ratings.append((i,0, dynamicdf.iloc[i]['wifi']))
       if not ( i in dynamicdf_lte_rated) :    
           ratings.append((i,1, dynamicdf.iloc [i]['lte']))
          
     return (ratings, u, v)
     
 
def plot_ratings(ratings):
   xs = []
   ys = []
   for i in range(len(ratings)):
    xs.append(ratings[i][1])
    ys.append(ratings[i][2])
   pylab.plot(xs, ys, 'bx')
   pylab.show()
 

def plot_latent_vectors(U, V):
   fig = plt.figure()
   ax = fig.add_subplot(121)
   cmap = cm.jet
   ax.imshow(U, cmap=cmap, interpolation='nearest')
   plt.title("Users")
   plt.axis("off")
   ax = fig.add_subplot(122)
   ax.imshow(V, cmap=cmap, interpolation='nearest')
   plt.title("Items")
   plt.axis("off")

 

def predicted_ratings(U, V):

     r_hats = -5 * numpy.ones((U.shape[0] + U.shape[1] + 1, V.shape[0] + V.shape[1] + 1))
     for i in range(U.shape[0]):
      for j in range(U.shape[1]):
        r_hats[i + V.shape[1] + 1, j] = U[i, j]
  
     for i in range(V.shape[0]):
      for j in range(V.shape[1]):
        r_hats[j, i + U.shape[1] + 1] = V[i, j]
 
     for i in range(U.shape[0]):
      for j in range(V.shape[0]):
       r_hats[i + U.shape[1] + 1, j + V.shape[1] + 1] = numpy.dot(U[i], V[j]) / 10
 
     return r_hats 
     #fig = plt.figure()
     #ax = fig.add_subplot(111)
     #ax.imshow(r_hats, cmap=cm.gray, interpolation='nearest')
     #plt.title("Predicted Ratings")
     #plt.axis("off")



def round_robin_scheduler(df):
     wifi_scheduled_buffer_rr= numpy.array (range (0,9999,2) )
     wifi_block_rr = numpy.c_[wifi_scheduled_buffer_rr, df['WiFiSentTimes'], df['WiFiArrivalTimes']]
     lte_scheduled_buffer_rr= numpy.array (range (1,10000,2) )
     lte_block_rr= numpy.c_[lte_scheduled_buffer_rr, df['LTESentTimes'], df['LTEArrivalTimes'] ]
     snd_rcvd_block_rr = numpy.r_ [wifi_block_rr, lte_block_rr ]
     return  snd_rcvd_block_rr
    
def edpf_scheduler(df, pkt_length):
     avg_delay=numpy.asarray([numpy.mean ( df.WiFiDelayPackets) , numpy.mean (df.LTEDelayPackets )])
     avg_bw = numpy.asarray( [ numpy.mean(pkt_length * 8 /  df.WiFiDelayPackets),  numpy.mean(pkt_length * 8 /  df.LTEDelayPackets) ] ) # avergae over bw of each packet
     estimated_wifi_dlv_edpf = numpy.asarray ( df['WiFiSentTimes']  + pkt_length / avg_bw[0] )
     estimated_lte_dlv_edpf = numpy.asarray ( df['LTESentTimes']  + pkt_length / avg_bw[1] )
     estimated_dlv_edpf =  numpy.append (estimated_wifi_dlv_edpf, estimated_lte_dlv_edpf)
     sorted_slots= numpy.argsort(estimated_dlv_edpf, axis=-1, kind='mergesort')
     path_sorted = numpy.empty( [10000,1])
     path_sorted [ sorted_slots < 5000] = 0
     path_sorted [sorted_slots >= 5000] = 1
     wifi_scheduled_buffer_edpf = numpy.where(path_sorted == 0 )[0]
     wifi_block_edpf = numpy.c_[wifi_scheduled_buffer_edpf, estimated_wifi_dlv_edpf, df['WiFiSentTimes'], df['WiFiArrivalTimes'] ]
     lte_scheduled_buffer_edpf = numpy.where (path_sorted ==1 ) [0]
     lte_block_edpf= numpy.c_[lte_scheduled_buffer_edpf, estimated_lte_dlv_edpf, df['LTESentTimes'], df['LTEArrivalTimes'] ]
     snd_rcvd_block_edpf = numpy.r_ [wifi_block_edpf, lte_block_edpf ]
     return  snd_rcvd_block_edpf



def compute_expected_reordering (snd_rcvd_block):
  rcvd_block_sorted_by_seq = snd_rcvd_block [numpy.argsort(snd_rcvd_block[:, 0])]
  rcvd_block_sorted_by_seq = numpy.c_ [ rcvd_block_sorted_by_seq, numpy.asarray (numpy.zeros((rcvd_block_sorted_by_seq.shape[0],1)))]
  for k in range (0, rcvd_block_sorted_by_seq.shape[0]-1):
       rcvd_block_sorted_by_seq[k,-1]= max(rcvd_block_sorted_by_seq[:k+1,-2])     
  expected_reordering_delay =numpy.mean ( rcvd_block_sorted_by_seq[:,-1] - rcvd_block_sorted_by_seq[:,-3] )
  return expected_reordering_delay 
    
if __name__ == "__main__":

    
     # read data
     with open(sys.argv[1], 'r') as trace_file:
      df=pd.read_csv(trace_file)
     trace_file.close()
     n_slots=df.shape[0]
     pkt_length=1440 # packet length in bytes
     nof_pmf_calls = 0 
     # initialise
     nof_probes_last = 0 # to make sure that we go through pmf predictions for the first round
     nof_predictions_after_current_slot=20    # number of predicted delays that we draw from   from a normal distribution with the previous mean and sd values                
     nof_probed_slots = 0 # start with no feedback
     
     #scheduling for EDPF
     print "EDPF scheduler running.."
     snd_rcvd_edpf=edpf_scheduler(df,pkt_length)
     expected_reordering_delay_edpf=compute_expected_reordering(snd_rcvd_edpf)
     print "EDPF scheduler fimished. Expected reordering delay is:  %f" %expected_reordering_delay_edpf

     #scheduling for RR
     print "Round Robin scheduler running.. "
     snd_rcvd_rr=round_robin_scheduler(df)
     expected_reordering_delay_rr = compute_expected_reordering(snd_rcvd_rr)
     print "Round Robin scheduler finished.Expected reordering delay is:  %f" %expected_reordering_delay_rr

     ############## pmf ############################
     #start pmf from slot 5, no pint to start earlier 'casue there is no feedback
     for i  in range (5, n_slots):

            # construct matrix of received delay values assuming each packet sends rtts back
            dynamicdf = update_delay_profiles (df, i)
            (ratings, true_o, true_d) = update_list_ratings(dynamicdf)
            if len(ratings) > 0 :
                 nof_probed_slots =  int( numpy.max((numpy.array(ratings))[:,0])+1 )
            if nof_probed_slots  >  50:
                 dynamicdf = dynamicdf.drop(range(0,nof_probed_slots-50+1))
            (ratings, true_o, true_d) = update_list_ratings(dynamicdf)      
            if len(ratings) - nof_probes_last  >  5: # if had 5 more probes compared to the predictions call pmf/ otherwise just copy packets across both paths
                if nof_pmf_calls==0 :
                     # first  packets down both paths to build up some model
                     nof_duplicate_pkts=i
                     first_arrivals_best_path= numpy.argmin ( numpy.c_ [ df.iloc[0: nof_duplicate_pkts]['WiFiArrivalTimes'], df.iloc[0: nof_duplicate_pkts]['LTEArrivalTimes'] ],  axis=1 )
                     first_sentTimes=numpy.zeros(first_arrivals_best_path.shape[0])
                     for k  in range (nof_duplicate_pkts): 
                       if first_arrivals_best_path[k]==0 :
                           first_sentTimes[k]= df.iloc [k]['WiFiSentTimes']
                       else:
                           first_sentTimes[k]= df.iloc [k]['LTESentTimes']


                     first_arrivals= numpy.minimum( df.iloc[0:nof_duplicate_pkts]['WiFiArrivalTimes'], df.iloc[0:nof_duplicate_pkts]['LTEArrivalTimes'] )
                     snd_rcvd_block_last= numpy.c_[numpy.array(range(0,nof_duplicate_pkts)), numpy.zeros(nof_duplicate_pkts) , first_sentTimes, first_arrivals ]

                nof_pmf_calls= nof_pmf_calls+1
                print "calling pmf at slot  %d for %dth time"   %(i ,nof_pmf_calls)
                start_time=timeit.default_timer() # time the execution of the pmf and scheduler's sort
                pmf_instance = ProbabilisticMatrixFactorization(ratings, latent_d=3)
                #check to see if the map updates are done correctely
                liks = []
                while (pmf_instance.update()):
                   lik = pmf_instance.likelihood()
                   liks.append(lik)
                   #print "L=", lik
                   pass
                predicted_ratings= numpy.dot(pmf_instance.users, pmf_instance.items.transpose())
                mu_predicted=numpy.mean (predicted_ratings, axis=0)
                sigma_predicted= numpy.std (predicted_ratings,axis=0)
                
                if nof_probed_slots > 50:
                    predicted_delays_last= numpy.r_ [predicted_delays_last[:nof_probed_slots-50+1],predicted_ratings ]

                else:    
                    predicted_delays_last=predicted_ratings
                   
                    
                nof_predicted_ratings=predicted_delays_last.shape[0]
                nof_draws=i-nof_predicted_ratings+nof_predictions_after_current_slot
                forward_predicted_delays=numpy.c_[numpy.random.normal(mu_predicted[0], sigma_predicted[0], nof_draws ),  numpy.random.normal(mu_predicted[1], sigma_predicted[1], nof_draws )]
            
                #assume i.i.d distribution of delays: copy the same delay profile for the next 20 slots
                predicted_delays = numpy.r_ [ predicted_delays_last,forward_predicted_delays]
                last_predicted_slot = predicted_delays.shape[0]-1

                estimated_wifi_dlv_pmf= predicted_delays[i,0] + df.iloc[i:last_predicted_slot]['WiFiSentTimes']
                estimated_lte_dlv_pmf = predicted_delays[i,1] + df.iloc[i:last_predicted_slot]['LTESentTimes']
                estimated_dlv_pmf=  numpy.append (estimated_wifi_dlv_pmf, estimated_lte_dlv_pmf) 
                sorted_slots_pmf= numpy.argsort(estimated_dlv_pmf, axis=-1, kind='mergesort')
                path_sorted_pmf = numpy.empty( [sorted_slots_pmf.shape[0],1])
                path_sorted_pmf [ sorted_slots_pmf < estimated_wifi_dlv_pmf.shape[0]  ] = 0
                path_sorted_pmf [sorted_slots_pmf >= estimated_wifi_dlv_pmf.shape[0] ] = 1
                wifi_scheduled_buffer_pmf = numpy.where(path_sorted_pmf == 0 )[0] + i
                wifi_block_pmf = numpy.c_[wifi_scheduled_buffer_pmf, estimated_wifi_dlv_pmf, df.iloc[i:last_predicted_slot]['WiFiSentTimes'], df.iloc[i:last_predicted_slot]['WiFiArrivalTimes'] ]
                lte_scheduled_buffer_pmf = numpy.where (path_sorted_pmf ==1 ) [0] + i
                lte_block_pmf= numpy.c_[lte_scheduled_buffer_pmf, estimated_lte_dlv_pmf, df.iloc[i:last_predicted_slot]['LTESentTimes'], df.iloc[i:last_predicted_slot]['LTEArrivalTimes'] ]
                snd_rcvd_block_pmf = numpy.r_ [wifi_block_pmf, lte_block_pmf]
                snd_rcvd_block_last=numpy.r_ [ snd_rcvd_block_last [ snd_rcvd_block_last[:,0] < i , ]  , snd_rcvd_block_pmf ]
                nof_probs_last=  len(ratings)
                elapsed=timeit.default_timer() - start_time
                print "pmf for slot no. %d is finished in %f sec" %(i,elapsed)

     finished=1
     pdb.set_trace()
     expected_reordering_delay_pmf = compute_expected_reordering(snd_rcvd_block_last)

