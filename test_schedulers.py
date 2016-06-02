import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy
import math
import pickle
import pdb
import csv
import pandas as pd
import timeit
import sys
from pmf import ProbabilisticMatrixFactorization




def delay_profiles(df, slot):

     delayDF = pd.DataFrame(numpy.nan, index= range(0,slot), columns=['wifi', 'lte'] )
     wifi_rcvd_delay_list = df[df['feedbackreceivedWiFi'] < df.iloc [slot]['WiFiSentTimes'] ].index.tolist()
     lte_rcvd_delay_list = df[df['feedbackreceivedLTE'] < df.iloc [slot]['LTESentTimes' ] ].index.tolist()
     delayDF.wifi [ wifi_rcvd_delay_list ] = df.WiFiDelayPackets [wifi_rcvd_delay_list]
     delayDF.lte [ lte_rcvd_delay_list  ] =  df.LTEDelayPackets [lte_rcvd_delay_list]      
     return delayDF

def update_delay_profiles(df_duplicates, current_sent_times ,feedbackarray,slot, nof_duplicates):
     
     delayDF = pd.DataFrame(numpy.nan, index= range(0,slot), columns=['wifi', 'lte'] )
     sent_wifi_slots= numpy.where (feedbackarray [:, 1]== 0) 
     sent_lte_slots=  numpy.where (feedbackarray [:, 1]== 1) 
     feedback_times =  feedbackarray  [: , 3]+ 2* ( feedbackarray [: , 4]-  feedbackarray [:, 3] ) 
     rcvd_wifi_slots= feedbackarray[  (feedback_times  <   current_sent_times[0] ) & (feedbackarray[:,1]==0)   , 0]
     rcvd_lte_slots= feedbackarray[ ( feedback_times  <   current_sent_times[0])   & (feedbackarray[:,1]==1)   , 0]
     if len(sent_wifi_slots)==len(sent_lte_slots) :
            wifi_rcvd_delay_list = numpy.append(  rcvd_wifi_slots  , numpy.asarray( df_duplicates[df_duplicates['feedbackreceivedWiFi'] <  current_sent_times[0]  ].index.tolist()) )
            lte_rcvd_delay_list= numpy.append(  rcvd_lte_slots ,  numpy.asarray (df_duplicates[df_duplicates ['feedbackreceivedLTE'] <  current_sent_times[1]    ].index.tolist()) )
     else:
          #throw an expetion
          raise ValueError, 'Something us wong'
     delayDF.wifi [ wifi_rcvd_delay_list ] = df.WiFiDelayPackets [wifi_rcvd_delay_list]
     delayDF.lte [ lte_rcvd_delay_list  ] =  df.LTEDelayPackets [lte_rcvd_delay_list]

     
     return delayDF



def update_list_ratings( dynamicdf ):

     u = []
     v = []
     ratings = [] #delays
     num_slots = dynamicdf.shape[0] #items
     num_paths = 2 #users
     latent_dimension = 4
     dynamicdf_wifi_rated = dynamicdf.index[dynamicdf['wifi'].apply(numpy.isnan)]
     dynamicdf_lte_rated = dynamicdf.index[dynamicdf['lte'].apply(numpy.isnan)]
     num_ratings = len(dynamicdf_wifi_rated) + len(dynamicdf_lte_rated)
     # Generate the latent user and item vectors
     for i in range(num_paths):
        u.append(2 * numpy.random.randn(latent_dimension))
     for i in range(num_slots):
        v.append(2 * numpy.random.randn(latent_dimension))
     
     # Get num_ratings ratings per user.
     ind=0
     for i in dynamicdf.index:
       if not ( i in dynamicdf_wifi_rated) :
           ratings.append((0, ind , float(dynamicdf.wifi[dynamicdf.index==i]) ))
       if not ( i in dynamicdf_lte_rated) :    
           ratings.append((1,ind , float(dynamicdf.lte[dynamicdf.index==i]) ) )
       ind=ind+1   
     return (ratings, u, v)
     


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


#assigns pkts to slots of each path intermittently 
def round_robin_scheduler(df):
     wifi_scheduled_buffer_rr= numpy.array (range (0,9999,2) )
     wifi_block_rr = numpy.c_[wifi_scheduled_buffer_rr, df['WiFiSentTimes'], df['WiFiArrivalTimes']]
     lte_scheduled_buffer_rr= numpy.array (range (1,10000,2) )
     lte_block_rr= numpy.c_[lte_scheduled_buffer_rr, df['LTESentTimes'], df['LTEArrivalTimes'] ]
     snd_rcvd_block_rr = numpy.r_ [wifi_block_rr, lte_block_rr ]
     return  snd_rcvd_block_rr

#edpf estimates packet delivery times based on average transmision delays, here assumed propagation delay is zero    
def edpf_scheduler(df, pkt_length):
     A=[0 , 0] #estimated time that wireless channel becomes available for tranmission
     estimated_wifi_dlv_edpf=[]
     estimated_lte_dlv_edpf=[]
     avg_delay=numpy.asarray([numpy.mean ( df.WiFiDelayPackets) , numpy.mean (df.LTEDelayPackets )])
     avg_bw = numpy.asarray( [ numpy.mean(pkt_length * 8 /  df.WiFiDelayPackets),  numpy.mean(pkt_length * 8 /  df.LTEDelayPackets) ] ) # avergae over bw of each packet
     for slot in range(df.shape[0]):
            estimated_wifi_dlv_edpf.append ( max( df.iloc[slot]['WiFiSentTimes'], A[0] )  + pkt_length * 8 / avg_bw[0] )
            estimated_lte_dlv_edpf.append  (  max (df.iloc[slot]['LTESentTimes'], A[1] )  + pkt_length * 8  / avg_bw[1] )
            A=[ estimated_wifi_dlv_edpf[slot],  estimated_lte_dlv_edpf[slot]]     
     estimated_dlv_edpf =  numpy.append (numpy.asarray(estimated_wifi_dlv_edpf), numpy.asarray(estimated_lte_dlv_edpf) )
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

#sedpf assumes invariant gaussian arrivals : it assume that sent times and delays are both gaussian variables (only makes sense to apply to my traces
#with fixed transmit rate when transmission rate is higher than the channel bw, then randomness in actual sent time could be a valid assumption
#def sedpf_scheduler(a_first, transmision_times_mu, transmision_times_sigma, current_slot):
   ##what is the estimated window length
   #length_Delta_wifi= math.ceil(3*(transmision_times_mu[0] -transmision_times_sigma[0] )  / transmision_times__mu[0] )
   #length_Delta_lte= math.ceil (3*(transmision_times_mu[0] -transmision_times_sigma[0] )  / transmision_times__mu[0] )
   ## estimates the mean arrival(delivery to the physical layer) for slots within a given window
   #Delta_mean_wifi = range (transmision_times_mu[0],   transmission_times_mu[0]*(length_Delta_wifi )     , transmission_times_mu[0])
   #Delta_mean_lte = range (transmision_times_mu[1],   transmission_times_mu[1]*(length_Delta_lte  )     , transmission_times_mu[1])
   #Y_wifi = gassian_appx( Delta_mean_wifi, [transmision_times_sigma[0]]*  length_Delta_wifi   ) + a_first[0] # this is actually random variable Z+a_first
   #Y_lte = gaussian_appx(  Delta_mean_lte, [transmision_times_sigma[1]]*  length_Delta_lte   ) + a_first[1]
   #a_slot_wifi= [slot_time_wifi + transmision_times_mu [0], transmission_times_sigma[0] ]
   #a_slot_lte= [slot_time_lte + transmision_times_mu [1], transmission_times_sigma[1] ]
   #estimate_reoedering_wifi=guassian_appx ([Y_wifi[0], Y_LTE[0], a_slot_wifi[0] ],[Y_wifi[1], Y_lte[1], a_slot_wifi[1]])
   #estimate_reoedering_lte=guassian_appx ([Y_wifi[0], Y_LTE[0], a_slot_lte[0] ],[Y_wifi[1], Y_lte[1], a_slot_lte[1]])
   #numpy.minimum ( estimate_reoedering_wifi

   
#just computes the reordering delays based on different schedules   
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
     #replace na values with previous (i.e. packet losses are taken care of with some coding process, valid?)
     df.loc[df[pd.isnull(df['WiFiDelayPackets'])].index,'WiFiDelayPackets']=df.loc[df[pd.isnull(df['WiFiDelayPackets'])].index-1, 'WiFiDelayPackets' ] 
     df.loc[df[pd.isnull(df['LTEDelayPackets'])].index,'LTEDelayPackets']=df.loc[df[pd.isnull(df['LTEDelayPackets'])].index-1, 'LTEDelayPackets' ] 
     df.loc[df[pd.isnull(df['WiFiArrivalTimes'])].index,'WiFiArrivalTimes']=df.loc[df[pd.isnull(df['WiFiArrivalTimes'])].index, 'WiFiSentTimes' ] + df.loc[df[pd.isnull(df['WiFiArrivalTimes'])].index-1, 'WiFiDelayPackets' ] 
     df.loc[df[pd.isnull(df['LTEArrivalTimes'])].index,'LTEArrivalTimes']=df.loc[df[pd.isnull(df['LTEArrivalTimes'])].index, 'LTESentTimes' ] + df.loc[df[pd.isnull(df['LTEArrivalTimes'])].index-1, 'LTEDelayPackets' ] 
     n_slots=df.shape[0]
     pkt_length=1440 # packet length in bytes
     nof_pmf_calls = 0 
     # initialise pmf
     nof_probes_last = 0 # to make sure that we go through pmf predictions for the first round
     nof_predictions_after_current_slot=20    # number of predicted delays that we draw from   from a normal distribution with the previous mean and sd values                
     nof_probed_slots = 0 # start with no feedback
     
     #scheduling for EDPF
     print "EDPF scheduler running.."
     snd_rcvd_edpf=edpf_scheduler(df,pkt_length)
     expected_reordering_delay_edpf=compute_expected_reordering(snd_rcvd_edpf)
     print "EDPF scheduler finished. Expected reordering delay is:  %f" %expected_reordering_delay_edpf

     #scheduling for RR
     print "Round Robin scheduler running.. "
     snd_rcvd_rr=round_robin_scheduler(df)
     expected_reordering_delay_rr = compute_expected_reordering(snd_rcvd_rr)
     print "Round Robin scheduler finished.Expected reordering delay is:  %f" %expected_reordering_delay_rr

    
     for s_i  in range (5, n_slots):

            # construct matrix of received delay values assuming each packet sends rtts back
            if nof_pmf_calls==0:                                  
                 dynamicdf = delay_profiles (  df, s_i)
            else:
                 duplicates_df= df[:nof_duplicate_pkts]
                 send_times= [ df.iloc [s_i]['WiFiSentTimes'], df.iloc [s_i]['LTESentTimes'] ]
                 dynamicdf = update_delay_profiles (  duplicates_df, send_times, snd_rcvd_block_last[nof_duplicate_pkts:s_i,:], s_i,  nof_duplicate_pkts)
            (ratings, true_o, true_d) = update_list_ratings(dynamicdf)
            if len(ratings) > 0 :
                 nof_probed_slots =  int( numpy.max((numpy.array(ratings))[:,1])+1 )
            if nof_probed_slots  >  50:
                 dynamicdf = dynamicdf.drop(range(0,nof_probed_slots-50+1))
            (ratings, true_o, true_d) = update_list_ratings(dynamicdf)      
            if len(ratings) - nof_probes_last  >  5: # if had 5 more probes compared to the predictions call pmf/ otherwise just copy packets across both paths
                if nof_pmf_calls==0 :
                     # first  packets down both paths to build up some model
                     nof_duplicate_pkts=s_i
                     first_arrivals_best_path= numpy.argmin ( numpy.c_ [ df.iloc[0: nof_duplicate_pkts]['WiFiArrivalTimes'], df.iloc[0: nof_duplicate_pkts]['LTEArrivalTimes'] ],  axis=1 )
                     first_sentTimes=numpy.zeros(first_arrivals_best_path.shape[0])
                     for k  in range (nof_duplicate_pkts): 
                       if first_arrivals_best_path[k]==0 :
                           first_sentTimes[k]= df.iloc [k]['WiFiSentTimes']
                       else:
                           first_sentTimes[k]= df.iloc [k]['LTESentTimes']


                     first_arrivals= numpy.minimum( df.iloc[0:nof_duplicate_pkts]['WiFiArrivalTimes'], df.iloc[0:nof_duplicate_pkts]['LTEArrivalTimes'] )
                     # a block  to be populated by #1: pkt id, #2: scheduled path #3: estimate arrival #4: sent time #5: arrival  time
                     snd_rcvd_block_last= numpy.c_[numpy.array(range(0,nof_duplicate_pkts)), 3*numpy.ones(nof_duplicate_pkts), numpy.zeros(nof_duplicate_pkts) , first_sentTimes, first_arrivals ]
                     ## initialise sedpf from previous duplicate packets
                     transmision_times_mu=[]
                     transmision_times_sigma=[]
                     transmision_times_mu.append ( numpy.mean(  numpy.asarray(df.iloc[:nof_duplicate_pkts]['WiFiDelayPackets']) ) )
                     transmision_times_mu.append ( numpy.mean(  numpy.asarray(df.iloc[:nof_duplicate_pkts]['LTEDelayPackets']) ) )
                     transmision_times_sigma.append (  numpy.std (  numpy.asarray(df.iloc[:nof_duplicate_pkts]['WiFiDelayPackets']) ) )
                     transmision_times_sigma.append (  numpy.std (  numpy.asarray(df.iloc[:nof_duplicate_pkts]['LTEDelayPackets']) ) )
                     length_Delta_wifi= math.ceil(3*(transmision_times_mu[0] -transmision_times_sigma[0] )  ) #anything within 3sigma of the mean of the distribution
                     length_Delta_lte= math.ceil (3*(transmision_times_mu[0] -transmision_times_sigma[0] )   )
                
                # calling sedpf/not yet sedpf is messy/ can I apply it to my data?      
                nof_pmf_calls= nof_pmf_calls+1
                print "calling pmf at slot  %d for %dth time"   %(s_i ,nof_pmf_calls)
                start_time=timeit.default_timer() # time the execution of the pmf and scheduler's sort
                pmf_instance = ProbabilisticMatrixFactorization( ratings, latent_d=4)
                #check to see if the map updates are done correctely
                liks = []
                while (pmf_instance.update()):
                   lik = pmf_instance.likelihood()
                   liks.append(lik)
                   #print "L=", lik
                   pass
                predicted_ratings= numpy.transpose (numpy.dot(pmf_instance.users, pmf_instance.items.transpose()))
                mu_predicted=numpy.mean (predicted_ratings, axis=0)
                sigma_predicted= numpy.std (predicted_ratings,axis=0)
                
                if nof_probed_slots > 50:
                    predicted_delays_last= numpy.r_ [predicted_delays_last[:nof_probed_slots-50+1],predicted_ratings ]

                else:    
                    predicted_delays_last=predicted_ratings
                   
                    
                nof_predicted_ratings=predicted_delays_last.shape[0]
                nof_draws=s_i-nof_predicted_ratings+nof_predictions_after_current_slot
                forward_predicted_delays=numpy.c_[numpy.random.normal(mu_predicted[0], sigma_predicted[0], nof_draws ),  numpy.random.normal(mu_predicted[1], sigma_predicted[1], nof_draws )]
            
                #assume i.i.d distribution of delays: copy the same delay profile for the next 20 slots
                predicted_delays = numpy.r_ [ predicted_delays_last,forward_predicted_delays]
                last_predicted_slot = predicted_delays.shape[0]-1

                estimated_wifi_dlv_pmf= predicted_delays[s_i,0] + df.iloc[s_i:last_predicted_slot]['WiFiSentTimes']
                estimated_lte_dlv_pmf = predicted_delays[s_i,1] + df.iloc[s_i:last_predicted_slot]['LTESentTimes']
                estimated_dlv_pmf=  numpy.append (estimated_wifi_dlv_pmf, estimated_lte_dlv_pmf) 
                sorted_slots_pmf= numpy.argsort(estimated_dlv_pmf, axis=-1, kind='mergesort')
                path_sorted_pmf = numpy.empty( [sorted_slots_pmf.shape[0],1])
                path_sorted_pmf [ sorted_slots_pmf < estimated_wifi_dlv_pmf.shape[0]  ] = 0
                path_sorted_pmf [sorted_slots_pmf >= estimated_wifi_dlv_pmf.shape[0] ] = 1
                wifi_scheduled_buffer_pmf = numpy.where(path_sorted_pmf == 0 )[0] + s_i
                wifi_block_pmf = numpy.c_[wifi_scheduled_buffer_pmf, numpy.zeros(wifi_scheduled_buffer_pmf.shape[0]),  estimated_wifi_dlv_pmf, df.iloc[s_i:last_predicted_slot]['WiFiSentTimes'], df.iloc[s_i:last_predicted_slot]['WiFiArrivalTimes'] ]
                lte_scheduled_buffer_pmf = numpy.where (path_sorted_pmf ==1 ) [0] + s_i
                lte_block_pmf= numpy.c_[lte_scheduled_buffer_pmf, numpy.ones(lte_scheduled_buffer_pmf.shape[0]), estimated_lte_dlv_pmf, df.iloc[s_i:last_predicted_slot]['LTESentTimes'], df.iloc[s_i:last_predicted_slot]['LTEArrivalTimes'] ]
                snd_rcvd_block_pmf = numpy.r_ [wifi_block_pmf, lte_block_pmf]
                snd_rcvd_block_last=numpy.r_ [ snd_rcvd_block_last [ snd_rcvd_block_last[:,0] < s_i , ]  , snd_rcvd_block_pmf ]
                nof_probs_last=  len(ratings)
                elapsed=timeit.default_timer() - start_time
                print "pmf for slot no. %d is finished in %f sec with likelihood %f" %(s_i,elapsed, lik)
                

     finished=1
     pdb.set_trace()
     expected_reordering_delay_pmf = compute_expected_reordering(snd_rcvd_block_last)

