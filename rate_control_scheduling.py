import pylab
import  scipy.stats as stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from operator import add
import numpy
import math
import pickle
import pdb
import csv
import pandas as pd
import timeit
import sys
from pmf import ProbabilisticMatrixFactorization
from mg import MaxofGaussians


########### get the feedback based on previous transmissions#######################
def update_delay_profiles(current_sent_times ,array_wifi, array_lte,slot):
   
 
     sent_wifi_slots= array_wifi.shape[0]
     sent_lte_slots=   array_lte.shape[0]
     feedback_times_wifi =  array_wifi  [: , 3]+ 2* ( array_wifi [: , 4]-  array_wifi [:, 3] )
     feedback_times_lte =  array_lte  [: , 3]+ 2* ( array_lte [: , 4]-  array_lte [:, 3] ) 
     rcvd_wifi_slots=numpy.array ( [ map(int, x) for x in numpy.where(feedback_times_wifi  <   current_sent_times[0]) ][0] ) 
     rcvd_lte_slots=numpy.array( [ map(int, x) for x in numpy.where(feedback_times_lte <   current_sent_times[1]) ][0] ) 
     
     delayDF = pd.DataFrame(numpy.nan, index= range(0,slot), columns=['wifi', 'lte'] )   
     delayDF.wifi [ rcvd_wifi_slots ] = df.WiFiDelayPackets [rcvd_wifi_slots]
     delayDF.lte [ rcvd_lte_slots  ] =  df.LTEDelayPackets [rcvd_lte_slots]
     return delayDF

#########################################################################
# sort packets t-ransmission times over paths so that they arrive in order
def sort_block_packets(predicted_delays,s_i, df, remaining_pkt_set):
   
   last_predicted_slot = predicted_delays.shape[0]-1
   estimated_wifi_dlv= predicted_delays[s_i:last_predicted_slot,0] + df.iloc[s_i:last_predicted_slot]['WiFiSentTimes']
   estimated_lte_dlv = predicted_delays[s_i:last_predicted_slot,1] + df.iloc[s_i:last_predicted_slot]['LTESentTimes']
   estimated_dlv=  numpy.append (estimated_wifi_dlv, estimated_lte_dlv) 
   sorted_slots= numpy.argsort(estimated_dlv, axis=-1, kind='mergesort')
   path_sorted = numpy.empty( [sorted_slots.shape[0],1])
   path_sorted [ sorted_slots < estimated_wifi_dlv.shape[0]  ] = 0
   path_sorted [sorted_slots >= estimated_wifi_dlv.shape[0] ] = 1
   wifi_scheduled_buffer = remaining_pkt_set [ numpy.where(path_sorted == 0)[0]  ] 
   wifi_block= numpy.c_[wifi_scheduled_buffer, numpy.zeros(wifi_scheduled_buffer.shape[0]),  estimated_wifi_dlv, df.iloc[s_i:last_predicted_slot]['WiFiSentTimes'], df.iloc[s_i:last_predicted_slot]['WiFiArrivalTimes'] ]
   lte_scheduled_buffer = remaining_pkt_set [ numpy.where(path_sorted == 1)[0]  ]
   lte_block= numpy.c_[lte_scheduled_buffer, numpy.ones(lte_scheduled_buffer.shape[0]), estimated_lte_dlv, df.iloc[s_i:last_predicted_slot]['LTESentTimes'], df.iloc[s_i:last_predicted_slot]['LTEArrivalTimes'] ]

   #snd_rcvd_block = numpy.r_ [wifi_block, lte_block]
   return wifi_block, lte_block
##########################################################################
#just computes the reordering delays for different schedules   
def reordering_stats (snd_rcvd_block):
  rcvd_block_sorted_by_seq = snd_rcvd_block [numpy.argsort(snd_rcvd_block[:, 0])]
  rcvd_block_sorted_by_seq = numpy.c_ [ rcvd_block_sorted_by_seq, numpy.asarray (numpy.zeros((rcvd_block_sorted_by_seq.shape[0],1)))]
  for k in range (0, rcvd_block_sorted_by_seq.shape[0]):
       rcvd_block_sorted_by_seq[k,-1]= numpy.max(rcvd_block_sorted_by_seq[:k+1,-2])     
  expected_reordering_delay =numpy.mean ( rcvd_block_sorted_by_seq[:,-1] - rcvd_block_sorted_by_seq[:,-3] )
  std_reordering_delay =numpy.std ( rcvd_block_sorted_by_seq[:,-1] - rcvd_block_sorted_by_seq[:,-3] )
  return expected_reordering_delay, std_reordering_delay, rcvd_block_sorted_by_seq
    
if __name__ == "__main__":

    
     # read data
     with open(sys.argv[1], 'r') as trace_file:
      df=pd.read_csv(trace_file)
     trace_file.close()
 
     #replace na values with previous (i.e. packet losses are taken care of with some coding process, valid?)
     wifi_delay_padding_list=df.loc[df[pd.isnull(df['WiFiDelayPackets'])].index-1, 'WiFiDelayPackets' ] .values.T.tolist() 
     second_nan_idx_wifi= [map(int, x) for x in numpy.where(numpy.isnan(wifi_delay_padding_list))][0]
     for iw in second_nan_idx_wifi:
         wifi_delay_padding_list[iw]=wifi_delay_padding_list[iw-1]
     lte_delay_padding_list= df.loc[df[pd.isnull(df['LTEDelayPackets'])].index-1, 'LTEDelayPackets' ] .values.T.tolist() 
     second_nan_idx_lte= [map(int, x) for x in numpy.where(numpy.isnan(lte_delay_padding_list))][0]
     for il in second_nan_idx_lte:
         lte_delay_padding_list[il]=lte_delay_padding_list[il-1]
     wifi_arrival_padding_list=map(add, df.loc[df[pd.isnull(df['WiFiArrivalTimes'])].index, 'WiFiSentTimes' ].values.T.tolist() , df.loc[df[pd.isnull(df['WiFiArrivalTimes'])].index-1, 'WiFiDelayPackets' ]  . values.T.tolist() )
     for iwa in second_nan_idx_wifi:
         wifi_arrival_padding_list[iwa]=wifi_arrival_padding_list[iwa-1]
     lte_arrival_padding_list = map(add,  df.loc[df[pd.isnull(df['LTEArrivalTimes'])].index, 'LTESentTimes' ]. values.T.tolist() , df.loc[df[pd.isnull(df['LTEArrivalTimes'])].index-1, 'LTEDelayPackets' ] . values.T.tolist() )
     for iwl in second_nan_idx_lte:
         lte_arrival_padding_list[iwl]=lte_arrival_padding_list[iwl-1]

     df.loc[df[pd.isnull(df['WiFiDelayPackets'])].index,'WiFiDelayPackets']= wifi_delay_padding_list
     df.loc[df[pd.isnull(df['LTEDelayPackets'])].index,'LTEDelayPackets']= lte_delay_padding_list
     df.loc[df[pd.isnull(df['WiFiArrivalTimes'])].index,'WiFiArrivalTimes']= wifi_arrival_padding_list
     df.loc[df[pd.isnull(df['LTEArrivalTimes'])].index,'LTEArrivalTimes']= lte_arrival_padding_list
     
     #dummy delay***for test************************
     #df['LTEDelayPackets']=df['LTEDelayPackets']+0.1
     #df['LTEArrivalTimes']=df['LTEArrivalTimes']+0.1
     #df['feedbackreceivedLTE']=df['LTESentTimes']+ 2 * df['LTEDelayPackets']
     #**********************************************
     
     # initialise 
     n_slots=df.shape[0] # buffer size per path(link)
     pkt_seq=0 # sum of  the n_slots on each path
     pkt_length=1440 # packet length in bytes
     nof_calls = 0 
     n_runs=1  # number of simulations
     n_schedulers=1
     expected_reordering_delay = numpy.zeros((n_runs, n_schedulers ))
     expected_time_towait= numpy.zeros((n_runs, n_schedulers ))
     std_reordering_delay = numpy.zeros((n_runs, n_schedulers ))
     RMSE = numpy.zeros((n_runs, n_schedulers-1 ))
     best=[] # record of best paths- sedpf
     wifi_buffer= numpy.zeros(n_slots)
     lte_buffer= numpy.zeros(n_slots)
     


    
     for s_i  in range (0, n_slots):

                ################# calling LRTT##########################
                start_time=timeit.default_timer() 
                nof_calls= nof_calls+1
                current_sent_times=[df.iloc[s_i]['WiFiSentTimes'], df.iloc[s_i]['LTESentTimes'] ]
                print "calling LRTT at slot  %d for %dth time"   %(s_i ,nof_calls)
		allRcvd= update_delay_profiles(current_sent_times , wifi_buffer, lte_buffer ,s_i)

                index_ack_rcvd_wifi=  numpy.isfinite(numpy.asarray(allRcvd.wifi))
                index_ack_rcvd_lte=  numpy.isfinite(numpy.asarray(allRcvd.lte))

                

                if len(index_ack_rcvd_wifi)==len(index_ack_rcvd_lte):
                  wifi_buffer[s_i]= pkt_seq
                  lte_buffer[s_i]= pkt_seq + 1
                  pkt_seq= pkt_seq + 2
                elif len(index_ack_rcvd_wifi) < len(index_ack_rcvd_lte):
                  wifi_buffer[s_i]= pkt_seq 
                elif len(index_ack_rcvd_lte) < len(index_ack_rcvd_wifi):
                  lte_buffer[s_i]= pkt_seq 


                elapsed=timeit.default_timer() - start_time
                print "LRTT for slot no. %d is finished in %f sec" %(s_i,elapsed)


                        
                
                
     finished=1
     

    
