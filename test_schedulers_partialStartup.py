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


##### get the feedback based on previous transmissions/ seperate out wifi and lte#######################     
def update_delay_profiles(*args):
     pdb.set_trace()
     if len(args)==5: 
     	current_sent_times=args[0]
     	feedbackarray_wifi=args[1]
     	feedbackarray_lte=args[2]
     	slot_wifi=args[3]
     	slot_lte=args[4]
     elif len(args)==4:
     	current_sent_times=args[0]
     	feedbackarray_wifi=[]
     	feedbackarray_lte=args[2]
     	slot_wifi=args[3]
     	slot_lte=args[4]
     	
     sent_wifi_slots= feedbackarray_wifi.shape[0]
     sent_lte_slots=   feedbackarray_lte.shape[0]
     feedback_times_wifi =  feedbackarray_wifi  [: , 3]+ 2* ( feedbackarray_wifi [: , 4]-  feedbackarray_wifi [:, 3] )
     feedback_times_lte =  feedbackarray_lte  [: , 3]+ 2* ( feedbackarray_lte [: , 4]-  feedbackarray_lte [:, 3] ) 
     rcvd_wifi_slots=numpy.array ( [ map(int, x) for x in numpy.where(feedback_times_wifi  <   current_sent_times[0]) ][0] ) + nof_duplicates
     rcvd_lte_slots=numpy.array( [ map(int, x) for x in numpy.where(feedback_times_lte <   current_sent_times[0]) ][0] ) +nof_duplicates
     wifi_rcvd_delay_list = numpy.append(  rcvd_wifi_slots  , numpy.asarray( df_duplicates[df_duplicates['feedbackreceivedWiFi'] <  current_sent_times[0]  ].index.tolist()) )
     lte_rcvd_delay_list= numpy.append(  rcvd_lte_slots ,  numpy.asarray (df_duplicates[df_duplicates ['feedbackreceivedLTE'] <  current_sent_times[1]    ].index.tolist()) )
     
     wifidelays = pd.DataFrame(numpy.nan, index= range(0,slot_wifi), columns=['wifi'] )
     ltedelays = pd.DataFrame(numpy.nan, index= range(0,slot_lte), columns=['lte'] )
     wifidelays.wifi[ wifi_rcvd_delay_list ] = df.WiFiDelayPackets [wifi_rcvd_delay_list]   
     ltedelays.lte[ lte_rcvd_delay_list  ] =  df.LTEDelayPackets [lte_rcvd_delay_list]
     return wifidelays,ltedelays

#######change the format of feedback to tuples######
def update_list_ratings( dynamicdf_wifi, dynamicdf_lte ):

     pdb.set_trace()
     ratings = [] #delays
     num_slots = dynamicdf.shape[0] #items
     num_paths = 2 #users
     dynamicdf_wifi_rated = dynamicdf.index[dynamicdf_wifi['wifi'].apply(numpy.isnan)]
     dynamicdf_lte_rated = dynamicdf.index[dynamicdf_lte['lte'].apply(numpy.isnan)]
     num_ratings = len(dynamicdf_wifi_rated) + len(dynamicdf_lte_rated)
     
     # Get num_ratings ratings per user.
     ind=0
     for i in dynamicdf.index:
       if not ( i in dynamicdf_wifi_rated) :
           ratings.append((0, ind , float(dynamicdf.wifi[dynamicdf.index==i]) ))
       if not ( i in dynamicdf_lte_rated) :    
           ratings.append((1,ind , float(dynamicdf.lte[dynamicdf.index==i]) ) )
       ind=ind+1   
     return (ratings, u, v)
     

########gets the pmf data
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

############################### moving average################ 
def moving_average (feedbacks_wifi, feedbacks_lte, slot_n, slot_n_wifi, prediction_length, mmc):
     
     past_delays_wifi_fitted=numpy.asarray(feedbacks_wifi)
     past_delays_lte_fitted=numpy.asarray(feedbacks_lte)
     wifi_mean = stats.nanmean(past_delays_fitted_wifi)
     wifi_std = stats.nanstd(past_delays_fitted_wifi)
     lte_mean = stats.nanmean(past_delays_fitted_lte)
     lte_std = stats.nanstd(past_delays_fitted_lte)
     col_mean=[wifi_mean, lte_mean]
     col_std=[wifi_std, lte_std]
     inds_wifi = numpy.where(numpy.isnan(past_delays_fitted_wifi))
     inds_lte = numpy.where(numpy.isnan(past_delays_fitted_lte))
     past_delays_fitted_wifi[inds_wifi]=numpy.take(wifi_mean,inds_wifi[1])
     past_delays_fitted_lte[inds_lte]=numpy.take(lte_mean,inds_lte[1])
     wifi_delays=past_delays_fitted_wifi
     lte_delays=past_delays_fitted_lte
     
     if mmc: 
            
	    forward_predicted_delays_mmc=numpy.c_[numpy.random.normal(col_mean[0], col_std[0], prediction_length),  numpy.random.normal(col_mean[1], col_std[1], prediction_length )]
            forward_predicted_delays_mmc[ forward_predicted_delays_mmc <0 ]=0 # truncate negative samples 
            predicted_wifi_ma=numpy.r_ [past_delays_wifi_fitted, forward_predicted_delays_mmc[:,0]]
            predicted_lte_ma=numpy.r_ [past_delays_lte_fitted, forward_predicted_delays_mmc[:,1]]
            return  col_mean, col_std, predicted_wifi_ma , predicted_lte_ma 
     else:
            predicted_wifi_ma=numpy.r_ [past_delays_wifi_fitted, numpy.zeros((prediction_length,1)) ]
            predicted_lte_ma=numpy.r_ [past_delays_lte_fitted, numpy.zeros((prediction_length,1)) ]
	    for pl in range (past_delays_wifi_fitted.shape[0] , past_delays_wifi_fitted.shape[0]+prediction_length):
		 predicted_wifi_ma[pl]= numpy.sum(numpy.divide( wifi_delays , range(wifi_delays.shape[0]+1,1,-1) ))
	    for pl2 in range (past_delays_lte_fitted.shape[0] , past_delays_lte_fitted.shape[0]+prediction_length):
		 predicted_lte_ma[pl]= numpy.sum(numpy.divide( lte_delays , range(lte_delays.shape[0]+1,1,-1) ))
            return  predicted_wifi_ma, predicted_lte_ma
     
#assigns pkts to slots of each path intermittently 
def round_robin_scheduler(df):
     wifi_scheduled_buffer_rr= numpy.array (range (0,(df.shape[0])*2-1,2) )
     wifi_block_rr = numpy.c_[wifi_scheduled_buffer_rr, df['WiFiSentTimes'], df['WiFiArrivalTimes']]
     lte_scheduled_buffer_rr= numpy.array (range (1,(df.shape[0])*2,2) )
     lte_block_rr= numpy.c_[lte_scheduled_buffer_rr, df['LTESentTimes'], df['LTEArrivalTimes'] ]
     snd_rcvd_block_rr = numpy.r_ [wifi_block_rr, lte_block_rr ]
     return  snd_rcvd_block_rr
####################################################################################################################
#####edpf estimates packet delivery times based on average transmision delays#######################################   
def edpf_scheduler(df, pkt_length, nof_duplicates):
     wifi_buffer_slot=nof_duplicates
     lte_buffer_slot=nof_duplicates
     max_slots=df.shape[0] # maximum number of slots per path because of our traces
     A=[0 , 0] #estimated time that wireless channel becomes available for tranmission
     wifi_scheduled_buffer_edpf=range(0, nof_duplicates)
     lte_scheduled_buffer_edpf=range(0, nof_duplicates)
     estimated_wifi_dlv_edpf=range(0, nof_duplicates)
     estimated_lte_dlv_edpf=range(0, nof_duplicates)
     avg_delay = numpy.asarray( [ numpy.mean( numpy.asarray(df.iloc[:]['WiFiDelayPackets'])),  numpy.mean( numpy.asarray(df.iloc[:]['LTEDelayPackets']) ) ] ) # avergae over bw of each packet
     for pkt in range(nof_duplicates, (df.shape[0])*2 - nof_duplicates):
            
            if (wifi_buffer_slot< max_slots  and lte_buffer_slot < max_slots):
		    estimated_wifi_dlv= max( df.iloc[wifi_buffer_slot]['WiFiSentTimes'], A[0] )  +  avg_delay[0] 
		    estimated_lte_dlv=  max (df.iloc[lte_buffer_slot]['LTESentTimes'], A[1] )  + avg_delay[1] 
		    if (estimated_wifi_dlv < estimated_lte_dlv):
			wifi_scheduled_buffer_edpf.append(pkt)
		        estimated_wifi_dlv_edpf=numpy.append(estimated_wifi_dlv_edpf,estimated_wifi_dlv)
			wifi_buffer_slot=wifi_buffer_slot+1
                    else:
		        lte_scheduled_buffer_edpf.append(pkt)
                        estimated_lte_dlv_edpf=numpy.append(estimated_lte_dlv_edpf, estimated_lte_dlv)
			lte_buffer_slot=lte_buffer_slot+1
            elif (wifi_buffer_slot >= max_slots):
		        lte_scheduled_buffer_edpf.append(pkt)
			estimated_lte_dlv_edpf= numpy.append( estimated_lte_dlv_edpf, max( df.iloc[lte_buffer_slot]['LTESentTimes'], A[0] )  + avg_delay[0] )
			lte_buffer_slot=lte_buffer_slot+1
            elif (lte_buffer_slot >= max_slots):
		        wifi_scheduled_buffer_edpf.append(pkt)
			estimated_wifi_dlv_edpf= numpy.append( estimated_wifi_dlv_edpf, max (df.iloc[wifi_buffer_slot]['WiFiSentTimes'], A[1] )  + avg_delay[1] )
                        wifi_buffer_slot=wifi_buffer_slot+1
        
     estimated_dlv_edpf =  numpy.append (numpy.asarray(estimated_wifi_dlv_edpf), numpy.asarray(estimated_lte_dlv_edpf) )
     wifi_block_edpf = numpy.c_[wifi_scheduled_buffer_edpf, numpy.zeros((len(wifi_scheduled_buffer_edpf), 1)), estimated_wifi_dlv_edpf, df['WiFiSentTimes'], df['WiFiArrivalTimes'] ]
     lte_block_edpf= numpy.c_[lte_scheduled_buffer_edpf, numpy.ones((len(lte_scheduled_buffer_edpf), 1)),estimated_lte_dlv_edpf, df['LTESentTimes'], df['LTEArrivalTimes'] ]
     snd_rcvd_block_edpf = numpy.r_ [wifi_block_edpf, lte_block_edpf ]
     return  snd_rcvd_block_edpf
####################################################################################################################
###############################################sedpf ###############################################################
def sedpf2_scheduler(df,snd_rcvd_block_last_sedpf_wifi,snd_rcvd_block_last_sedpf_lte, Z_mean, Z_sigma, length_Delta_slot):
     
     nof_duplicates=snd_rcvd_block_last_sedpf_wifi.shape[0]
     wifi_buffer_slot_sedpf=nof_duplicates
     lte_buffer_slot_sedpf=nof_duplicates
     max_slots=df.shape[0] # maximum number of slots per path because of our traces 
     wifi_scheduled_buffer_sedpf=range(0, nof_duplicates)
     lte_scheduled_buffer_sedpf=range(0, nof_duplicates)
     estimated_wifi_dlv_sedpf=range(0, nof_duplicates)
     estimated_lte_dlv_sedpf=range(0, nof_duplicates) 
       
     for pkt in range(nof_duplicates, (df.shape[0])*2 - nof_duplicates):
            print "pkt %d" %(pkt)
            
            if (wifi_buffer_slot_sedpf < max_slots  and lte_buffer_slot_sedpf < max_slots):
		    send_times= [ df.iloc [wifi_buffer_slot_sedpf]['WiFiSentTimes'], df.iloc [lte_buffer_slot_sedpf]['LTESentTimes'] ]
		    dynamicrfwifi, dynamicrflte= update_delay_profiles_sedpf (df[:nof_duplicates], send_times, snd_rcvd_block_last_sedpf_wifi[nof_duplicates:wifi_buffer_slot_sedpf,:], snd_rcvd_block_last_sedpf_lte[nof_duplicates:lte_buffer_slot_sedpf:], wifi_buffer_slot_sedpf,lte_buffer_slot_sedpf,  nof_duplicates)
		        
		    Rcvd_arrival_stat_wifi=df.iloc[0:dynamicrfwifi.shape[0]]['WiFiSentTimes'] + dynamicrfwifi.iloc[:]['wifi']
		    Rcvd_arrival_stat_lte=df.iloc[0: dynamicrflte.shape[0]]['LTESentTimes'] +  dynamicrflte.iloc[:]['lte']
		   
		    wifi_nan=numpy.where(numpy.isnan(Rcvd_arrival_stat_wifi))[0]
		    lte_nan=numpy.where(numpy.isnan(Rcvd_arrival_stat_lte))[0]   
		    if not len(wifi_nan)==0:
			    if numpy.std(Rcvd_arrival_stat_wifi)==0:
				Rcvd_arrival_stat_wifi[numpy.isnan(Rcvd_arrival_stat_wifi)]=numpy.mean(Rcvd_arrival_stat_wifi)
			    else:    
				Rcvd_arrival_stat_wifi[numpy.isnan(Rcvd_arrival_stat_wifi)]=numpy.random.normal(numpy.mean(Rcvd_arrival_stat_wifi),numpy.std(Rcvd_arrival_stat_wifi), len(wifi_nan)) +numpy.array(Rcvd_arrival_stat_wifi[numpy.isfinite(Rcvd_arrival_stat_wifi)])[-1]
		    if not len(lte_nan)==0:		
			    if  numpy.std(Rcvd_arrival_stat_lte)==0:
				Rcvd_arrival_stat_lte[numpy.isnan(Rcvd_arrival_stat_lte)]=numpy.mean(Rcvd_arrival_stat_lte)
			    else:
		                Rcvd_arrival_stat_lte[numpy.isnan(Rcvd_arrival_stat_lte)]=numpy.random.normal(numpy.mean(Rcvd_arrival_stat_lte),numpy.std(Rcvd_arrival_stat_lte), len(lte_nan)) + numpy.array (Rcvd_arrival_stat_lte[numpy.isfinite(Rcvd_arrival_stat_lte)])[-1]
		    
		    rcvd_index_wifi= numpy.asarray (Rcvd_arrival_stat_wifi.index[Rcvd_arrival_stat_wifi.apply(numpy.isfinite)] )
		    rcvd_index_lte= numpy.asarray (Rcvd_arrival_stat_lte.index[Rcvd_arrival_stat_lte.apply(numpy.isfinite)]) 
		    if wifi_buffer_slot_sedpf < length_Delta_slot[0] and lte_buffer_slot_sedpf > length_Delta_slot[1]:             
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(wifi_buffer_slot_sedpf),  lte_buffer_slot_sedpf- int(length_Delta_slot[1])])
		    elif wifi_buffer_slot_sedpf > length_Delta_slot[0] and lte_buffer_slot_sedpf< length_Delta_slot[1]:             
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(length_Delta_slot[0]),  lte_buffer_slot_sedpf- int(lte_buffer_slot_sedpf)])
		    elif wifi_buffer_slot_sedpf < length_Delta_slot[0] and lte_buffer_slot_sedpf< length_Delta_slot[1]:             
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(wifi_buffer_slot_sedpf),  lte_buffer_slot_sedpf- int(lte_buffer_slot_sedpf)])  
		    else:            
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(length_Delta_slot[0]),  lte_buffer_slot_sedpf- int(length_Delta_slot[1])])  
		    afirst=[Rcvd_arrival_stat_wifi[afirst_index[0]], Rcvd_arrival_stat_lte[afirst_index[1]] ]    
		    Z_wifi= numpy.random.normal(Z_mean[0], Z_sigma[0])
		    Z_lte= numpy.random.normal(Z_mean[1], Z_sigma[1] )
		    Y_wifi= Z_wifi + afirst[0]
		    Y_lte= Z_lte + afirst[1]
		    LinkStats_wifi= numpy.c_[numpy.r_[ (Z_mean+afirst).T, Z_mean[0]+afirst[0]] , numpy.r_[Z_sigma.T,Z_sigma[0]]]
		    LinkStats_lte= numpy.c_[numpy.r_[ (Z_mean+afirst).T, Z_mean[1]+afirst[1]] , numpy.r_[Z_sigma.T,Z_sigma[1]]]
		    mg_wifi=MaxofGaussians(LinkStats_wifi)
		    mg_lte=MaxofGaussians(LinkStats_lte)
		    
		    best_path=numpy.argmin( [mg_wifi.gaussian_approx()[0], mg_lte.gaussian_approx()[0] ]) 
                    if (best_path==0):
			wifi_scheduled_buffer_sedpf.append(pkt)
		        estimated_wifi_dlv_sedpf=numpy.append(estimated_wifi_dlv_sedpf, Y_wifi)
		        snd_rcvd_block_sedpf_wifi=numpy.c_[pkt, 0, Y_wifi, df.iloc[wifi_buffer_slot_sedpf]['WiFiSentTimes'], df.iloc[wifi_buffer_slot_sedpf]['WiFiArrivalTimes'] ]
		        snd_rcvd_block_last_sedpf_wifi  =numpy.r_[snd_rcvd_block_last_sedpf_wifi, snd_rcvd_block_sedpf_wifi]  
			wifi_buffer_slot_sedpf=wifi_buffer_slot_sedpf+1
                    else:
		        lte_scheduled_buffer_sedpf.append(pkt)
                        estimated_lte_dlv_sedpf=numpy.append(estimated_lte_dlv_sedpf, Y_lte)
                        snd_rcvd_block_sedpf_lte=numpy.c_[pkt, 1, Y_lte, df.iloc[lte_buffer_slot_sedpf]['LTESentTimes'], df.iloc[lte_buffer_slot_sedpf]['LTEArrivalTimes'] ]
                        snd_rcvd_block_last_sedpf_lte  =numpy.r_[snd_rcvd_block_last_sedpf_lte, snd_rcvd_block_sedpf_lte]        
			lte_buffer_slot_sedpf=lte_buffer_slot_sedpf+1
            elif (wifi_buffer_slot_sedpf >= max_slots and lte_buffer_slot_sedpf < max_slots):
                        Z_lte= numpy.random.normal(Z_mean[1], Z_sigma[1] )
                        Y_lte= Z_lte + afirst[1]
		        lte_scheduled_buffer_sedpf.append(pkt)
			estimated_lte_dlv_sedpf= numpy.append( estimated_lte_dlv_sedpf, Y_lte)
			snd_rcvd_block_sedpf_lte=numpy.c_[pkt, 1, Y_lte, df.iloc[lte_buffer_slot_sedpf]['LTESentTimes'], df.iloc[lte_buffer_slot_sedpf]['LTEArrivalTimes'] ]
			snd_rcvd_block_last_sedpf_lte  =numpy.r_[snd_rcvd_block_last_sedpf_lte, snd_rcvd_block_sedpf_lte]    
			lte_buffer_slot_sedpf=lte_buffer_slot_sedpf+1
            elif (lte_buffer_slot_sedpf >= max_slots and wifi_buffer_slot_sedpf < max_slots):
                        Z_wifi= numpy.random.normal(Z_mean[0], Z_sigma[0])
                        Y_wifi= Z_wifi + afirst[0]
		        wifi_scheduled_buffer_sedpf.append(pkt)
			estimated_wifi_dlv_sedpf= numpy.append( estimated_wifi_dlv_sedpf,  Y_wifi)
			snd_rcvd_block_sedpf_wifi=numpy.c_[pkt, 0, Y_wifi, df.iloc[wifi_buffer_slot_sedpf]['WiFiSentTimes'], df.iloc[wifi_buffer_slot_sedpf]['WiFiArrivalTimes'] ] 
			snd_rcvd_block_last_sedpf_wifi  =numpy.r_[snd_rcvd_block_last_sedpf_wifi, snd_rcvd_block_sedpf_wifi]     
                        wifi_buffer_slot_sedpf=wifi_buffer_slot_sedpf+1
                        

    
     estimated_dlv_sedpf =  numpy.append (numpy.asarray(estimated_wifi_dlv_sedpf), numpy.asarray(estimated_lte_dlv_sedpf) )
     wifi_block_sedpf = numpy.c_[wifi_scheduled_buffer_sedpf, numpy.zeros((len(wifi_scheduled_buffer_sedpf), 1)), estimated_wifi_dlv_sedpf, df['WiFiSentTimes'], df['WiFiArrivalTimes'] ]
     lte_block_sedpf= numpy.c_[lte_scheduled_buffer_sedpf, numpy.ones((len(lte_scheduled_buffer_sedpf), 1)),estimated_lte_dlv_sedpf, df['LTESentTimes'], df['LTEArrivalTimes'] ]
     snd_rcvd_block_sedpf = numpy.r_ [wifi_block_sedpf, lte_block_sedpf ]
     pdb.set_trace()
     return  snd_rcvd_block_sedpf     

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
     
     # initialise 
     n_slots=df.shape[0] # buffer size per path(link)
     pkt_set=range(0, 2*n_slots) # sum of  the n_slots on each path
     pkt_length=1440 # packet length in bytes
     nof_pmf_calls = 0 
     max_nof_predictions_pmf=50
     nof_probes_last = 0 # to make sure that we go through pmf predictions for the first round
     nof_predictions_after_current_slot=20 # number of predicted delays that    (block length)            
     nof_probed_slots = 0 # start with no feedback
     probed_bar=0 
     mu_predicted_wifi=[]  # pmf statistics
     mu_predicted_lte=[]
     sigma_predicted_wifi=[]
     sigma_predicted_lte=[]
     mu_rcvd_wifi=[]  # real feedback statistics
     mu_rcvd_lte=[]
     sigma_rcvd_wifi=[]
     sigma_rcvd_lte=[]
     n_runs=1  # number of simulations
     n_schedulers=6
     expected_reordering_delay = numpy.zeros((n_runs, n_schedulers ))
     std_reordering_delay = numpy.zeros((n_runs, n_schedulers ))
     RMSE = numpy.zeros((n_runs, n_schedulers-1 ))
     initial_window_sedpf=160 
     best=[] # record of best paths- sedpf
     stop_sedpf=False
     partial_s_i =1000 # slot at which the second path becomes available
     for s_i  in range (5, n_slots):
            
            # if only LTE path is available
            if s_i < partial_s_i:
                 send_times= df.iloc [s_i]['LTESentTimes'] 
                 wifi_fb, lte_fb  = update_delay_profiles (  send_times, snd_rcvd_block_last_lte[:s_i,:], s_i, [])
            else:
                 s_i_wifi=s_i- partial_s_i 
            	 send_times= [df.iloc [s_i_wifi]['WiFiSentTimes'],  df.iloc [s_i]['LTESentTimes'] ]
                 wifi_fb, lte_fb  = update_delay_profiles ( send_times,snd_rcvd_block_last_lte[:s_i_wifi,:], snd_rcvd_block_last_lte[:s_i,:], s_i, s_i_wifi)
            
            ratings=update_list_ratings(wifi_fb, lte_fb)
         
            #scheduling for RR
   	    print "Round Robin scheduler running.. "
            snd_rcvd_rr=round_robin_scheduler(df)
            expected_reordering_delay[n_runs-1,0],std_reordering_delay[n_runs-1,0], rr_sequenced  = reordering_stats (snd_rcvd_rr[nof_duplicate_pkts:,])
            print "Round Robin scheduler finished.Expected reordering delay is:  %f" %expected_reordering_delay[n_runs-1,0]
 
	    #scheduling for EDPF
	    print "EDPF scheduler running.."
	    snd_rcvd_edpf=edpf_scheduler(df,pkt_length,nof_duplicate_pkts)
	    expected_reordering_delay[n_runs-1,1],std_reordering_delay[n_runs-1,1], edpf_sequenced = reordering_stats(snd_rcvd_edpf[nof_duplicate_pkts:,])
	    RMSE[n_runs-1,0]=numpy.sqrt(numpy.mean((snd_rcvd_edpf[nof_duplicate_pkts:,2]-snd_rcvd_edpf[nof_duplicate_pkts:,4])**2))
	    print "EDPF scheduler finished. Expected reordering delay is:  %f" %expected_reordering_delay[n_runs-1,1]

	    if len(ratings==0):
	         if s_i < partial_s_i:
	             snd_rcvd_block_last_wifi=[]
	             first_sentTimes_lte= df.iloc [0:s_i]['LTESentTimes'] 
	             first_arrivals_lte= df.iloc[0:s_i]['LTEArrivalTimes'] 
	             snd_rcvd_block_last_lte= numpy.c_[numpy.array(range(0,s_i)), 1*numpy.ones(s_i), numpy.zeros(s_i) , first_sentTimes_lte, first_arrivals_lte ]
	         elif s_i >= partial_s_i:
	             first_sentTimes_wifi= df.iloc [0:s_i_wifi]['WiFiSentTimes'] 
	             first_arrivals_wifi= df.iloc[0:s_i]['WiFiArrivalTimes'] 
	             snd_rcvd_block_last_wifi= numpy.c_[partial_s_i+s_i_wifi, 0, 0 , first_sentTimes_wifi, first_arrivals_wifi ]   
	             s_i_wifi=s_i_wifi+1 
                 ## initialise sedpf 1st and 2nd moment stats from all measurements
		 increment_wifi=numpy.zeros(df.shape[0]-initial_window_sedpf)
		 increment_lte=numpy.zeros(df.shape[0]-initial_window_sedpf)
                 for w in range(initial_window_sedpf, df.shape[0]):
                    increment_wifi[w-initial_window_sedpf]= df.iloc[w]['WiFiArrivalTimes'] -df.iloc[w-initial_window_sedpf]['WiFiArrivalTimes']
                    increment_lte[w-initial_window_sedpf]= df.iloc[w]['LTEArrivalTimes'] -df.iloc[w-initial_window_sedpf]['LTEArrivalTimes']
                 Z_mean=numpy.array ([ numpy.mean(  increment_wifi) , numpy.mean( increment_lte) ] )
                 Z_sigma=numpy.array ( [ numpy.std(  increment_wifi) , numpy.std( increment_lte) ]	)	
                 length_Delta_time=  numpy.maximum ( 3*Z_sigma - Z_mean, 3*Z_sigma + Z_mean )  #  in time
	         length_Delta_slot= length_Delta_time / numpy.mean (numpy.subtract(df.iloc[1:]['WiFiArrivalTimes'],df.iloc[0:df.shape[0]-1]['WiFiArrivalTimes']))
	              
	    elif len(ratings) > 0 :
		     ratings_array=numpy.array(ratings)
		     length_lte_ratings=ratings_array[ratings_array[:,0]==1,1].shape[0]
		     length_wifi_ratings=ratings_array[ratings_array[:,0]==0,1].shape[0]
		     if length_wifi_ratings == 0: 
				last_wifi_idx=0
		     else:
			        last_wifi_idx=int ( numpy.max( ratings_array[ratings_array[:,0]==0,1] ) )
		     if length_lte_ratings == 0: 
				last_lte_idx=0
		     else:
			        last_lte_idx=int ( numpy.max( ratings_array[ratings_array[:,0]==1,1] ) )
		     nof_probed_slots= int ( numpy.max( ratings_array[:,1] ) +1 )
                     
                     
                     ########################SEDPF######################################
		     print "SEDPF scheduler running.."
                     snd_rcvd_sedpf= sedpf2_scheduler(df,snd_rcvd_block_last_wifi,snd_rcvd_block_last_lte, Z_mean, Z_sigma, length_Delta_slot) 
		     expected_reordering_delay[n_runs-1,2],std_reordering_delay[n_runs-1,2], sedpf_sequenced = reordering_stats(snd_rcvd_sedpf[nof_duplicate_pkts:,])
		     RMSE[n_runs-1,1]=numpy.sqrt(numpy.mean((snd_rcvd_sedpf[nof_duplicate_pkts:,2]-snd_rcvd_edpf[nof_duplicate_pkts:,4])**2))
		     print "SEDPF scheduler finished. Expected reordering delay is:  %f" %expected_reordering_delay[n_runs-1,2]

                     pdb.set_trace()
		              

                
                ##################################### calling Moving averages##########################
                start_time_ma=timeit.default_timer() # time the execution of the ma and scheduler's sort
                nof_ma_calls= nof_ma_calls+1
                print "calling MA+MMC at slot  %d for %dth time"   %(s_i ,nof_pmf_calls)
		
                predicted_delays_wifi_ma, predicted_delays_lte_ma= moving_average(wifi_fb,lte_fb,s_i,s_i_wifi, nof_predictions_after_current_slot, False)
                if n_slots < predicted_delays_wifi_ma.shape[0]-1:
                    predicted_delays_wifi_ma=numpy.delete(predicted_delays_wifi_ma,range(n_slots, predicted_delays_wifi_ma.shape[0]) ,0)
                    predicted_delays_lte_ma=numpy.delete(predicted_delays_lte_ma,range(n_slots, predicted_delays_lte_ma.shape[0]) ,0)
                
                #mmc instead of moving averages
                col_mean, col_std, predicted_delays_wifi_mmc, predicted_delays_lte_mmc= moving_average(wifi_fb,lte_fb,s_i,s_i_wifi, nof_predictions_after_current_slot, True )
                mu_rcvd_wifi.append(col_mean[0])
                mu_rcvd_lte.append(col_mean[1])
                sigma_rcvd_wifi.append(col_std[0])
                sigma_rcvd_lte.append(col_std[1])
                if n_slots < predicted_delays_mmc.shape[0]-1:
                    predicted_delays_wifi_mmc=numpy.delete(predicted_delays_wifi_mmc,range(n_slots, predicted_delays_wifi_mmc.shape[0]) ,0)
                    predicted_delays_lte_mmc=numpy.delete(predicted_delays_lte_mmc,range(n_slots, predicted_delays_lte_mmc.shape[0]) ,0)
                
                #MA sorting
                remaining_pkt_set_ma=numpy.delete (pkt_set,  numpy.union1d(snd_rcvd_block_last_ma_wifi [0: s_i , 0], snd_rcvd_block_last_ma_lte [0: s_i , 0]) )
                snd_rcvd_block_ma_wifi, snd_rcvd_block_ma_lte = sort_block_packets(predicted_delays_ma,s_i, df, remaining_pkt_set_ma)
                snd_rcvd_block_last_ma_wifi=numpy.r_ [ snd_rcvd_block_last_ma_wifi [0:s_i, ]  , snd_rcvd_block_ma_wifi ]
                snd_rcvd_block_last_ma_lte=numpy.r_ [ snd_rcvd_block_last_ma_lte [0:s_i, ]  , snd_rcvd_block_ma_lte ]

                #MMC sorting
                remaining_pkt_set_mmc=numpy.delete (pkt_set,  numpy.union1d(snd_rcvd_block_last_mmc_wifi [0: s_i , 0], snd_rcvd_block_last_mmc_lte [0: s_i , 0]) )
                snd_rcvd_block_mmc_wifi, snd_rcvd_block_mmc_lte = sort_block_packets(predicted_delays_mmc,s_i, df, remaining_pkt_set_mmc)
                snd_rcvd_block_last_mmc_wifi=numpy.r_ [ snd_rcvd_block_last_mmc_wifi [0:s_i, ]  , snd_rcvd_block_mmc_wifi ]
                snd_rcvd_block_last_mmc_lte=numpy.r_ [ snd_rcvd_block_last_mmc_lte [0:s_i, ]  , snd_rcvd_block_mmc_lte ]



                elapsed_ma=timeit.default_timer() - start_time_ma
                print "MA+MMC for slot no. %d is finished in %f sec" %(s_i,elapsed_ma)


                
                
                
                
     finished=1
     pdb.set_trace()
     sedpf_data=snd_rcvd_sedpf
     ma_data=numpy.r_[snd_rcvd_block_last_ma_wifi[nof_duplicate_pkts:n_slots], snd_rcvd_block_last_ma_lte[nof_duplicate_pkts:n_slots] ]
     mmc_data=numpy.r_[snd_rcvd_block_last_mmc_wifi[nof_duplicate_pkts:n_slots], snd_rcvd_block_last_mmc_lte[nof_duplicate_pkts:n_slots] ]
     expected_reordering_delay[n_runs-1,2],std_reordering_delay[n_runs-1,2], sedpf_sequenced  = reordering_stats(sedpf_data)
     expected_reordering_delay[n_runs-1,4],std_reordering_delay[n_runs-1,4], ma_sequenced = reordering_stats(ma_data)
     expected_reordering_delay[n_runs-1,5],std_reordering_delay[n_runs-1,5], mmc_sequenced  = reordering_stats(mmc_data)
     

     RMSE[n_runs-1, 1]= numpy.sqrt(numpy.mean((sedpf_data[:,2]-sedpf_data[:,4])**2))
     RMSE[n_runs-1, 3]=numpy.sqrt(numpy.mean((ma_data[:,2]-ma_data[:,4])**2))
     RMSE[n_runs-1, 4]=numpy.sqrt(numpy.mean((mmc_data[:,2]-mmc_data[:,4])**2))
     
     
     plt.figure(1)
     sedpf=plt.scatter( range(nof_duplicate_pkts,nof_duplicate_pkts+ sedpf_sequenced.shape[0]),  sedpf_sequenced[:,-1]-sedpf_sequenced[:,-3],marker='o', facecolors='none', color= 'k', s=8  , label='SEDPF') 
     mcm=plt.scatter( range(nof_duplicate_pkts,nof_duplicate_pkts+mmc_sequenced.shape[0]),  mmc_sequenced[:,-1]- mmc_sequenced[:,-3], marker= 'v', facecolors='none', color= 'g' , s=8, label='MCM') 
     rr=plt.scatter( range(nof_duplicate_pkts,rr_sequenced.shape[0]+nof_duplicate_pkts),  rr_sequenced[:,-1]-rr_sequenced[:,-3], marker= '^', facecolors='none', color= 'b', s=8, label='Round Robin' ) 
     plt.legend(loc='upper right', frameon=False,  markerscale=4., scatterpoints=1, fontsize=14)
     plt.xlabel('packet number', fontsize=20)
     plt.ylabel('Reordering delay (s)', fontsize=20)
     plt.axis([0.0, 10000,0, .1])
     plt.tick_params(labelsize=20)
     plt.show()
     
     plt.figure(2)
     re_sedpf=sedpf_sequenced[:,-1]-sedpf_sequenced[:,-3]
     re_edpf=edpf_sequenced[:,-1]-edpf_sequenced[:,-3]
     re_edpf=edpf_sequenced[:,-1]-edpf_sequenced[:,-3]
     plt.hist(re_sedpf,1000, lw=2, color='red', label='SEDPF')
     plt.hist(re_edpf, 300, lw=2, color='cyan', label='EDPF')
     plt.legend(loc='upper right', frameon=False, fontsize=14)
     plt.xlabel('re-ordering delay (s)', fontsize=20)
     plt.ylabel('Frequency', fontsize=20)
     plt.axis([0, 0.05, 0.0, 5000])
     plt.tick_params(labelsize=20)
     plt.show()
     
     
    
