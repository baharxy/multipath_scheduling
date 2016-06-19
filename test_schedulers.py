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


############ get the feedback times assuming all transmissions receievd##
def delay_profiles(df, slot):

             wifi_rcvd_delay_list = df[df['feedbackreceivedWiFi'] < df.iloc [slot]['WiFiSentTimes'] ].index.tolist()
             lte_rcvd_delay_list = df[df['feedbackreceivedLTE'] < df.iloc [slot]['LTESentTimes' ] ].index.tolist()
	     delayDF = pd.DataFrame(numpy.nan, index= range(0,slot), columns=['wifi', 'lte'] )
	     delayDF.wifi [ wifi_rcvd_delay_list ] = df.WiFiDelayPackets [wifi_rcvd_delay_list]
	     delayDF.lte [ lte_rcvd_delay_list  ] =  df.LTEDelayPackets [lte_rcvd_delay_list]      
	     return delayDF
########### get the feedback based on previous transmissions#######################
def update_delay_profiles(df_duplicates, current_sent_times ,feedbackarray_wifi, feedbackarray_lte,slot, nof_duplicates):
   
 
     sent_wifi_slots= feedbackarray_wifi.shape[0]
     sent_lte_slots=   feedbackarray_lte.shape[0]
     feedback_times_wifi =  feedbackarray_wifi  [: , 3]+ 2* ( feedbackarray_wifi [: , 4]-  feedbackarray_wifi [:, 3] )
     feedback_times_lte =  feedbackarray_lte  [: , 3]+ 2* ( feedbackarray_lte [: , 4]-  feedbackarray_lte [:, 3] ) 
     rcvd_wifi_slots=numpy.array ( [ map(int, x) for x in numpy.where(feedback_times_wifi  <   current_sent_times[0]) ][0] ) + nof_duplicates
     rcvd_lte_slots=numpy.array( [ map(int, x) for x in numpy.where(feedback_times_lte <   current_sent_times[1]) ][0] ) +nof_duplicates
     if sent_wifi_slots==sent_lte_slots :
            wifi_rcvd_delay_list = numpy.append(  rcvd_wifi_slots  , numpy.asarray( df_duplicates[df_duplicates['feedbackreceivedWiFi'] <  current_sent_times[0]  ].index.tolist()) )
            lte_rcvd_delay_list= numpy.append(  rcvd_lte_slots ,  numpy.asarray (df_duplicates[df_duplicates ['feedbackreceivedLTE'] <  current_sent_times[1]    ].index.tolist()) )
     else:
          pdb.set_trace()
          #throw an expetion
          raise ValueError, 'Something is wong'

     delayDF = pd.DataFrame(numpy.nan, index= range(0,slot), columns=['wifi', 'lte'] )   
     delayDF.wifi [ wifi_rcvd_delay_list ] = df.WiFiDelayPackets [wifi_rcvd_delay_list]
     delayDF.lte [ lte_rcvd_delay_list  ] =  df.LTEDelayPackets [lte_rcvd_delay_list]
     return delayDF
##### get the feedback based on previous transmissions/ seperate out wifi and lte#######################     
def update_delay_profiles_sedpf(df_duplicates, current_sent_times ,feedbackarray_wifi, feedbackarray_lte,slot_wifi,slot_lte, nof_duplicates):
   
     
     sent_wifi_slots= feedbackarray_wifi.shape[0]
     sent_lte_slots=   feedbackarray_lte.shape[0]
     feedback_times_wifi =  feedbackarray_wifi  [: , 3]+ 2* ( feedbackarray_wifi [: , 4]-  feedbackarray_wifi [:, 3] )
     feedback_times_lte =  feedbackarray_lte  [: , 3]+ 2* ( feedbackarray_lte [: , 4]-  feedbackarray_lte [:, 3] ) 
     rcvd_wifi_slots=numpy.array ( [ map(int, x) for x in numpy.where(feedback_times_wifi  <   current_sent_times[0]) ][0] ) + nof_duplicates
     rcvd_lte_slots=numpy.array( [ map(int, x) for x in numpy.where(feedback_times_lte <   current_sent_times[1]) ][0] ) +nof_duplicates
     wifi_rcvd_delay_list = numpy.append(  rcvd_wifi_slots  , numpy.asarray( df_duplicates[df_duplicates['feedbackreceivedWiFi'] <  current_sent_times[0]  ].index.tolist()) )
     lte_rcvd_delay_list= numpy.append(  rcvd_lte_slots ,  numpy.asarray (df_duplicates[df_duplicates ['feedbackreceivedLTE'] <  current_sent_times[1]    ].index.tolist()) )
     
     wifidelays = pd.DataFrame(numpy.nan, index= range(0,slot_wifi), columns=['wifi'] )
     ltedelays = pd.DataFrame(numpy.nan, index= range(0,slot_lte), columns=['lte'] )
     wifidelays.wifi[ wifi_rcvd_delay_list ] = df.WiFiDelayPackets [wifi_rcvd_delay_list]   
     ltedelays.lte[ lte_rcvd_delay_list  ] =  df.LTEDelayPackets [lte_rcvd_delay_list]
     return wifidelays,ltedelays

#######change the format of feedback to tuples######
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
     

########gets the pmf data############################
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
def moving_average (feedbacks, slot_n, prediction_length, mmc):
     
     past_delays_fitted=numpy.asarray(feedbacks)
     col_mean = stats.nanmean(past_delays_fitted,axis=0)
     col_std = stats.nanstd(past_delays_fitted,axis=0)
     inds = numpy.where(numpy.isnan(past_delays_fitted))
     past_delays_fitted[inds]=numpy.take(col_mean,inds[1])
     wifi_delays=past_delays_fitted[:,0]
     lte_delays=past_delays_fitted[:,1]
     
     if mmc: 
            
	    forward_predicted_delays_mmc=numpy.c_[numpy.random.normal(col_mean[0], col_std[0], prediction_length),  numpy.random.normal(col_mean[1], col_std[1], prediction_length )]
            forward_predicted_delays_mmc[ forward_predicted_delays_mmc <0 ]=0 # truncate negative samples 
            predicted_ma=numpy.r_ [past_delays_fitted, forward_predicted_delays_mmc]
            return  col_mean, col_std, predicted_ma 
     else:
            predicted_ma=numpy.r_ [past_delays_fitted, numpy.zeros((prediction_length,2)) ]
	    for pl in range (past_delays_fitted.shape[0] , past_delays_fitted.shape[0]+prediction_length):
		 predicted_ma[pl,0]= numpy.sum(numpy.divide( wifi_delays , range(wifi_delays.shape[0]+1,1,-1) ))
		 predicted_ma[pl,1]= numpy.sum(numpy.divide( lte_delays , range(lte_delays.shape[0]+1,1,-1) ) )
            
            return  predicted_ma
     
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
     
     return  wifi_block_edpf, lte_block_edpf
#############################fill previous arrivals, cause sedpf needs it #########################################
def update_arrivals_first(Rcvd_wifi,Rcvd_lte,delta,Z_mean, Z_sigma):
     
     wifi_nan=numpy.where(numpy.isnan(Rcvd_wifi))[0]
     lte_nan=numpy.where(numpy.isnan(Rcvd_lte))[0]  
     wifi_finite=numpy.where(numpy.isfinite(Rcvd_wifi))[0]
     lte_finite=numpy.where(numpy.isfinite(Rcvd_lte))[0] 
        
     if ( not len(wifi_nan)==0 or not len(lte_nan)==0 ):
        for wifi_finite_elements in wifi_finite:
         if wifi_finite_elements-int(delta[0])>=0:
             if numpy.isnan(Rcvd_wifi[wifi_finite_elements-int(delta[0])]):
           	Rcvd_wifi[wifi_finite_elements-int(delta[0])]=Rcvd_wifi[wifi_finite_elements]-numpy.random.normal(Z_mean[0],Z_sigma[0],1)  
         if wifi_finite_elements+int(delta[0])< Rcvd_wifi.shape[0]:
             if numpy.isnan(Rcvd_wifi[wifi_finite_elements+int(delta[0])]):
           	Rcvd_wifi[wifi_finite_elements+int(delta[0])]=Rcvd_wifi[wifi_finite_elements]+numpy.random.normal(Z_mean[0],Z_sigma[0],1)  
           	  	
        for lte_finite_elements in lte_finite:
          if lte_finite_elements-int(delta[1])>=0:
             if numpy.isnan(Rcvd_lte[lte_finite_elements-int(delta[1])]) :
           	Rcvd_lte[lte_finite_elements-int(delta[1])]=Rcvd_lte[lte_finite_elements]-numpy.random.normal(Z_mean[1],Z_sigma[1],1)    
          if lte_finite_elements+int(delta[1])< Rcvd_lte.shape[0]:
             if numpy.isnan(Rcvd_lte[lte_finite_elements+int(delta[1])]) :
           	Rcvd_lte[lte_finite_elements+int(delta[1])]=Rcvd_lte[lte_finite_elements]+numpy.random.normal(Z_mean[1],Z_sigma[1],1)    
           	 	
        return Rcvd_wifi, Rcvd_lte   
        	    	
        Rcvd_wifi, Rcvd_lte= update_arrivals_first(Rcvd_wifi,Rcvd_lte,delta, Z_mean, Z_sigma) 
             
     else:
        return Rcvd_wifi, Rcvd_lte 
     
            
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
            if (pkt == 50):
            	a=1
            	pdb.set_trace()
            if (wifi_buffer_slot_sedpf < max_slots  and lte_buffer_slot_sedpf < max_slots):
		    send_times= [ df.iloc [wifi_buffer_slot_sedpf]['WiFiSentTimes'], df.iloc [lte_buffer_slot_sedpf]['LTESentTimes'] ]
		    dynamicrfwifi, dynamicrflte= update_delay_profiles_sedpf (df[:nof_duplicates], send_times, snd_rcvd_block_last_sedpf_wifi[nof_duplicates:wifi_buffer_slot_sedpf,:], snd_rcvd_block_last_sedpf_lte[nof_duplicates:lte_buffer_slot_sedpf:], wifi_buffer_slot_sedpf,lte_buffer_slot_sedpf,  nof_duplicates)
		    
		    Rcvd_wifi=df.iloc[0:dynamicrfwifi.shape[0]]['WiFiSentTimes'] + dynamicrfwifi.iloc[:]['wifi']
		    Rcvd_lte=df.iloc[0: dynamicrflte.shape[0]]['LTESentTimes'] +  dynamicrflte.iloc[:]['lte']
		    
		    
		    # fill up the first arrivals based on the levy process
		    Rcvd_arrival_stat_wifi, Rcvd_arrival_stat_lte= update_arrivals_first(Rcvd_wifi,Rcvd_lte,length_Delta_slot,Z_mean, Z_sigma)
		   
		    rcvd_index_wifi= numpy.asarray (Rcvd_arrival_stat_wifi.index[Rcvd_arrival_stat_wifi.apply(numpy.isfinite)] )
		    rcvd_index_lte= numpy.asarray (Rcvd_arrival_stat_lte.index[Rcvd_arrival_stat_lte.apply(numpy.isfinite)]) 
		    
            # what is the distance 

		    if wifi_buffer_slot_sedpf < length_Delta_slot[0] and lte_buffer_slot_sedpf > length_Delta_slot[1]:             
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(wifi_buffer_slot_sedpf),  lte_buffer_slot_sedpf- int(length_Delta_slot[1])])
		    elif wifi_buffer_slot_sedpf > length_Delta_slot[0] and lte_buffer_slot_sedpf< length_Delta_slot[1]:             
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(length_Delta_slot[0]),  lte_buffer_slot_sedpf- int(lte_buffer_slot_sedpf)])
		    elif wifi_buffer_slot_sedpf < length_Delta_slot[0] and lte_buffer_slot_sedpf< length_Delta_slot[1]:             
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(wifi_buffer_slot_sedpf),  lte_buffer_slot_sedpf- int(lte_buffer_slot_sedpf)])  
		    else:            
		      afirst_index=numpy.array ([wifi_buffer_slot_sedpf- int(length_Delta_slot[0]),  lte_buffer_slot_sedpf- int(length_Delta_slot[1])])
		    
		    if numpy.isnan(Rcvd_arrival_stat_wifi[afirst_index[0]]):
		          afirst_index[0]= rcvd_index_wifi[numpy.argmin ( numpy.absolute ( afirst_index[0] - rcvd_index_wifi ))]
		    if numpy.isnan(Rcvd_arrival_stat_lte[afirst_index[1]]):
		          afirst_index[1]= rcvd_index_lte [numpy.argmin ( numpy.absolute ( afirst_index[1] - rcvd_index_lte ))]     
		            
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
                        
	    
     pdb.set_trace()
     estimated_dlv_sedpf =  numpy.append (numpy.asarray(estimated_wifi_dlv_sedpf), numpy.asarray(estimated_lte_dlv_sedpf) )
     wifi_block_sedpf = numpy.c_[wifi_scheduled_buffer_sedpf, numpy.zeros((len(wifi_scheduled_buffer_sedpf), 1)), estimated_wifi_dlv_sedpf, df['WiFiSentTimes'], df['WiFiArrivalTimes'] ]
     lte_block_sedpf= numpy.c_[lte_scheduled_buffer_sedpf, numpy.ones((len(lte_scheduled_buffer_sedpf), 1)),estimated_lte_dlv_sedpf, df['LTESentTimes'], df['LTEArrivalTimes'] ]
     
     return  wifi_block_sedpf, lte_block_sedpf  

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
     expected_time_towait= numpy.zeros((n_runs, n_schedulers ))
     std_reordering_delay = numpy.zeros((n_runs, n_schedulers ))
     RMSE = numpy.zeros((n_runs, n_schedulers-1 ))
     initial_window_sedpf=10 
     best=[] # record of best paths- sedpf
     stop_sedpf=False
    
     for s_i  in range (5, n_slots):
            
            # construct matrix of received delay values assuming each packet sends rtts back
            if nof_pmf_calls==0:                                 
                 dynamicdf = delay_profiles (  df, s_i)
            else:
                 duplicates_df= df[:nof_duplicate_pkts]
                 send_times= [ df.iloc [s_i]['WiFiSentTimes'], df.iloc [s_i]['LTESentTimes'] ]
                 dynamicdf = update_delay_profiles (  duplicates_df, send_times, snd_rcvd_block_last_wifi[nof_duplicate_pkts:s_i,:], snd_rcvd_block_last_lte[nof_duplicate_pkts:s_i,:], s_i,  nof_duplicate_pkts)
            
            (ratings, true_o, true_d) = update_list_ratings(dynamicdf)
            
            if len(ratings) > 0 :
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
                 probed_bar= int( max (0, 10* ( numpy.log(length_lte_ratings)+ numpy.log(length_wifi_ratings) ) ) ) #so we have fair nof lte vs. wifi probes!
#            if probed_bar  >  max_nof_predictions_pmf:
#                 nof_cuts=min (nof_probed_slots-max_nof_predictions_pmf+1, min(last_lte_idx,last_wifi_idx) )
#                 if nof_cuts > 0: 
#                     dynamicdf = dynamicdf.drop(range(0,nof_cuts))
#            (ratings, true_o, true_d) = update_list_ratings(dynamicdf)      
            if (probed_bar > 0 ): # if had 5 more probes compared to the predictions call pmf/ otherwise just copy packets across both paths
                if nof_pmf_calls==0 :
                     # first  packets down both paths to build up some model
                     nof_duplicate_pkts=s_i
                     first_arrivals_best_path= numpy.argmin ( numpy.c_ [ df.iloc[0: nof_duplicate_pkts]['WiFiArrivalTimes'], df.iloc[0: nof_duplicate_pkts]['LTEArrivalTimes'] ],  axis=1 )
                     first_sentTimes=numpy.zeros(first_arrivals_best_path.shape[0])
                     first_sentTimes_wifi= df.iloc [0:nof_duplicate_pkts]['WiFiSentTimes']
                     first_sentTimes_lte= df.iloc [0:nof_duplicate_pkts]['LTESentTimes']  
                     first_arrivals_wifi=  df.iloc[0:nof_duplicate_pkts]['WiFiArrivalTimes']
                     first_arrivals_lte= df.iloc[0:nof_duplicate_pkts]['LTEArrivalTimes'] 
                     # a block  to be populated by #1: pkt id, #2: scheduled path #3: estimate arrival #4: sent time #5: arrival  time
                     
                     snd_rcvd_block_last_wifi= numpy.c_[numpy.array(range(0,nof_duplicate_pkts)), 3*numpy.ones(nof_duplicate_pkts), numpy.zeros(nof_duplicate_pkts) , first_sentTimes_wifi, first_arrivals_wifi ]
                     snd_rcvd_block_last_lte= numpy.c_[numpy.array(range(0,nof_duplicate_pkts)), 3*numpy.ones(nof_duplicate_pkts), numpy.zeros(nof_duplicate_pkts) , first_sentTimes_wifi, first_arrivals_lte ]
                     # first blocks are the same for all schedulers , because of duplication
                     snd_rcvd_block_last_ma_wifi =snd_rcvd_block_last_wifi
                     snd_rcvd_block_last_ma_lte =snd_rcvd_block_last_lte
                     snd_rcvd_block_last_mmc_wifi =snd_rcvd_block_last_wifi
                     snd_rcvd_block_last_mmc_lte =snd_rcvd_block_last_lte
                     snd_rcvd_block_last_sedpf_wifi =snd_rcvd_block_last_wifi
                     snd_rcvd_block_last_sedpf_lte =snd_rcvd_block_last_lte
                     #scheduling for RR
   		     print "Round Robin scheduler running.. "
                     snd_rcvd_rr=round_robin_scheduler(df)
                     expected_reordering_delay[n_runs-1,0],std_reordering_delay[n_runs-1,0], rr_sequenced  = reordering_stats (snd_rcvd_rr[nof_duplicate_pkts:,])
                     print "Round Robin scheduler finished.Expected reordering delay is:  %f" %expected_reordering_delay[n_runs-1,0]
 
		     #scheduling for EDPF
		     print "EDPF scheduler running.."
		     snd_rcvd_edpf_wifi,snd_rcvd_edpf_lte =edpf_scheduler(df,pkt_length,nof_duplicate_pkts)
		     snd_rcvd_edpf =numpy.r_[snd_rcvd_edpf_wifi[nof_duplicate_pkts:], snd_rcvd_edpf_lte[nof_duplicate_pkts:]]
		     expected_reordering_delay[n_runs-1,1],std_reordering_delay[n_runs-1,1], edpf_sequenced = reordering_stats(snd_rcvd_edpf[nof_duplicate_pkts:,])
		     RMSE[n_runs-1,0]=numpy.sqrt(numpy.mean((snd_rcvd_edpf[nof_duplicate_pkts:,2]-snd_rcvd_edpf[nof_duplicate_pkts:,4])**2))
		     print "EDPF scheduler finished. Expected reordering delay is:  %f" %expected_reordering_delay[n_runs-1,1]
		     pdb.set_trace()

                     
                     ## initialise sedpf 1st and 2nd moment stats from previous duplicate packets
#                     Z_mean=numpy.array ([ numpy.mean(  numpy.asarray(df.iloc[:nof_duplicate_pkts]['WiFiArrivalTimes']) - df.iloc[0]['WiFiArrivalTimes']) , numpy.mean( numpy.asarray(df.iloc[:nof_duplicate_pkts]['LTEArrivalTimes'])- df.iloc[0]['LTEArrivalTimes'] ) ] )
#                     Z_sigma=numpy.array ( [ numpy.std(  numpy.asarray(df.iloc[:nof_duplicate_pkts]['WiFiArrivalTimes']) - df.iloc[0]['WiFiArrivalTimes']) , numpy.std( numpy.asarray(df.iloc[:nof_duplicate_pkts]['LTEArrivalTimes'])- df.iloc[0]['LTEArrivalTimes'] ) ]	)
                     ## initialise sedpf 1st and 2nd moment stats from all measurements
                     different_deltas=(1,10, 50)
		     delta_colors=iter(cm.rainbow(numpy.linspace(0,1,len(different_deltas))))
		     Z_mean=numpy.zeros((len(different_deltas), 2))
		     Z_sigma=numpy.zeros((len(different_deltas), 2))
		     plt.figure(5)
		     for index,initial_window_sedpf in enumerate(different_deltas):
		             increment_wifi=numpy.zeros(df.shape[0]-initial_window_sedpf)
		             increment_lte=numpy.zeros(df.shape[0]-initial_window_sedpf)
		             for w in range(initial_window_sedpf, df.shape[0]):
		                        
		             		increment_wifi[w-initial_window_sedpf]= df.iloc[w]['WiFiArrivalTimes'] -df.iloc[w-initial_window_sedpf]['WiFiArrivalTimes']
		             		increment_lte[w-initial_window_sedpf]= df.iloc[w]['LTEArrivalTimes'] -df.iloc[w-initial_window_sedpf]['LTEArrivalTimes']
			     next_color=next(delta_colors)
			     plt.figure(5)        
                             plt.hist( increment_wifi,10, lw=2, color=next_color, label='delta= %s' %(str(initial_window_sedpf)) )
                             plt.legend(loc='upper right', frameon=False,   fontsize=14)
                             
                             plt.figure(6) 	
                     	     plt.hist( increment_lte,10, lw=2, color=next_color, label='delta= %s' %(str(initial_window_sedpf)) )
                     	     plt.legend(loc='upper right', frameon=False,   fontsize=14)
                     
			     
			     
		             Z_mean[index,:]=  numpy.array ([numpy.mean(increment_wifi) , numpy.mean(increment_lte) ] ) 
		             Z_sigma[index,:] =  numpy.array ( [ numpy.std(increment_wifi) , numpy.std(increment_lte) ] ) 	
		            
		     
		     plt.show()
                     
                     	
                     length_Delta_time=  numpy.maximum ( 3*Z_sigma[1,:] - Z_mean[1,:], 3*Z_sigma[1,:] + Z_mean[1,:] )  #  in time
	             length_Delta_slot= length_Delta_time / numpy.mean (numpy.subtract(df.iloc[1:]['WiFiArrivalTimes'],df.iloc[0:df.shape[0]-1]['WiFiArrivalTimes']))
                     
                     ########################SEDPF######################################
		     print "SEDPF scheduler running.."
                     snd_rcvd_sedpf_wifi, snd_rcvd_sedpf_lte= sedpf2_scheduler(df,snd_rcvd_block_last_wifi,snd_rcvd_block_last_lte, Z_mean[1,:], Z_sigma[1,:], length_Delta_slot) 
                     snd_rcvd_sedpf= numpy.r_ [ snd_rcvd_sedpf_wifi[nof_duplicate_pkts:,:], snd_rcvd_sedpf_lte[nof_duplicate_pkts:,:] ]
		     expected_reordering_delay[n_runs-1,2],std_reordering_delay[n_runs-1,2], sedpf_sequenced = reordering_stats(snd_rcvd_sedpf)
		     RMSE[n_runs-1,1]=numpy.sqrt(numpy.mean((snd_rcvd_sedpf[:,2]-snd_rcvd_sedpf[:,4])**2))
		     print "SEDPF scheduler finished. Expected reordering delay is:  %f" %expected_reordering_delay[n_runs-1,2]
		     pdb.set_trace()
                    
		              

                
                ##################################### calling Moving averages##########################
                start_time_ma=timeit.default_timer() # time the execution of the ma and scheduler's sort
                nof_pmf_calls= nof_pmf_calls+1
                print "calling MA+MMC at slot  %d for %dth time"   %(s_i ,nof_pmf_calls)
		allRcvd= dynamicdf
                predicted_delays_ma= moving_average(allRcvd, s_i, nof_predictions_after_current_slot, False)
                if n_slots < predicted_delays_ma.shape[0]-1:
                    predicted_delays_ma=numpy.delete(predicted_delays_ma,range(n_slots, predicted_delays_ma.shape[0]) ,0)
                
                #mmc instead of moving averages
                col_mean, col_std, predicted_delays_mmc= moving_average(allRcvd, s_i, nof_predictions_after_current_slot, True )
                mu_rcvd_wifi.append(col_mean[0])
                mu_rcvd_lte.append(col_mean[1])
                sigma_rcvd_wifi.append(col_std[0])
                sigma_rcvd_lte.append(col_std[1])
                if n_slots < predicted_delays_mmc.shape[0]-1:
                    predicted_delays_mmc=numpy.delete(predicted_delays_mmc,range(n_slots, predicted_delays_mmc.shape[0]) ,0)
                

                
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

     expected_time_towait[n_runs-1,0]=numpy.mean(rr_sequenced[:,-1]-rr_sequenced[:, -2])
     expected_time_towait[n_runs-1,1]=numpy.mean(edpf_sequenced[:,-1]- edpf_sequenced[:, -2])
     expected_time_towait[n_runs-1,2]=numpy.mean(sedpf_sequenced[:,-1]- sedpf_sequenced[:, -2])


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
     plt.tick_params(labelsize=20)
     plt.show()
     
     plt.figure(2)
     re_sedpf=sedpf_sequenced[:,-1]-sedpf_sequenced[:,-3]
     re_edpf=edpf_sequenced[:,-1]-edpf_sequenced[:,-3]
     re_edpf=edpf_sequenced[:,-1]-edpf_sequenced[:,-3]
     plt.hist(re_sedpf,100, lw=2, color='red', label='SEDPF')
     plt.hist(re_edpf, 100, lw=2, color='cyan', label='EDPF')
     plt.legend(loc='upper right', frameon=False, fontsize=14)
     plt.xlabel('re-ordering delay (s)', fontsize=20)
     plt.ylabel('Frequency', fontsize=20)
     plt.tick_params(labelsize=20)
     plt.show()
     
     
     plt.figure(3)
     plt.plot(range(0,edpf_sequenced.shape[0]), edpf_sequenced[:, -2]-edpf_sequenced[:,-1])
     plt.show()

    
