
library(formatR)
library(tidyr)
library(ggplot2)
library(readxl)
library(data.table)


setwd("/Users/stephenkinane/Library/CloudStorage/OneDrive-UniversityofGeorgia/2024/Research/StochasticFertilization/Code/")
'%Notin%' <- Negate('%in%')

header.true <- function(df) {
  names(df) <- as.character(unlist(df[1,]))
  df[-1,]
}

round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  
  df[,nums] <- round(df[,nums], digits = digits)
  
  (df)
}



#options(scipen=999,digits=6)

file_path <- ''

stand_lister <- read_excel(path = paste(file_path,'model_input_fert.xlsx',sep=""),sheet = 'stand')

regime_lister <- read_excel(path = paste(file_path,'model_input_fert.xlsx',sep=""),sheet = 'regimes')

units_product <- read_excel(path = paste(file_path,'model_input_fert.xlsx',sep=""),sheet = 'specs')

stand_regime <- merge(stand_lister,regime_lister)

stand_regime$UniqueID <- paste(stand_regime$stand,stand_regime$regime,sep="-")

############################################################################
### LOAD FUNCTIONS
############################################################################

hdom_proj_fun <- function(hdom1, age1, age2){
  
  hdom2 <-  hdom1*((1-exp(-0.014452*age2))/(1-exp(-0.014452*age1)))**0.8216
  
  return(hdom2)
}

hdom_pred_fun <- function(si.25, age1){  # equation 14
  si.25*(0.30323/(1-exp(-0.014452*age1)))**-0.8216
}

tpa_fun <- function(init_tpa, si, age1, age2){
  
  tpa2 <- 100+((init_tpa-100)**-0.745339+0.0003425**2*si*(age2**1.97472-age1**1.97472))**(-1/0.745339)
  
  return(tpa2)
}

ba_proj_fun <- function(init_ba,tpa1,tpa2,age1,age2,hdom1,hdom2,region){
  
  ifelse(region=="LCP",
         ba2 <-  exp(log(init_ba)+-42.689283*((1/age2)-(1/age1))+0.367244*
                       (log(tpa2)-log(tpa1))+0.659985*(log(hdom2)-log(hdom1))+
                       2.012724*((log(tpa2)/age2)-(log(tpa1)/age1))+7.703502*
                       ((log(hdom2)/age2)-(log(hdom1)/age1))),
         
         
         ifelse(region=="PUCP",
                ba2 <-  exp(log(init_ba)+-36.050347*((1/age2)-(1/age1))+0.299071*
                              (log(tpa2)-log(tpa1))+0.980246*(log(hdom2)-log(hdom1))+
                              3.309212*((log(tpa2)/age2)-(log(tpa1)/age1))+3.787258*
                              ((log(hdom2)/age2)-(log(hdom1)/age1))),0))
  
  return(ba2)
}


ba_pred_fun <- function(init.tpa,hdom,age1,region){ #equation 16
  ifelse(region=="LCP",
         exp(0.0 +(-42.689283/age1)+0.367244*log(init.tpa)+0.659985*log(hdom)+2.012724*(log(init.tpa)/age1)+7.703502*(log(hdom)/age1)),
         ifelse(region=="PUCP",
                exp(-0.855557 +(-36.050347/age1)+0.299071*log(init.tpa)+0.980246*log(hdom)+3.309212*(log(init.tpa)/age1)+3.787258*(log(hdom)/age1)),
                0))
}



yield_pred_fun <- function(hdom,ba,tpa,age,unit,region){
  ifelse(unit=="TVOB"&region=="PUCP",exp(0.0+0.268552*log(hdom)+1.368844*log(ba)+-7.466863*(log(tpa)/age)+8.934524*(log(hdom)/age)+3.553411*(log(ba)/age)),
         ifelse(unit=="TVIB"&region=="PUCP",exp(0.0+0.350394*log(hdom)+1.263708*log(ba)+-8.608165*(log(tpa)/age)+7.193937*(log(hdom)/age)+6.309586*(log(ba)/age)),
                ifelse(unit=="GWOB"&region=="PUCP",exp(-3.818016+0.430179*log(hdom)+1.276768*log(ba)+-8.088792*(log(tpa)/age)+7.428472*(log(hdom)/age)+5.554509*(log(ba)/age)),
                       ifelse(unit=="DWIB"&region=="PUCP",exp(-4.987560+0.446433*log(hdom)+1.348843*log(ba)+-7.757842*(log(tpa)/age)+7.857337*(log(hdom)/age)+4.222016*(log(ba)/age)),
                              ifelse(unit=="TVOB"&region=="LCP", exp(-1.520877+0.200680*log(tpa)+1.207586*(log(hdom))+0.703405*(log(ba))+-5.139064*(log(tpa)/age)+6.744164*(log(ba)/age)),
                                     ifelse(unit=="TVIB"&region=="LCP",exp(-2.088857+0.177587*log(tpa)+1.303770*(log(hdom))+0.726950*(log(ba))+-5.091474*(log(tpa)/age)+6.676532*(log(ba)/age)),
                                            ifelse(unit=="GWOB"&region=="LCP",exp(-5.175922+0.198424*log(tpa)+1.232028*(log(hdom))+0.705769*(log(ba))+-5.129853*(log(tpa)/age)+6.731477*(log(ba)/age)),
                                                   ifelse(unit=="DWIB"&region=="LCP",exp(-6.332502+0.145815*log(tpa)+1.296629*(log(hdom))+0.814967*(log(ba))+-4.660198*(log(tpa)/age)+5.383589*(log(ba)/age)),0))))))))
}


product_function <- function(yield, top.dia, dbh.lim, qmd, tpa,unit,region){
  ifelse(unit=="TVOB"&region=="PUCP",yield*(exp(-0.982648*(top.dia/qmd)**3.991140+-0.748261*(tpa**-0.111206)*(dbh.lim/qmd)**5.784780)),
         ifelse(unit=="TVIB"&region=="PUCP",yield*(exp(-1.036792*(top.dia/qmd)**3.900677+-0.511939*(tpa**-0.046007)*(dbh.lim/qmd)**5.640610)),
                ifelse(unit=="GWOB"&region=="PUCP",yield*(exp(-1.007482*(top.dia/qmd)**3.931373+-0.518057*(tpa**-0.048385)*(dbh.lim/qmd)**5.660573)),
                       ifelse(unit=="DWIB"&region=="PUCP",yield*(exp(-0.934936*(top.dia/qmd)**4.111618+-0.590269*(tpa**-0.065355)*(dbh.lim/qmd)**5.596179)),
                              ifelse(unit=="TVOB"&region=="LCP",yield*(exp(-1.034486*(top.dia/qmd)**3.940848+-5.062955*(tpa**-0.422892)*(dbh.lim/qmd)**6.004646)),
                                     ifelse(unit=="TVIB"&region=="LCP",yield*(exp(-1.105225*(top.dia/qmd)**3.878664+-4.459271*(tpa**-0.404057)*(dbh.lim/qmd)**5.984225)),
                                            ifelse(unit=="GWOB"&region=="LCP",yield*(exp(-1.064132*(top.dia/qmd)**3.818683+-5.048319*(tpa**-0.422117)*(dbh.lim/qmd)**5.991728)),
                                                   ifelse(unit=="DWIB"&region=="LCP",yield*(exp(-0.963185*(top.dia/qmd)**4054202+-4.540672*(tpa**-0.406561)*(dbh.lim/qmd)**5.962867)),0))))))))
  
}

qmd_fun <- function(tpa,ba){
  sqrt((ba/tpa)/0.005454154)
}


thin_removed_fun <- function(ba_init, tpa_before,tpa_row_remove, tpa_select_remove){ #equation 25
  ba_init*(tpa.row.remove/tpa_before)+(1-(tpa_row_remove/tpa_before))*(tpa_select_remove/(tpa_before-tpa_row_remove))**1.2345
}

competition_index <- function(ba_thinned,ba_unthinned){ # equation 26
  1-(ba_thinned/ba_unthinned)
}

projected_ci <- function(ci_init, age1,age2,region){ # equation 27
  ifelse(region=="PUCP",ci_init*exp(-0.076472*(age2-age1)),
         ifelse(region=="LCP",ci_init*exp(-0.110521*(age2-age1)),
                0))
}

ba_thinned_projected <- function(ba_unthinned,ci_projected){ #equation 28
  ba_unthinned*(1-ci_projected)
}

## solve for tpa removed in selection, solve for TPAs in equation 25
tpa_select_remove <- function(ba_thin,ba_before,tpa_before,tpa_row_remove){
  (((ba_thin*tpa_before-ba_before*tpa_row_remove)/(-ba_before*tpa_row_remove+ba_before*tpa_before))**(2000/2469))*(tpa_before-tpa_row_remove)
}

hd_response_fert <- function(N,P,Yst){
  r_hd <- (0.00106*N+0.2506*P)*Yst*exp(-0.1096*Yst)
  return(r_hd)
}

ba_response_fert <- function(N,P,Yst){
  r_ba <- (0.0121*N+1.3639*P)*Yst*exp(-0.2635*Yst)
  return(r_ba)
}


ba_hdwd_piedmont <- function(hdom, tpa, phwd, age){
  ba <- exp(-0.904066*(-33.811815/age)+0.321301*log(tpa)+0.985342*log(hdom)+3.381071 *(log(tpa)/age)+2.548207*(log(hdom)/age)-0.003689*phwd)
}

############################################################################
### NON THINNED DATA
############################################################################

non_thinned <- subset(stand_regime,stand_regime$thin1_age==0)

non_thinned$MaxAge <- non_thinned$age+50 #every stand will have a max age

n <- length(unique(non_thinned$UniqueID))
# or pre-allocate for slightly more efficiency
datalist = vector("list", length = n)

if (n>0){
  
for (i in 1:n){
  
  stand_of_interest <- unique(non_thinned$UniqueID)[i]
  
  new_data <- subset(non_thinned,non_thinned$UniqueID==stand_of_interest)
  new_periods <- data.frame(seq(new_data$age+1,new_data$MaxAge,1))
  names(new_periods) <- "age"
  
  init_age <- new_data$age
  init_tpa <- new_data$tpa
  init_ba <- new_data$ba
  
  init_hd <- new_data$hd
  init_si <- new_data$si
  init_region <- new_data$region
  
  init_hd <- ifelse(init_hd==0,hdom_pred_fun(si.25 = init_si,age1 = init_age),init_hd)
  init_ba <- ifelse(init_ba==0,ba_pred_fun(init.tpa = init_tpa,hdom = init_hd,age1=init_age,region = init_region),init_ba)
  new_data$ba <- init_ba
  new_data$hd <- init_hd
  
  new_data_small <- new_data[,c('age','tpa','hd','ba')]
  
  for (j in 1:dim(new_periods)[1]){
    age <- new_periods[j,1]
    age_hd <- hdom_proj_fun(hdom1=init_hd,age1=init_age,age2=age)
    age_tpa <- tpa_fun(init_tpa = init_tpa, si = init_si,age1 = init_age,age2 = age)
    age_ba <- ba_proj_fun(init_ba = init_ba,tpa1 = init_tpa,tpa2 = age_tpa,age1=init_age,age2=age,hdom1=init_hd,hdom2=age_hd,region=init_region)
    
    new_data_small <- rbind(new_data_small, c(age, age_tpa, age_hd, age_ba))
  }
  
  new_data_small$stand <- stand_of_interest
  
  yields <- merge(new_data[,c("stand","si","region","regime","UniqueID")],new_data_small,by.x = c("UniqueID"), by.y=c("stand"),all=T)
  
  datalist[[i]] <- yields
}

nonthinned2 = do.call(rbind, datalist)

nonthinned2$thin_remove <- 0
} else{
  
  nonthinned2 <- non_thinned
  
}


############################################################################
### THINNED DATA
############################################################################

thinned <- subset(stand_regime,stand_regime$thin1_age>0)

thinned$MaxAge <- thinned$age+50 #every stand will have a max age


n <- length(unique(thinned$UniqueID))
# or pre-allocate for slightly more efficiency
datalist = vector("list", length = n)

if (n>0){
  
#datalist = list()
for (i in 1:n){
  
  stand_of_interest <- unique(thinned$UniqueID)[i]
  print(stand_of_interest)
  #stand_of_interest <- "stand1-regime2"
  
  new_data <- subset(thinned,thinned$UniqueID==stand_of_interest)
  new_periods <- data.frame(seq(new_data$age+1,new_data$MaxAge,1))
  names(new_periods) <- "age"
  
  ####
  
  thin_regime_info <- subset(stand_regime,stand_regime$UniqueID==stand_of_interest)
  thin_regime_info2 <- thin_regime_info[,c("UniqueID","thin1_age","thin1_residBA","thin2_age","thin2_residBA")]
  
  setDT(thin_regime_info2)
  thin_regime_info3 <- melt(thin_regime_info2, measure.vars = patterns(c('_age', '_residBA')), value.name = c('age', 'residBA'))
  thin_regime_info3$thinUniqueID <- with(thin_regime_info3,paste(UniqueID,variable,sep="-"))
  thin_regime_info4 <- subset(thin_regime_info3,thin_regime_info3$age>0)
  
  ## thin numbers
  
  number_of_thins <- length(unique(thin_regime_info4$thinUniqueID))
  
  thin_id <- (thin_regime_info4$thinUniqueID)
  thin_regime <- subset(thin_regime_info4,thin_regime_info4$thinUniqueID==thin_id)
  regime_id <- unique(thin_regime$UniqueID)
  
  init_age <- new_data$age
  thin_age <- new_data$thin1_age
  thin_ba <- new_data$thin1_residBA
  init_tpa <- new_data$tpa
  init_ba <- new_data$ba
  
  init_hd <- new_data$hd
  init_si <- new_data$si
  init_region <- new_data$region
  
  init_hd <- ifelse(init_hd<=0,hdom_pred_fun(si.25 = init_si,age1 = init_age),init_hd)
  init_ba <- ifelse(init_ba<=0,ba_pred_fun(init.tpa = init_tpa,hdom = init_hd,age1=init_age,region = init_region),init_ba)
  new_data$ba <- init_ba
  new_data$hd <- init_hd
  
  
  new_data_small <- new_data[,c('age','tpa','hd','ba')]
  
  if (init_age > thin_age){
    print(paste(  print(stand_of_interest), 'cannot meet thin conditions, proceeding without thin', sep = ": "))
    
    for (j in 1:dim(new_periods)[1]){
      age <- new_periods[j,1]
      age_hd <- hdom_proj_fun(hdom1=init_hd,age1=init_age,age2=age)
      age_tpa <- tpa_fun(init_tpa = init_tpa, si = init_si,age1 = init_age,age2 = age)
      age_ba <- ba_proj_fun(init_ba = init_ba,tpa1 = init_tpa,tpa2 = age_tpa,age1=init_age,age2=age,hdom1=init_hd,hdom2=age_hd,region=init_region)
      
      new_data_small <- rbind(new_data_small, c(age, age_tpa, age_hd, age_ba))
    }
    
    new_data_small$stand <- stand_of_interest
    new_data_small$thin_remove <- 0
    
    yields <- merge(new_data[,c("stand","si","region","regime","UniqueID")],new_data_small,by.x = c("UniqueID"), by.y=c("stand"),all=T)
    
    datalist[[i]] <- yields
    
  } else if (init_age <= thin_age){
    
    thin_age <- ifelse(thin_age==init_age,thin_age+1,thin_age)
    period_of_thin <- subset(new_periods,new_periods$age==thin_age)
    period_of_thin2 <- as.numeric(rownames(period_of_thin))
    
    for (j in 1:period_of_thin2){
      age <- new_periods[j,1]
      age_hd <- hdom_proj_fun(hdom1=init_hd,age1=init_age,age2=age)
      age_tpa <- tpa_fun(init_tpa = init_tpa, si = init_si,age1 = init_age,age2 = age)
      age_ba <- ba_proj_fun(init_ba = init_ba,tpa1 = init_tpa,tpa2 = age_tpa,age1=init_age,age2=age,hdom1=init_hd,hdom2=age_hd,region=init_region)
      
      new_data_small <- rbind(new_data_small, c(age, age_tpa, age_hd, age_ba))
    }
    
    ## get stand conditions immediately pre thin
    
    prethin_ba <- new_data_small[nrow(new_data_small),'ba' ]
    prethin_tpa <- new_data_small[nrow(new_data_small),'tpa' ]
    prethin_hd <- new_data_small[nrow(new_data_small),'hd' ]
    
    thin_remove_ba <- prethin_ba - thin_ba
    
    if (thin_remove_ba <= 0){
      print(paste(  print(stand_of_interest), 'cannot meet thin conditions, proceeding without thin', sep = ": "))
      
      for (j in 1:dim(new_periods)[1]){
        age <- new_periods[j,1]
        age_hd <- hdom_proj_fun(hdom1=init_hd,age1=init_age,age2=age)
        age_tpa <- tpa_fun(init_tpa = init_tpa, si = init_si,age1 = init_age,age2 = age)
        age_ba <- ba_proj_fun(init_ba = init_ba,tpa1 = init_tpa,tpa2 = age_tpa,age1=init_age,age2=age,hdom1=init_hd,hdom2=age_hd,region=init_region)
        
        
        new_data_small <- rbind(new_data_small, c(age, age_tpa, age_hd, age_ba))
      }
      
      new_data_small$stand <- stand_of_interest
      new_data_small$thin_remove <- 0
      
      yields <- merge(new_data[,c("stand","si","region","regime","UniqueID")],new_data_small,by.x = c("UniqueID"), by.y=c("stand"),all=T)
      
      datalist[[i]] <- yields
      
    } else if (thin_remove_ba > 0){
      
      
      ba_thin_row <- prethin_ba*0.25
      ba_thin_sel <- thin_remove_ba - ba_thin_row
      ba_thin_sel <- ifelse(ba_thin_sel<0,0,ba_thin_sel)
      
      #thin check
      
      ba_check <- thin_remove_ba - ba_thin_row - ba_thin_sel
      
      if (ba_check<0) {
        print(paste(  print(stand_of_interest), 'row thin will remove more than target BA, caution', sep = ": "))
      }
      postthin_ba <- prethin_ba-(ba_thin_row + ba_thin_sel)
      prethin_qmd <- qmd_fun(tpa = prethin_tpa, ba = prethin_ba)
      thin_tpa_row <- prethin_tpa*0.25 # this is assuming a 4th row thin 
      thin_tpa_sel <- ifelse(ba_thin_sel>0,tpa_select_remove(ba_thin = thin_remove_ba,ba_before = prethin_ba,tpa_before = prethin_tpa,tpa_row_remove = thin_tpa_row),0)
      
      postthin_tpa <- prethin_tpa-thin_tpa_row-thin_tpa_sel
      postthin_qmd <- qmd_fun(tpa = postthin_tpa, ba = postthin_ba)
      init_unit <- "GWOB"
      prethin_yield <- yield_pred_fun(hdom=prethin_hd,ba = prethin_ba,tpa = prethin_tpa,age = thin_age,unit = init_unit,region = init_region)
      postthin_yield<- yield_pred_fun(hdom=prethin_hd,ba = postthin_ba,tpa = postthin_tpa,age = thin_age,unit = init_unit,region = init_region)
      yield_thinned <- prethin_yield-postthin_yield
      
      qmd_thinned <- qmd_fun(tpa = (thin_tpa_row+thin_tpa_sel),ba = (thin_remove_ba))
      
      units_product2 <- gather(units_product, condition, measurement, top:length, factor_key=TRUE)
      units_product2$product2 <- with(units_product2, paste(product,condition,sep="."))
      units_product2 <- subset(units_product2,units_product2$species=="pine"&units_product2$product=="pulp")
      units_product3 <- data.frame(t(units_product2[,c("product2","measurement")]))
      units_product3 <- header.true(units_product3)
      units_product3 <- (sapply( units_product3, as.numeric ))
      units_product3 <- data.frame(lapply(units_product3, type.convert), stringsAsFactors=FALSE)
      
      pulp_thinned <- product_function(yield = yield_thinned,top.dia = units_product3$pulp.top,dbh.lim = units_product3$pulp.dbh,qmd =qmd_thinned,tpa = (thin_tpa_row+thin_tpa_sel),unit = init_unit,region = init_region )
      
      new_data_small2 <- new_data_small[1:nrow(new_data_small)-1,]
      new_data_small2$thin_remove <- 0
      postthin_stand <- data.frame(cbind(thin_age, postthin_tpa,prethin_hd,postthin_ba,pulp_thinned))
      names(postthin_stand) <- names(new_data_small2)
      new_data_small3 <- rbind(new_data_small2,postthin_stand)

      unthinned_counterpart <- ba_pred_fun(postthin_tpa, prethin_hd, thin_age, init_region)
      postthin_CI <- competition_index(ba_thinned = postthin_ba,ba_unthinned = unthinned_counterpart)
      
      
      
      if (number_of_thins == 1){
        
        max_periods <- dim(new_periods)[1]
        next_period <- period_of_thin2+1
        
        
        for (k in next_period:max_periods){
          
          age <- new_periods[k,1]
          projected_CI <- projected_ci(postthin_CI, thin_age, age,init_region ) # project the TRUE unthinned counterpart
          age_hd <- hdom_proj_fun(hdom1=prethin_hd,age1=thin_age,age2=age)
        # correct for projections when TPA < 100
         age_tpa <- ifelse(postthin_tpa>100,tpa_fun(init_tpa = postthin_tpa, si = init_si,age1 = thin_age,age2 = age),
                            (postthin_tpa*(0.99)^(age-thin_age)))
         #age_tpa <- tpa_fun(init_tpa = postthin_tpa, si = init_si,age1 = thin_age,age2 = age)
         
         
          age_ba <- ba_proj_fun(init_ba = postthin_ba,tpa1 = postthin_tpa,tpa2 = age_tpa,age1=thin_age,age2=age,hdom1=prethin_hd,hdom2=age_hd,region=init_region)
          
          age_ba_ntc       <- ba_pred_fun(age_tpa,age_hd,age,init_region)
          age_ba_thinned   <- age_ba_ntc * (1-projected_CI)
          
          new_data_small3 <- rbind(new_data_small3, c(age, age_tpa, age_hd, age_ba_thinned,0))
        }
        
        new_data_small3$stand <- stand_of_interest
        
        yields <- merge(new_data[,c("stand","si","region","regime","UniqueID")],new_data_small3,by.x = c("UniqueID"), by.y=c("stand"),all=T)
        
        datalist[[i]] <- yields
        
      } else if (number_of_thins == 2){
        
        next_period <- period_of_thin2+1
        
        thin2_age <- new_data$thin2_age
        thin2_ba <- new_data$thin2_residBA
        
        period_of_thin_2 <- subset(new_periods,new_periods$age==thin2_age)
        period_of_thin_22 <- as.numeric(rownames(period_of_thin_2))
        
        
        for (k in next_period:period_of_thin_22){
          
          age <- new_periods[k,1]
          projected_CI <- projected_ci(postthin_CI, thin_age, age,init_region ) # project the TRUE unthinned counterpart
          age_hd <- hdom_proj_fun(hdom1=prethin_hd,age1=thin_age,age2=age)
        # correct for projections when TPA < 100
          age_tpa <- ifelse(postthin_tpa>100,tpa_fun(init_tpa = postthin_tpa, si = init_si,age1 = thin_age,age2 = age),
                            (postthin_tpa*(0.99)^(age-thin_age)))
          #age_tpa <- tpa_fun(init_tpa = postthin_tpa, si = init_si,age1 = thin_age,age2 = age)


          age_ba <- ba_proj_fun(init_ba = postthin_ba,tpa1 = postthin_tpa,tpa2 = age_tpa,age1=thin_age,age2=age,hdom1=prethin_hd,hdom2=age_hd,region=init_region)
          
          age_ba_ntc       <- ba_pred_fun(age_tpa,age_hd,age,init_region)
          age_ba_thinned   <- age_ba_ntc * (1-projected_CI)
          thin_remove <- 0
          new_data_small3 <- rbind(new_data_small3, c(age, age_tpa, age_hd, age_ba_thinned,thin_remove))
        }
        
        #####
        
        prethin2_ba <- new_data_small3[nrow(new_data_small3),'ba' ]
        prethin2_tpa <- new_data_small3[nrow(new_data_small3),'tpa' ]
        prethin2_hd <- new_data_small3[nrow(new_data_small3),'hd' ]
        
        
        thin_remove_ba <- prethin2_ba- thin2_ba
        thin_tpa_row <- 0
        ba_thin_sel <- thin_remove_ba
        ba_thin_sel <- ifelse(ba_thin_sel<0,0,ba_thin_sel)
        
        #thin check
        
        ba_check <- thin_remove_ba  - ba_thin_sel
        
        if (ba_check<0) {
          print(paste(  print(stand_of_interest), 'row thin will remove more than target BA, caution', sep = ": "))
        }
        postthin_ba <- prethin2_ba - ba_thin_sel
        prethin_qmd <- qmd_fun(tpa = prethin2_tpa, ba = prethin2_ba)
        thin_tpa_sel <- ifelse(ba_thin_sel>0,tpa_select_remove(ba_thin = thin_remove_ba,ba_before = prethin_ba,tpa_before = prethin2_tpa,tpa_row_remove = thin_tpa_row),0)
        
        postthin_tpa <- prethin2_tpa-thin_tpa_row-thin_tpa_sel
        postthin_qmd <- qmd_fun(tpa = postthin_tpa, ba = postthin_ba)
        init_unit <- "GWOB"
        prethin_yield <- yield_pred_fun(hdom=prethin2_hd,ba = prethin2_ba,tpa = prethin2_tpa,age = thin2_age,unit = init_unit,region = init_region)
        postthin_yield<- yield_pred_fun(hdom=prethin2_hd,ba = postthin_ba,tpa = postthin_tpa,age = thin2_age,unit = init_unit,region = init_region)
        yield_thinned <- prethin_yield-postthin_yield
        
        qmd_thinned <- qmd_fun(tpa = (thin_tpa_row+thin_tpa_sel),ba = (thin_remove_ba))
        
        units_product2 <- gather(units_product, condition, measurement, top:length, factor_key=TRUE)
        units_product2$product2 <- with(units_product2, paste(product,condition,sep="."))
        units_product2 <- subset(units_product2,units_product2$species=="pine"&units_product2$product=="pulp")
        units_product3 <- data.frame(t(units_product2[,c("product2","measurement")]))
        units_product3 <- header.true(units_product3)
        units_product3 <- (sapply( units_product3, as.numeric ))
        units_product3 <- data.frame(lapply(units_product3, type.convert), stringsAsFactors=FALSE)
        
        pulp_thinned <- product_function(yield = yield_thinned,top.dia = units_product3$pulp.top,dbh.lim = units_product3$pulp.dbh,qmd =qmd_thinned,tpa = (thin_tpa_row+thin_tpa_sel),unit = init_unit,region = init_region )
        
        #pulp_thinned <- product_function(yield = yield_thinned,top.dia = 3,dbh.lim = 4.5,qmd =qmd_thinned,tpa = (thin_tpa_row+thin_tpa_sel),unit = init_unit,region = init_region )
        
        new_data_small4 <- new_data_small3[1:nrow(new_data_small3)-1,]
        
        postthin_stand <- data.frame(cbind(thin2_age, postthin_tpa,prethin2_hd,postthin_ba,pulp_thinned))
        names(postthin_stand) <- names(new_data_small4)
        new_data_small5 <- rbind(new_data_small4,postthin_stand)
        
        unthinned_counterpart <- ba_pred_fun(postthin_tpa, prethin2_hd, thin2_age, init_region)
        postthin_CI <- competition_index(ba_thinned = postthin_ba,ba_unthinned = unthinned_counterpart)
        
        
        ################
        
        max_periods <- dim(new_periods)[1]
        next_period <- period_of_thin_22+1
        
        
        for (k in next_period:max_periods){
          
          age <- new_periods[k,1]
          projected_CI <- projected_ci(postthin_CI, thin2_age, age,init_region ) # project the TRUE unthinned counterpart
          age_hd <- hdom_proj_fun(hdom1=prethin2_hd,age1=thin2_age,age2=age)
                  # correct for projections when TPA < 100
         age_tpa <- ifelse(postthin_tpa>100,tpa_fun(init_tpa = postthin_tpa, si = init_si,age1 = thin_age,age2 = age),
                            (postthin_tpa*(0.99)^(age-thin2_age)))
         #age_tpa <- tpa_fun(init_tpa = postthin_tpa, si = init_si,age1 = thin_age,age2 = age)
        
        # correct for projections when TPA < 100
          
        age_ba <- ba_proj_fun(init_ba = postthin_ba,tpa1 = postthin_tpa,tpa2 = age_tpa,age1=thin_age,age2=age,hdom1=prethin_hd,hdom2=age_hd,region=init_region)

          
          age_ba_ntc       <- ba_pred_fun(age_tpa,age_hd,age,init_region)
          age_ba_thinned   <- age_ba_ntc * (1-projected_CI)
          thin_remove      <- 0
          new_data_small5 <- rbind(new_data_small5, c(age, age_tpa, age_hd, age_ba_thinned,thin_remove))
        }
        new_data_small5$stand <- stand_of_interest
        
        yields <- merge(new_data[,c("stand","si","region","regime","UniqueID")],new_data_small5,by.x = c("UniqueID"), by.y=c("stand"),all=T)
        
        datalist[[i]] <- yields
        
        
      }
      
      
    }
  }
}
print('here')
thinned2 = do.call(rbind, datalist)

} else{
  
  thinned2 <- thinned
  
}


## compound interest calculator
# A = P(1 + r/n)^(nt)
#a = 85*(0.995)^(2)
############################################################################
### FERTILIZED DATA
############################################################################
all_data <- rbind(nonthinned2,thinned2)
yields_clean2 <- all_data[,c('stand','UniqueID','regime','age','ba','tpa','hd','thin_remove')]


fertilize <- subset(stand_regime,stand_regime$fert1_age>0)

if (length(unique(fertilize$UniqueID))>0){
  

stands_to_fertilize <- unique(fertilize$UniqueID)

stand_data_to_fert <- subset(yields_clean2,yields_clean2$UniqueID%in%stands_to_fertilize)

n_fert <- length(unique(stands_to_fertilize))
# or pre-allocate for slightly more efficiency
datafert = vector("list", length = n_fert)


for (i in 1:n_fert){
  
  stand_of_interest <- stands_to_fertilize[i]
  new_data <- subset(stand_data_to_fert,stand_data_to_fert$UniqueID==stand_of_interest)
  
  fert_regime_info <- subset(stand_regime,stand_regime$UniqueID==stand_of_interest)
  fert_regime_info2 <- fert_regime_info[,c("UniqueID","fert1_age","fert1_N","fert1_P","fert2_age","fert2_N","fert2_P","fert3_age","fert3_N","fert3_P")]
  
  setDT(fert_regime_info2)
  fert_regime_info3 <- melt(fert_regime_info2, measure.vars = patterns(c('_age', '_N', '_P')), value.name = c('age', 'N', 'P'))
  fert_regime_info3$fertUniqueID <- with(fert_regime_info3,paste(UniqueID,variable,sep="-"))
  fert_regime_info4 <- subset(fert_regime_info3,fert_regime_info3$age>0)
  
  
  for (j in 1:length(unique(fert_regime_info4$fertUniqueID))){
    
    fert_id <- unique(fert_regime_info4$fertUniqueID)[j]
    fert_regime <- subset(fert_regime_info4,fert_regime_info4$fertUniqueID==fert_id)
    regime_id <- unique(fert_regime$UniqueID)
    
    fert_age <- fert_regime[,"age"]
    fert_N <- fert_regime[,"N"]
    fert_P <- ifelse(fert_regime[,"P"]>0,1,0)
    
    #fert_data <- subset(new_data,new_data$UniqueID==regime_id)
    new_data$fert_HD <- ifelse(new_data$age>=as.numeric(fert_age),hd_response_fert(N = as.numeric(fert_N),P = as.numeric(fert_P),Yst = new_data$age-as.numeric(fert_age)),0)
    new_data$fert_BA <- ifelse(new_data$age>=as.numeric(fert_age),ba_response_fert(N = as.numeric(fert_N),P = as.numeric(fert_P),Yst = new_data$age-as.numeric(fert_age)),0)
    new_data$hd <- new_data$hd + new_data$fert_HD
    new_data$ba <- new_data$ba + new_data$fert_BA
    
    new_data <-  new_data[,c('stand','UniqueID','regime','age','ba','tpa','hd','thin_remove')]
    
    print(fert_age)
    
  }
  
  
  datafert[[i]] <- new_data
  
}



fert2 = do.call(rbind, datafert)


yields_clean2_no_fert <- subset(yields_clean2,yields_clean2$UniqueID%Notin%stands_to_fertilize)

yield_clean_with_fert <- rbind(yields_clean2_no_fert,fert2)

} else{
  
  yield_clean_with_fert <- yields_clean2
  

}



############################################################################
### MERCHANDIZE DATA
############################################################################

stand_lister2 <- stand_lister[,c('stand','si','region')]

yields_clean3 <- merge(stand_lister2, yield_clean_with_fert,by="stand")
yields_clean4 <- subset(yields_clean3,is.na(yields_clean3$age)==F)
#yields <- merge(yields, units_product3)

yields_clean4$unit <- "GWOB"
yields_clean4$volume <- with(yields_clean4,yield_pred_fun(hdom = hd,ba = ba,tpa = tpa,age=age,unit=unit,region=region))
yields_clean4$qmd <- with(yields_clean4,qmd_fun(tpa=tpa,ba=ba))

### carbon products

#yields_clean4$unit2 <- "DWIB"
#yields_clean4$volumeDWIB <- with(yields_clean4,yield_pred_fun(hdom = hd,ba = ba,tpa = tpa,age=age,unit=unit,region=region))

units_product2 <- gather(units_product, condition, measurement, top:length, factor_key=TRUE)
units_product2$product2 <- with(units_product2, paste(product,condition,sep="."))
units_product3 <- data.frame(t(units_product2[,c("product2","measurement")]))
units_product3 <- header.true(units_product3)

units_product3 <- (sapply( units_product3, as.numeric ))

units_product3 <- data.frame(lapply(units_product3, type.convert), stringsAsFactors=FALSE)

yields_clean5 <- merge(yields_clean4, units_product3)

yields_clean5$saw   <-with(yields_clean5,product_function(yield=volume,top.dia=saw.top, dbh.lim=saw.dbh,qmd=qmd,tpa=tpa ,unit=unit,region=region))
yields_clean5$chip  <-with(yields_clean5,product_function(yield=volume,top.dia=cns.top, dbh.lim=cns.dbh,qmd=qmd,tpa=tpa ,unit=unit,region=region))-yields_clean5$saw 
yields_clean5$pulp  <-with(yields_clean5,product_function(yield=volume,top.dia=pulp.top, dbh.lim=pulp.dbh,qmd=qmd,tpa=tpa ,unit=unit,region=region))- yields_clean5$chip-yields_clean5$saw 

yields_clean2 <- yields_clean5[,c("stand","si","region","UniqueID","regime","age","ba","tpa","hd","qmd","thin_remove","volume","saw","chip","pulp")]



yields_clean2<- yields_clean2[with(yields_clean2,order(UniqueID,age)),]
yields_clean2 <- round_df(yields_clean2, digits=4)

############################################################################
### WRITE DATA TO CSV
############################################################################


write.csv(yields_clean2,file = paste(file_path,"yields_generated_unthinned_20241115.csv",sep=""),row.names = F)

