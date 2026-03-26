# Stand Table Projection

#### Stephen Kinane

#### 2025-11-18

## Introduction

There are a handful of books and publications that can do a much
better job covering the theory of stand table projection (STP). In
practice, this approach is widely adopted to leverage the power of
inventory data (providing the stand table) and well-documented
stand-level models. In this document, we will explore the implementation
of this approach through the use of the PMRC 1996 models. I will do my
best to explain the intricacies of STP, but please be aware that I am
clearly not an expert.

### Simple data creation

In this step we will create a simple stand table:

```
dbh <- seq(6,13,1)
tpa <- c(40,50,65,45,35,20,15,8)
test_df <- data.frame(dbh,tpa)

head(test_df)
```

```
##   dbh tpa
## 1   6  40
## 2   7  50
## 3   8  65
## 4   9  45
## 5  10  35
## 6  11  20
```

To build out the rest of the data, we will add a current age, an age
to project to, current dominant height, and current basal area. Basal
area can be calculated directly from the provided stand table.

```
test_df$ba1 <- (0.005454*test_df$dbh**2)*test_df$tpa
test_df$age <- 22
test_df$age2 <- 35
test_df$hd1 <- 62.62
test_df$region <- "PUCP"

current_ba <- sum(test_df$ba1)
current_tpa <- sum(test_df$tpa)
current_age <- test_df$age[1]
projection_age <- test_df$age2[1]
current_hd <- test_df$hd1[1]
current_region <- test_df$region[1]
message("current ba:: ", current_ba)
```

```
## current ba:: 115.226658
```

```
message("current tpa:: ", current_tpa)
```

```
## current tpa:: 278
```

Now we will start projecting our stand-level variable (HD, BA, TPA)
to the projection age. First we need to load our 1996 PMRC
functions:

```
hdom_proj_fun <- function(hdom1, age1, age2){
  
  hdom2 <-  hdom1*((1-exp(-0.014452*age2))/(1-exp(-0.014452*age1)))**0.8216
  
  return(hdom2)
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
```

Now we can project stand-level variables:

```
test_df$hd2 <- hdom_proj_fun(hdom1=current_hd,age1=current_age,age2=projection_age)
    
test_df$si <- hdom_proj_fun(hdom1=current_hd,age1=current_age,age2=25)
test_df$tpa2 <- tpa_fun(init_tpa = current_tpa,si=test_df$si,age1=current_age,age2=projection_age)
test_df$ba2 <- ba_proj_fun(init_ba=current_ba,tpa1=current_tpa,tpa2=test_df$tpa2,age1=current_age,age2=projection_age,hdom1=current_hd,hdom2=test_df$hd2,region=current_region)
```

### Start Stand Table Projection

First we need to look at the individual diameter classes. We need to
calculate the basal area of the diameter class, then invert it and
sum.

```
#step 1: after T2 BA and TPA have been projected
#determine number of surviving trees per dbh class
#determine mortality probability in each of dbh classes at initial age

#calculate inverse of basal area of each midpoint diameter
test_df$b1i <- (0.005454154*test_df$dbh**2)
test_df$inv_b1i <- 1/(test_df$b1i)

test_df$sum_inv_b1i <- sum(test_df$inv_b1i)
```

With those calculated, we need to estimate the mortality per diameter
class. This is using the ratio of the inverse of the diameter class
basal area to the total sum. Here, our assumption is that the
conditional probability for a diameter class is inversely proportional
to the relative size of the diameter class (in terms of basal area).
Paraphrasing from the PMRC’s 2004 report, prior research makes the
assumption that smaller diameter classes experience higher mortality as
compared to larger diameter classes. The conditional probabilty that we
calculate is showing the probability that a dead tree belongs to a
particular diameter class (given that a tree has died).

```
dbh_class_mortality <- function(inv_ba_dclass,inv_sum_ba_dclass){
  pi = inv_ba_dclass/inv_sum_ba_dclass
}

# calculate pi from equation 4-5
test_df$pi <- dbh_class_mortality(inv_ba_dclass = test_df$inv_b1i ,
                                  inv_sum_ba_dclass = test_df$sum_inv_b1i)
print(test_df)
```

```
##   dbh tpa       ba1 age age2   hd1 region      hd2       si    tpa2      ba2
## 1   6  40  7.853760  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 2   7  50 13.362300  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 3   8  65 22.688640  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 4   9  45 19.879830  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 5  10  35 19.089000  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 6  11  20 13.198680  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 7  12  15 11.780640  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 8  13   8  7.373808  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
##         b1i  inv_b1i sum_inv_b1i         pi
## 1 0.1963495 5.092958     19.6699 0.25892135
## 2 0.2672535 3.741765     19.6699 0.19022793
## 3 0.3490659 2.864789     19.6699 0.14564326
## 4 0.4417865 2.263537     19.6699 0.11507615
## 5 0.5454154 1.833465     19.6699 0.09321168
## 6 0.6599526 1.515260     19.6699 0.07703445
## 7 0.7853982 1.273240     19.6699 0.06473034
## 8 0.9217520 1.084890     19.6699 0.05515484
```

Now we can get the estimated mortality that will occur for the trees
within each diameter class:

```
## equation 4-6 pmrc TR 2004-4
## conditional probability of mortality in each dbh class 

dbh_p_mortality <- function(n1i_pi,sum_n1i_pi){
  pi_i <- n1i_pi/sum_n1i_pi
}

## multiply by mortality
test_df$n1i_pi <- test_df$tpa*test_df$pi
test_df$sum_n1i_pi <- sum(test_df$n1i_pi)

test_df$pi_i <- dbh_p_mortality(n1i_pi = test_df$n1i_pi,sum_n1i_pi = test_df$sum_n1i_pi)

## calculate surviving trees per acre
##. subtract the result from n1i to obtain the
##. number of surviving trees/acre in each dbh class
```

Now we can allocate that mortality based upon our projected TPA (we
want to be the same):

```
test_df$mortality <- sum(test_df$tpa)-test_df$tpa2
test_df$n2i <- test_df$tpa-(test_df$mortality*test_df$pi_i)
test_df$sum_n2i <- sum(test_df$n2i)
print(test_df)
```

```
##   dbh tpa       ba1 age age2   hd1 region      hd2       si    tpa2      ba2
## 1   6  40  7.853760  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 2   7  50 13.362300  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 3   8  65 22.688640  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 4   9  45 19.879830  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 5  10  35 19.089000  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 6  11  20 13.198680  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 7  12  15 11.780640  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 8  13   8  7.373808  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
##         b1i  inv_b1i sum_inv_b1i         pi     n1i_pi sum_n1i_pi       pi_i
## 1 0.1963495 5.092958     19.6699 0.25892135 10.3568539   40.72878 0.25428834
## 2 0.2672535 3.741765     19.6699 0.19022793  9.5113964   40.72878 0.23353010
## 3 0.3490659 2.864789     19.6699 0.14564326  9.4668117   40.72878 0.23243543
## 4 0.4417865 2.263537     19.6699 0.11507615  5.1784269   40.72878 0.12714417
## 5 0.5454154 1.833465     19.6699 0.09321168  3.2624090   40.72878 0.08010083
## 6 0.6599526 1.515260     19.6699 0.07703445  1.5406890   40.72878 0.03782802
## 7 0.7853982 1.273240     19.6699 0.06473034  0.9709550   40.72878 0.02383953
## 8 0.9217520 1.084890     19.6699 0.05515484  0.4412387   40.72878 0.01083359
##   mortality       n2i sum_n2i
## 1  46.96905 28.056320 231.031
## 2  46.96905 39.031314 231.031
## 3  46.96905 54.082730 231.031
## 4  46.96905 39.028160 231.031
## 5  46.96905 31.237741 231.031
## 6  46.96905 18.223254 231.031
## 7  46.96905 13.880280 231.031
## 8  46.96905  7.491157 231.031
```

As we can see, the sum of our diameter class trees are now equivalent
to the projected tpa (225.7)!!!

### Midpoint diameter projection

This is where it starts to get tricky. We will start by calculating
the basal area for the average surviver tree at time 1. We will then sum
that basal area across all the diameter classes, and dividing the result
by the total number of surviving trees. Finally, we will calculate the
relative size of each diameter class:

```
## now we project the dbh class midpoints
test_df$BA_class_1 <- (0.005454*test_df$dbh**2)*test_df$n2i
test_df$sum_BA_class_1 <- sum(test_df$BA_class_1)
test_df$sum_BA_class_1_ratio <- test_df$sum_BA_class_1/test_df$sum_n2i
test_df$relative_size <- (0.005454*test_df$dbh**2)/test_df$sum_BA_class_1_ratio
```

### 

Now we will obtain the diameter class midpoint. We need to remember
that these functions are constrained such that the projected BA and TPA
from the whole stand models will be equivalent to our stand table total
BA and TPA.

To start this, we have to use the number of survivor trees and the
projected BA. We then use the midpoint projection equation for the
diameter class to estimate what the BA will be for that particular
class, and then we convert that basal area to an estimate of
diameter:

```
## equation 23 pmrc TR 1996-1
dbh_class_midpoint <- function(n2i,relative_size,age1,age2,region){
   ifelse(region%in%c("PUCP"), 
          midpointBA2 <- n2i*(relative_size)**(age2/age1)**(-0.2277),
          
    ifelse(region%in%c("LCP"),
           midpointBA2 <- n2i*(relative_size)**(age2/age1)**(-0.0525),0))
  return(midpointBA2)
}

# now we use equation 4-4

d_class_converter <- function(ba){
  d_i <- sqrt(ba/0.005454154)
  return(d_i)
}

test_df$ba_midpoint_2 <- dbh_class_midpoint(n2i = test_df$n2i,
                                            relative_size = test_df$relative_size,
                                            age1 = test_df$age,
                                            age2 = test_df$age2,
                                            region=test_df$region)
#obtain the projected basal area (BA/Class-2) for each diameter class using equation 4-4
## 
test_df$sum_ba_midpoint_2 <- sum(test_df$ba_midpoint_2)

test_df$BA_class_2 <- (test_df$ba_midpoint_2 *test_df$ba2)/test_df$sum_ba_midpoint_2
test_df$sum_BA_class_2 <- sum(test_df$BA_class_2)
test_df$b2i <- test_df$BA_class_2 /test_df$n2i
test_df$d2i <- d_class_converter(test_df$b2i)
print(test_df)
```

```
##   dbh tpa       ba1 age age2   hd1 region      hd2       si    tpa2      ba2
## 1   6  40  7.853760  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 2   7  50 13.362300  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 3   8  65 22.688640  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 4   9  45 19.879830  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 5  10  35 19.089000  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 6  11  20 13.198680  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 7  12  15 11.780640  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
## 8  13   8  7.373808  22   35 62.62   PUCP 85.34153 68.39558 231.031 154.5803
##         b1i  inv_b1i sum_inv_b1i         pi     n1i_pi sum_n1i_pi       pi_i
## 1 0.1963495 5.092958     19.6699 0.25892135 10.3568539   40.72878 0.25428834
## 2 0.2672535 3.741765     19.6699 0.19022793  9.5113964   40.72878 0.23353010
## 3 0.3490659 2.864789     19.6699 0.14564326  9.4668117   40.72878 0.23243543
## 4 0.4417865 2.263537     19.6699 0.11507615  5.1784269   40.72878 0.12714417
## 5 0.5454154 1.833465     19.6699 0.09321168  3.2624090   40.72878 0.08010083
## 6 0.6599526 1.515260     19.6699 0.07703445  1.5406890   40.72878 0.03782802
## 7 0.7853982 1.273240     19.6699 0.06473034  0.9709550   40.72878 0.02383953
## 8 0.9217520 1.084890     19.6699 0.05515484  0.4412387   40.72878 0.01083359
##   mortality       n2i sum_n2i BA_class_1 sum_BA_class_1 sum_BA_class_1_ratio
## 1  46.96905 28.056320 231.031   5.508690       98.92842            0.4282042
## 2  46.96905 39.031314 231.031  10.430962       98.92842            0.4282042
## 3  46.96905 54.082730 231.031  18.877901       98.92842            0.4282042
## 4  46.96905 39.028160 231.031  17.241626       98.92842            0.4282042
## 5  46.96905 31.237741 231.031  17.037064       98.92842            0.4282042
## 6  46.96905 18.223254 231.031  12.026145       98.92842            0.4282042
## 7  46.96905 13.880280 231.031  10.901239       98.92842            0.4282042
## 8  46.96905  7.491157 231.031   6.904794       98.92842            0.4282042
##   relative_size ba_midpoint_2 sum_ba_midpoint_2 BA_class_2 sum_BA_class_2
## 1     0.4585289      13.91141          229.2015   9.382266       154.5803
## 2     0.6241088      25.53960          229.2015  17.224667       154.5803
## 3     0.8151625      44.99946          229.2015  30.348973       154.5803
## 4     1.0316901      40.13913          229.2015  27.071025       154.5803
## 5     1.2736914      38.83320          229.2015  26.190267       154.5803
## 6     1.5411666      26.89239          229.2015  18.137028       154.5803
## 7     1.8341157      23.95502          229.2015  16.155976       154.5803
## 8     2.1525385      14.93125          229.2015  10.070082       154.5803
##         b2i       d2i
## 1 0.3344083  7.830235
## 2 0.4413038  8.995082
## 3 0.5611583 10.143294
## 4 0.6936280 11.277157
## 5 0.8384174 12.398423
## 6 0.9952683 13.508477
## 7 1.1639518 14.608438
## 8 1.3442627 15.699231
```

## Allocation of TPA in new diameter classes

This is another tricky part of the process. Now that we have
projected the midpoints of the diameter classes, we have to reallocate
the trees within the projected diameter classes. This is because we
assume uniform distribution of trees within in the diameter class, and
our new projected diameter classes do not fit within our traditional
one-inch classes “cleanly”. For example, if we project the 8-in diameter
class to 10.2-in, the +/- 0.5 bounds will be from 9.7 - 10.7, spanning
the 10 and 11-inch diameter classes. To reallocate, we just need to take
the proportion of trees that fall within the new classes and
recalculate.

```
# --- 1. Define the Core Transfer Function ---

#' Calculates the proportion of a projected DBH class that falls into a traditional DBH class.
#'
calculate_transfer_proportion <- function(trad_lower_limit, trad_upper_limit,
                                          proj_lower_limit, proj_upper_limit) {
  
  # 1. Determine the overlap interval
  overlap_start <- max(trad_lower_limit, proj_lower_limit)
  overlap_end <- min(trad_upper_limit, proj_upper_limit)
  
  # 2. Check if there is any overlap
  if (overlap_start >= overlap_end) {
    return(0) # No overlap
  }
  
  # 3. Calculate the length of the overlap
  overlap_length <- overlap_end - overlap_start
  
  # 4. Calculate the length of the projected class
  proj_class_length <- proj_upper_limit - proj_lower_limit
  
  # 5. Calculate the proportion. Use max(0, min(1, ...)) for robustness
  if (proj_class_length <= 0) {
    return(0)
  }
  
  proportion <- overlap_length / proj_class_length
  return(max(0, min(1, proportion)))
}

# Filter out rows that might represent dead trees or small growth
# Assuming 'n2i' (projected number of trees) is the count to be redistributed
data <- test_df
data <- data[data$n2i > 0, ]

# Calculate Projected Class Limits (proj_lower and proj_upper)
# The boundary between two projected midpoints is set halfway between them.

# Calculate the difference between successive d2i values
diff_d2i <- diff(data$d2i)

# The half-distance to the next midpoint
# For the last class, we assume the half-distance is the same as the last calculated half-distance
half_diff_d2i_next <- c(diff_d2i / 2, diff_d2i[length(diff_d2i)] / 2)

# The half-distance from the previous midpoint
# For the first class, we assume the half-distance is the same as the first calculated half-distance
half_diff_d2i_prev <- c(diff_d2i[1] / 2, diff_d2i / 2)

# Calculate Projected Limits
data$proj_upper <- data$d2i + 0.5#half_diff_d2i_next
data$proj_lower <- data$d2i - 0.5#half_diff_d2i_prev

# Final structure for projected classes
projected_classes <- data.frame(
  dbh_orig = data$dbh,
  trees_per_acre = data$n2i,
  proj_lower = data$proj_lower,
  proj_upper = data$proj_upper
)

# --- 3. Define Traditional (Target) Classes ---

# Determine the full range of DBH classes needed for the output
# We must cover all projected limits.
min_dbh_proj <- floor(min(projected_classes$proj_lower))
max_dbh_proj <- ceiling(max(projected_classes$proj_upper))

# Create traditional 1-inch DBH classes (e.g., 7" class goes from 6.5 to 7.5)
dbh_midpoints <- seq(from = max(5, min_dbh_proj), to = max_dbh_proj + 1) # Start at least at 5"
traditional_classes <- data.frame(
  dbh_trad = dbh_midpoints,
  trad_lower = dbh_midpoints - 0.5,
  trad_upper = dbh_midpoints + 0.5
)

# --- 4. Apply the Methodology (Class Transfer) ---

# Initialize a data frame to hold the final distribution
final_dbh_distribution <- data.frame(
  DBH_Traditional = traditional_classes$dbh_trad,
  Trees_per_Acre = 0.0
)

# Loop through each traditional (target) DBH class
for (i in 1:nrow(traditional_classes)) {
  trad_class <- traditional_classes[i, ]
  total_trees_in_trad_class <- 0
  
  # Loop through each projected (source) DBH class
  for (j in 1:nrow(projected_classes)) {
    proj_class <- projected_classes[j, ]
    
    # Calculate the transfer proportion
    proportion <- calculate_transfer_proportion(
      trad_class$trad_lower, trad_class$trad_upper,
      proj_class$proj_lower, proj_class$proj_upper
    )
    
    # Calculate the number of trees transferred
    trees_transferred <- proportion * proj_class$trees_per_acre
    
    # Accumulate the trees for the current traditional class
    total_trees_in_trad_class <- total_trees_in_trad_class + trees_transferred
  }
  
  # Store the final result for this traditional class
  final_dbh_distribution[i, "Trees_per_Acre"] <- total_trees_in_trad_class
}
    names(final_dbh_distribution) <- c("new_dclass","tpa2_dist")

print(final_dbh_distribution)
```

```
##    new_dclass tpa2_dist
## 1           7  4.762969
## 2           8 23.485303
## 3           9 38.839361
## 4          10 46.333017
## 5          11 35.960959
## 6          12 29.608815
## 7          13 21.402985
## 8          14 14.701094
## 9          15 10.698406
## 10         16  5.238046
## 11         17  0.000000
## 12         18  0.000000
```

### Final Projected Stand Table

Now we can bring together our newly reallocated TPA estimates and
clean up the table to have the projected information:

```
qmd_fun <- function(tpa,ba){
  sqrt((ba/tpa)/0.005454154)
}

## equation 24 1996-1
height_diameter <- function(HD,DBH_i,qmd,region){
  H_i = ifelse(region=="PUCP",HD*1.179240*(1-0.878092*exp(-1.618723*(DBH_i/qmd))),
               ifelse(region=="LCP",HD*1.185552*(1-0.949316*exp(-1.710774*(DBH_i/qmd))),0))
  return(H_i)
}

stand_table_summary <- merge(test_df,final_dbh_distribution,by.x="dbh",by.y="new_dclass",all=T)
    stand_table_summary$ba2_dist <- (0.005454*stand_table_summary$dbh**2)*stand_table_summary$tpa2_dist
    stand_table_summary$HD <- stand_table_summary$hd2[1]
    stand_table_summary$qmd <- qmd_fun(tpa=sum(stand_table_summary$tpa2_dist,na.rm=T),ba=sum(stand_table_summary$ba2_dist,na.rm=T))
    stand_table_summary$region <- stand_table_summary$region[1]
      stand_table_summary$age2 <- stand_table_summary$age2[1]

    stand_table_summary$dHeight <- height_diameter(HD=stand_table_summary$HD,
                                                    DBH_i = stand_table_summary$dbh,
                                                    qmd = stand_table_summary$qmd ,
                                                    region=stand_table_summary$region)
    
    stand_table_summary$dHeight <- ifelse(stand_table_summary$tpa2_dist==0,0,
                                           stand_table_summary$dHeight)
    stand_table_summary <- subset(stand_table_summary,stand_table_summary$tpa2_dist>0)
    stand_table_summary <- subset(stand_table_summary,is.na(stand_table_summary$tpa2_dist)==F)
    stand_table_summary2 <- stand_table_summary[,c("dbh","ba2_dist","tpa2_dist","age2","dHeight")]
    #names(stand_table_summary4) <-c("source","dclass","ba2_dist","TPA2","age","dHeight")
    
    stand_table_summary2$source <- "projected"
    names(stand_table_summary2) <- c("dbh","ba","tpa","age","dHt","source")
    
    print(stand_table_summary2)
```

```
##    dbh        ba       tpa age      dHt    source
## 2    7  1.272885  4.762969  35 68.84933 projected
## 3    8  8.197686 23.485303  35 73.16922 projected
## 4    9 17.158220 38.839361  35 76.90206 projected
## 5   10 25.270028 46.333017  35 80.12764 projected
## 6   11 23.731859 35.960959  35 82.91488 projected
## 7   12 23.254053 29.608815  35 85.32335 projected
## 8   13 19.727688 21.402985  35 87.40453 projected
## 9   14 15.715234 14.701094  35 89.20289 projected
## 10  15 13.128548 10.698406  35 90.75686 projected
## 11  16  7.313486  5.238046  35 92.09966 projected
```

Let’s compare our stand-level estimates to the standtable
components:

```
projected_tpa <- test_df$tpa2[1]
projected_ba <- test_df$ba2[1]

standtable_tpa <- sum(stand_table_summary2$tpa)
standtable_ba  <- sum(stand_table_summary2$ba)

message("projected ba:: ", projected_ba, "standtable ba::  ", standtable_ba)
```

```
## projected ba:: 154.580282826959standtable ba::  154.769685109904
```

```
message("projecte tpa:: ", projected_tpa, "standtable tpa:: ", standtable_tpa)
```

```
## projecte tpa:: 231.030954211377standtable tpa:: 231.030954211377
```

### Comparison

Now let’s compare our original vs. projected standtables.

```
test_df_compare <- test_df[,c("dbh","ba2","tpa","age","hd1")]
names(test_df_compare) <- c("dbh","ba","tpa","age","dHt")
test_df_compare$source <- "original"

combined_df <- rbind(test_df_compare,stand_table_summary2)
```

```
library(ggplot2)
```

```
## Warning: package 'ggplot2' was built under R version 4.3.3
```

```
ggplot(combined_df, aes(fill=source, y=tpa, x=dbh)) + 
    geom_bar(position = position_dodge(preserve = "single"), stat="identity",width=.5)
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABUAAAAPACAYAAAD0ZtPZAAAEDmlDQ1BrQ0dDb2xvclNwYWNlR2VuZXJpY1JHQgAAOI2NVV1oHFUUPpu5syskzoPUpqaSDv41lLRsUtGE2uj+ZbNt3CyTbLRBkMns3Z1pJjPj/KRpKT4UQRDBqOCT4P9bwSchaqvtiy2itFCiBIMo+ND6R6HSFwnruTOzu5O4a73L3PnmnO9+595z7t4LkLgsW5beJQIsGq4t5dPis8fmxMQ6dMF90A190C0rjpUqlSYBG+PCv9rt7yDG3tf2t/f/Z+uuUEcBiN2F2Kw4yiLiZQD+FcWyXYAEQfvICddi+AnEO2ycIOISw7UAVxieD/Cyz5mRMohfRSwoqoz+xNuIB+cj9loEB3Pw2448NaitKSLLRck2q5pOI9O9g/t/tkXda8Tbg0+PszB9FN8DuPaXKnKW4YcQn1Xk3HSIry5ps8UQ/2W5aQnxIwBdu7yFcgrxPsRjVXu8HOh0qao30cArp9SZZxDfg3h1wTzKxu5E/LUxX5wKdX5SnAzmDx4A4OIqLbB69yMesE1pKojLjVdoNsfyiPi45hZmAn3uLWdpOtfQOaVmikEs7ovj8hFWpz7EV6mel0L9Xy23FMYlPYZenAx0yDB1/PX6dledmQjikjkXCxqMJS9WtfFCyH9XtSekEF+2dH+P4tzITduTygGfv58a5VCTH5PtXD7EFZiNyUDBhHnsFTBgE0SQIA9pfFtgo6cKGuhooeilaKH41eDs38Ip+f4At1Rq/sjr6NEwQqb/I/DQqsLvaFUjvAx+eWirddAJZnAj1DFJL0mSg/gcIpPkMBkhoyCSJ8lTZIxk0TpKDjXHliJzZPO50dR5ASNSnzeLvIvod0HG/mdkmOC0z8VKnzcQ2M/Yz2vKldduXjp9bleLu0ZWn7vWc+l0JGcaai10yNrUnXLP/8Jf59ewX+c3Wgz+B34Df+vbVrc16zTMVgp9um9bxEfzPU5kPqUtVWxhs6OiWTVW+gIfywB9uXi7CGcGW/zk98k/kmvJ95IfJn/j3uQ+4c5zn3Kfcd+AyF3gLnJfcl9xH3OfR2rUee80a+6vo7EK5mmXUdyfQlrYLTwoZIU9wsPCZEtP6BWGhAlhL3p2N6sTjRdduwbHsG9kq32sgBepc+xurLPW4T9URpYGJ3ym4+8zA05u44QjST8ZIoVtu3qE7fWmdn5LPdqvgcZz8Ww8BWJ8X3w0PhQ/wnCDGd+LvlHs8dRy6bLLDuKMaZ20tZrqisPJ5ONiCq8yKhYM5cCgKOu66Lsc0aYOtZdo5QCwezI4wm9J/v0X23mlZXOfBjj8Jzv3WrY5D+CsA9D7aMs2gGfjve8ArD6mePZSeCfEYt8CONWDw8FXTxrPqx/r9Vt4biXeANh8vV7/+/16ffMD1N8AuKD/A/8leAvFY9bLAAAAOGVYSWZNTQAqAAAACAABh2kABAAAAAEAAAAaAAAAAAACoAIABAAAAAEAAAVAoAMABAAAAAEAAAPAAAAAALYRw1EAAEAASURBVHgB7N0HnB5VvT/gkw4hkAoEEAiBEJEYwgUSpKpYKIKAhSaCV7BgQdE/4kVF6UhREUTKRaWJAlKkeukqCIJcggQhAgGEQAgkpJAe/vc3Oq+7m1lINvNm33f2mc9ns/NOOXPOc3Y37373zJxub/zfkiwECBAgQIAAAQIECBAgQIAAAQIECBCooED3CrZJkwgQIECAAAECBAgQIECAAAECBAgQIJAJCEB9IRAgQIAAAQIECBAgQIAAAQIECBAgUFkBAWhlu1bDCBAgQIAAAQIECBAgQIAAAQIECBAQgPoaIECAAAECBAgQIECAAAECBAgQIECgsgIC0Mp2rYYRIECAAAECBAgQIECAAAECBAgQICAA9TVAgAABAgQIECBAgAABAgQIECBAgEBlBQSgle1aDSNAgAABAgQIECBAgAABAgQIECBAQADqa4AAAQIECBAgQIAAAQIECBAgQIAAgcoKCEAr27UaRoAAAQIECBAgQIAAAQIECBAgQICAANTXAAECBAgQIECAAAECBAgQIECAAAEClRUQgFa2azWMAAECBAgQIECAAAECBAgQIECAAAEBqK8BAgQIECBAgAABAgQIECBAgAABAgQqKyAArWzXahgBAgQIECBAgAABAgQIECBAgAABAgJQXwMECBAgQIAAAQIECBAgQIAAAQIECFRWoGdlW9YgDZs2bVqaO3dug9Sm2tXo06dPGjRoUJoyZUpatGhRtRtb0dattNJKaeDAgVnrZs2alWbOnFnRlla/WWuuuWaaPXt2in60NKfA0KFDU7du3dLChQvTyy+/3JyNUOu02mqrpfj/UR827xdD//79U9++fbMGTJ06NS1YsKB5G9OFa96rV680ZMiQ7Hsxfq5amk+gd+/eafDgwVnF4z3OjBkzmq8RapwJrL766tnv6PX6XSP/vRQ3AQKNJWAEaGP1h9oQIECAAAECBAgQIECAAAECBAgQIFCigAC0RExFESBAgAABAgQIECBAgAABAgQIECDQWAIC0MbqD7UhQIAAAQIECBAgQIAAAQIECBAgQKBEAQFoiZiKIkCAAAECBAgQIECAAAECBAgQIECgsQQEoI3VH2pDgAABAgQIECBAgAABAgQIECBAgECJAgLQEjEVRYAAAQIECBAgQIAAAQIECBAgQIBAYwkIQBurP9SGAAECBAgQIECAAAECBAgQIECAAIESBQSgJWIqigABAgQIECBAgAABAgQIECBAgACBxhIQgDZWf6gNAQIECBAgQIAAAQIECBAgQIAAAQIlCghAS8RUFAECBAgQIECAAAECBAgQIECAAAECjSUgAG2s/lAbAgQIECBAgAABAgQIECBAgAABAgRKFBCAloipKAIECBAgQIAAAQIECBAgQIAAAQIEGktAANpY/aE2BAgQIECAAAECBAgQIECAAAECBAiUKCAALRFTUQQIECBAgAABAgQIECBAgAABAgQINJaAALSx+kNtCBAgQIAAAQIECBAgQIAAAQIECBAoUUAAWiKmoggQIECAAAECBAgQIECAAAECBAgQaCwBAWhj9YfaECBAgAABAgQIECBAgAABAgQIECBQooAAtERMRREgQIAAAQIECBAgQIAAAQIECBAg0FgCAtDG6g+1IUCAAAECBAgQIECAAAECBAgQIECgRAEBaImYiiJAgAABAgQIECBAgAABAgQIECBAoLEEBKCN1R9qQ4AAAQIECBAgQIAAAQIECBAgQIBAiQIC0BIxFUWAAAECBAgQIECAAAECBAgQIECAQGMJCEAbqz/UhgABAgQIECBAgAABAgQIECBAgACBEgUEoCViKooAAQIECBAgQIAAAQIECBAgQIAAgcYSEIA2Vn+oDQECBAgQIECAAAECBAgQIECAAAECJQoIQEvEVBQBAgQIECBAgAABAgQIECBAgAABAo0lIABtrP5QGwIECBAgQIAAAQIECBAgQIAAAQIEShQQgJaIqSgCBAgQIECAAAECBAgQIECAAAECBBpLQADaWP2hNgQIECBAgAABAgQIECBAgAABAgQIlCggAC0RU1EECBAgQIAAAQIECBAgQIAAAQIECDSWgAC0sfpDbQgQIECAAAECBAgQIECAAAECBAgQKFFAAFoipqIIECBAgAABAgQIECBAgAABAgQIEGgsAQFoY/WH2hAgQIAAAQIECBAgQIAAAQIECBAgUKKAALRETEURIECAAAECBAgQIECAAAECBAgQINBYAgLQxuoPtSFAgAABAgQIECBAgAABAgQIECBAoEQBAWiJmIoiQIAAAQIECBAgQIAAAQIECBAgQKCxBASgjdUfakOAAAECBAgQIECAAAECBAgQIECAQIkCAtASMRVFgAABAgQIECBAgAABAgQIECBAgEBjCfRsrOqoDQECBOorsOpRX6/bBWaefFrdylYwAQIECBAgQIAAAQIECBAg0DEBI0A75uYsAgQIECBAgAABAgQIECBAgAABAgSaQEAA2gSdpIoECBAgQIAAAQIECBAgQIAAAQIECHRMQADaMTdnESBAgAABAgQIECBAgAABAgQIECDQBAIC0CboJFUkQIAAAQIECBAgQIAAAQIECBAgQKBjAgLQjrk5iwABAgQIECBAgAABAgQIECBAgACBJhAQgDZBJ6kiAQIECBAgQIAAAQIECBAgQIAAAQIdExCAdszNWQQIECBAgAABAgQIECBAgAABAgQINIGAALQJOkkVCRAgQIAAAQIECBAgQIAAAQIECBDomIAAtGNuziJAgAABAgQIECBAgAABAgQIECBAoAkEBKBN0EmqSIAAAQIECBAgQIAAAQIECBAgQIBAxwQEoB1zcxYBAgQIECBAgAABAgQIECBAgAABAk0gIABtgk5SRQIECBAgQIAAAQIECBAgQIAAAQIEOiYgAO2Ym7MIECBAgAABAgQIECBAgAABAgQIEGgCAQFoE3SSKhIgQIAAAQIECBAgQIAAAQIECBAg0DEBAWjH3JxFgAABAgQIECBAgAABAgQIECBAgEATCAhAm6CTVJEAAQIECBAgQIAAAQIECBAgQIAAgY4JCEA75uYsAgQIECBAgAABAgQIECBAgAABAgSaQEAA2gSdpIoECBAgQIAAAQIECBAgQIAAAQIECHRMQADaMTdnESBAgAABAgQIECBAgAABAgQIECDQBAIC0CboJFUkQIAAAQIECBAgQIAAAQIECBAgQKBjAgLQjrk5iwABAgQIECBAgAABAgQIECBAgACBJhAQgDZBJ6kiAQIECBAgQIAAAQIECBAgQIAAAQIdExCAdszNWQQIECBAgAABAgQIECBAgAABAgQINIGAALQJOkkVCRAgQIAAAQIECBAgQIAAAQIECBDomIAAtGNuziJAgAABAgQIECBAgAABAgQIECBAoAkEBKBN0EmqSIAAAQIECBAgQIAAAQIECBAgQIBAxwQEoB1zcxYBAgQIECBAgAABAgQIECBAgAABAk0gIABtgk5SRQIECBAgQIAAAQIECBAgQIAAAQIEOiYgAO2Ym7MIECBAgAABAgQIECBAgAABAgQIEGgCAQFoE3SSKhIgQIAAAQIECBAgQIAAAQIECBAg0DEBAWjH3JxFgAABAgQIECBAgAABAgQIECBAgEATCAhAm6CTVJEAAQIECBAgQIAAAQIECBAgQIAAgY4JCEA75uYsAgQIECBAgAABAgQIECBAgAABAgSaQEAA2gSdpIoECBAgQIAAAQIECBAgQIAAAQIECHRMQADaMTdnESBAgAABAgQIECBAgAABAgQIECDQBAIC0CboJFUkQIAAAQIECBAgQIAAAQIECBAgQKBjAgLQjrk5iwABAgQIECBAgAABAgQIECBAgACBJhAQgDZBJ6kiAQIECBAgQIAAAQIECBAgQIAAAQIdExCAdszNWQQIECBAgAABAgQIECBAgAABAgQINIGAALQJOkkVCRAgQIAAAQIECBAgQIAAAQIECBDomIAAtGNuziJAgAABAgQIECBAgAABAgQIECBAoAkEBKBN0EmqSIAAAQIECBAgQIAAAQIECBAgQIBAxwQEoB1zcxYBAgQIECBAgAABAgQIECBAgAABAk0gIABtgk5SRQIECBAgQIAAAQIECBAgQIAAAQIEOiYgAO2Ym7MIECBAgAABAgQIECBAgAABAgQIEGgCAQFoE3SSKhIgQIAAAQIECBAgQIAAAQIECBAg0DEBAWjH3JxFgAABAgQIECBAgAABAgQIECBAgEATCAhAm6CTVJEAAQIECBAgQIAAAQIECBAgQIAAgY4JCEA75uYsAgQIECBAgAABAgQIECBAgAABAgSaQEAA2gSdpIoECBAgQIAAAQIECBAgQIAAAQIECHRMQADaMTdnESBAgAABAgQIECBAgAABAgQIECDQBAI9m6COy1zFP/7xj+m+++5Ljz/+eOrWrVsaPnx42nfffdN6661XWNa8efPSlVdemR544IE0bdq0NGLEiDRmzJi08847px49ehSeYyMBAgQIECBAgAABAgQIECBAgAABAo0vUKkAdMGCBenHP/5xuvrqqzP5/v37p7lz56ZHH3003Xjjjelb3/pWet/73teqV6ZPn54OO+yw9Nxzz2XbBw0alG6++ebs45577knHHHNM6t27d6tzvCBAgAABAgQIECBAgAABAgQIECBAoDkEKnUL/IUXXpiFn6uvvno666yz0nXXXZduueWWdMghh6RFixalU045Jb300kuteua4447Lws9x48al66+/Pl177bXp8ssvTxtuuGG6++6705lnntnqeC8IECBAgAABAgQIECBAgAABAgQIEGgegcoEoLNnz87Cz+7du6fvfve7abPNNkuxHrewH3TQQdlt7TEa9K677qr1zoQJE9L999+fVl555XT88cenGDEayzrrrJPOOOOM7NybbropzZw5s3aOFQIECBAgQIAAAQIECBAgQIAAAQIEmkegMgHoNddckyIE/ehHP5pGjx69RA8cccQR6Wtf+1radNNNa/vuvPPObH3HHXdMK620Um17rMSt8GPHjk3z589PEYJaCBAgQIAAAQIECBAgQIAAAQIECBBoPoHKBKAxgVEs2223XWEvjBo1Ku25556tAtB4Nmgscft70RIBaCzjx48v2m0bAQIECBAgQIAAAQIECBAgQIAAAQINLlCZSZCmTp2aUW+00UbpmWeeyUZtPvzwwykmRho5cmTaZ599lpgF/vnnn8/OGTBgQGE35dvzCZIKD/q/jb/61a/SwoULC3fHbPJve9vbCvfZWK5Az57//HKORxosXry43MKVtkIE8j6Mi8V63759V8h1y7pIs9W3rHa3V06vXr2arg/ba0tX3h6Pk/G13bxfAfGztFu3bvqwebsw+/8wr37csRQ/Wy3NJxCP5Yol+tD71Obrv6hx3oex3ozvU6Peln8KxP+L+tBXA4GuJ1CZAHTKlCnZMz+ffvrpdNRRR2XP7Yw3iBGAPv744+l3v/tdtn2nnXaq9XLcMh9LHnTWdvxrZbXVVsvW8uPa7s9fn3DCCWnevHn5y1afTz755FajTlvt9KIuAquuumpdylXoihWIXxDaPpqijBrMKqOQdsrInyPczu4ut7lefdjlIDu5wRGA+tru5E4o4fL6sATEBiiiX79+DVALVVgeAe9Tl0evcc7t06dPig9L8wpEoF2P3zVCZM6cOc0Lo+YEKixQiVvgX3/99RQf8ZeceNbnxhtvnC6++OJ02223ZRMj7brrrikmQIowMp8FPv7yGttiae+NSP4ms71ws8JfF5pGgAABAgQIECBAgAABAgQIECBAoBIClRgBGqM8Y1m0aFE2g/upp55auz1oyJAh6Zvf/GaKEaLxnNBf/OIX6cgjj8xGi8at0vHXmfYCznx7796937SzY8RpXLtoGTFiRHrttdeKdtlWskDcxrDKKqtko3/dWlQy7goqLu/DuFz8gSL/Hizz8v+8Aa3MEv9dlu/1f1vEH5ZiErl69OG/r2KtngJxF0T8YTF+ns6cObOel1J2HQVidEv8bJ01q57j3+vYAEWneL+avxeNfmzvPSeqxhaI0WYxuCJ+nnqf2th91V7t8j6M/fH+Jh9M097xtjeuQHwvxiPs6tmH8bPbQoBAYwlUIgCN27rijWH8sh0THRU9G2nvvffOAtAnnnii1gMRjsbzPdv7xS7fHqHamy37779/u7unTZuWjU5t9wA7ShOI21CiryLU9stBaawrtKD4RT3/fos3JTGyu+ylng9IqEd9y27/iiovAtD44xSTFSVe/nXyx8DEL+r6sXzfFVVihJ/xS7s+XFHi5V8n3tfmAWj8sp7/4b/8KymxngLRjxG6RB+2N3dAPa+v7OUXiO/D/A7Ber1PXf5aKmFpBOL3jXq+T/V4hKXpBccQWPEClbgFPtgizIxl7bXXzj63/WedddbJNk2ePLm2Kz8nDzprO/61MmPGjGxt4MCBbXd5TYAAAQIECBAgQIAAAQIECBAgQIBAEwhUJgBdY401Mu5JkyYVsk+fPj3bPnz48Nr+/Jynnnqqtq3lSr59k002abnZOgECBAgQIECAAAECBAgQIECAAAECTSJQmQA0n939z3/+cyH9Qw89lG1/5zvfWdufn3PrrbfWtuUrccvf7bffnr0cM2ZMvtlnAgQIECBAgAABAgQIECBAgAABAgSaSKAyAWjM9D548OD04IMPposuuqhVFzz55JPp8ssvz56Bte2229b2bb311mnYsGFp4sSJ6aabbqptj5VLL700vfLKK2n99ddP48aNa7XPCwIECBAgQIAAAQIECBAgQIAAAQIEmkOgEpMgBXU8lPqII45I3/ve99L555+f7rvvvrTVVlulqVOnpptvvjmbqS9ma990001rPROz2x566KHpO9/5TjrxxBPTvffem2LW9kceeSRbj4eVx4zxcZyFAAECBAgQIECAAAECBAgQIECAAIHmE6hMABr0O+ywQzr33HPTySefnIWY48ePz3pko402ymaH32233ZbooTjnBz/4QRaA3nHHHSk+YomRoV/96lfT6NGjlzjHBgIECBAgQIAAAQIECBAgQIAAAQIEmkOgUgFokEfYecEFF6TXX389xSRGMfv7W83ivvnmm6crrrgiu+X9ueeeSzE50tChQ1P37pV5QkBzfDWqJQECBAgQIECAAAECBAgQIECAAIGSBSoXgOY+ffv2TaNGjcpfLtXneIZofFgIECBAgAABAgQIECBAgAABAgQIEKiGgCGO1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVENAAFqNftQKAgQIECBAgAABAgQIECBAgAABAgQKBASgBSg2ESBAgAABAgQIECBAgAABAgQIECBQDQEBaDX6USsIECBAgAABAgQIECBAgAABAgQIECgQEIAWoNhEgAABAgQIECBAgAABAgQIECBAgEA1BASg1ehHrSBAgAABAgQIECBAgAABAgQIECBAoEBAAFqAYhMBAgQIECBAgAABAgQIECBAgAABAtUQEIBWox+1ggABAgQIECBAgAABAgQIECBAgACBAgEBaAGKTQQIECBAgAABAgQIECBAgAABAgQIVEOgZzWa0bit6NatW+rVq1fjVrBCNevRo0fWmp49e6bu3WX7zdi1eR9G3aMPm+17p9nqW++vkWbsw3qbNGP5/h9rxl77d53z/w/9fPq3SbOt5X0Y9Y73OJbmFMj7Lj7Hz1VL8wnkfRg19x6n+fqvZY3jezB+76jX/40tf263vK51AgQ6V6DbG/+3dG4Vqn31119/PfXt27fajdS65RaYddABy11GewX0+8Wl7e3qkttZd8lu12gCBAgQIECAAAECK0Rgzpw5aeWVV14h13IRAgSWXsCfkZfeqkNHzp8/P82ePbtD5zpp2QR69+6d+vfvn1555ZW0ePHiZTu5k4+u53+PL7/8cie3bukvn/dhnBHfN/EHhLIX1mWLFpc3ePDgFG/+6tGHxVe0tWyBIUOGZKOUFi1alF599dWyi1feChLo169fNsJl2rRpK+iKLlO2QPRh/ot09OPChQvLvoTyVoBAjB4cOHBg0ocrALtOl4jRggMGDMhKj/c4s2bNqtOVFFtvgUGDBqV58+bV7ff0+H7Pf27Xuy3KJ0Bg6QUEoEtv1aEjY4CtN6odolvmk/Lbp+OX9fiw/FOgmb7+Wt5a1IzfO81kvSK+P+IPEUxWhHR9r9GM34v1FWmu0vM/CPpebK5+a1nbljdrxfsbfdlSp3nW89veo//0YfP0W8uatryt2XucljLNtx4/V+vZh/nvpc0no8YEqi3gQYnV7l+tI0CAAAECBAgQIECAAAECBAgQINClBQSgXbr7NZ4AAQIECBAgQIAAAQIECBAgQIBAtQUEoNXuX60jQIAAAQIECBAgQIAAAQIECBAg0KUFBKBduvs1ngABAgQIECBAgAABAgQIECBAgEC1BQSg1e5frSNAgAABAgQIECBAgAABAgQIECDQpQUEoF26+zWeAAECBAgQIECAAAECBAgQIECAQLUFBKDV7l+tI0CAAAECBAgQIECAAAECBAgQINClBQSgXbr7NZ4AAQIECBAgQIAAAQIECBAgQIBAtQUEoNXuX60jQIAAAQIECBAgQIAAAQIECBAg0KUFBKBduvs1ngABAgQIECBAgAABAgQIECBAgEC1BQSg1e5frSNAgAABAgQIECBAgAABAgQIECDQpQUEoF26+zWeAAECBAgQIECAAAECBAgQIECAQLUFBKDV7l+tI0CAAAECBAgQIECAAAECBAgQINClBQSgXbr7NZ4AAQIECBAgQIAAAQIECBAgQIBAtQUEoNXuX60jQIAAAQIECBAgQIAAAQIECBAg0KUFBKBduvs1ngABAgQIECBAgAABAgQIECBAgEC1BQSg1e5frSNAgAABAgQIECBAgAABAgQIECDQpQUEoF26+zWeAAECBAgQIECAAAECBAgQIECAQLUFBKDV7l+tI0CAAAECBAgQIECAAAECBAgQINClBQSgXbr7NZ4AAQIECBAgQIAAAQIECBAgQIBAtQUEoNXuX60jQIAAAQIECBAgQIAAAQIECBAg0KUFBKBduvs1ngABAgQIECBAgAABAgQIECBAgEC1BQSg1e5frSNAgAABAgQIECBAgAABAgQIECDQpQUEoF26+zWeAAECBAgQIECAAAECBAgQIECAQLUFBKDV7l+tI0CAAAECBAgQIECAAAECBAgQINClBQSgXbr7NZ4AAQIECBAgQIAAAQIECBAgQIBAtQUEoNXuX60jQIAAAQIECBAgQIAAAQIECBAg0KUFBKBduvs1ngABAgQIECBAgAABAgQIECBAgEC1BQSg1e5frSNAgAABAgQIECBAgAABAgQIECDQpQUEoF26+zWeAAECBAgQIECAAAECBAgQIECAQLUFBKDV7l+tI0CAAAECBAgQIECAAAECBAgQINClBQSgXbr7NZ4AAQIECBAgQIAAAQIECBAgQIBAtQUEoNXuX60jQIAAAQIECBAgQIAAAQIECBAg0KUFBKBduvs1ngABAgQIECBAgAABAgQIECBAgEC1BQSg1e5frSNAgAABAgQIECBAgAABAgQIECDQpQUEoF26+zWeAAECBAgQIECAAAECBAgQIECAQLUFBKDV7l+tI0CAAAECBAgQIECAAAECBAgQINClBQSgXbr7NZ4AAQIECBAgQIAAAQIECBAgQIBAtQUEoNXuX60jQIAAAQIECBAgQIAAAQIECBAg0KUFBKBduvs1ngABAgQIECBAgAABAgQIECBAgEC1BQSg1e5frSNAgAABAgQIECBAgAABAgQIECDQpQUEoF26+zWeAAECBAgQIECAAAECBAgQIECAQLUFela7eVpHgAABAgQIdFRg6PU3d/TUtzzvoS02e8tjHECAAAECBAgQIECAAIEyBIwALUNRGQQIECBAgAABAgQIECBAgAABAgQINKSAALQhu0WlCBAgQIAAAQIECBAgQIAAAQIECBAoQ0AAWoaiMggQIECAAAECBAgQIECAAAECBAgQaEgBAWhDdotKESBAgAABAgQIECBAgAABAgQIECBQhoAAtAxFZRAgQIAAAQIECBAgQIAAAQIECBAg0JACAtCG7BaVIkCAAAECBAgQIECAAAECBAgQIECgDAEBaBmKyiBAgAABAgQIECBAgAABAgQIECBAoCEFBKAN2S0qRYAAAQIECBAgQIAAAQIECBAgQIBAGQIC0DIUlUGAAAECBAgQIECAAAECBAgQIECAQEMKCEAbsltUigABAgQIECBAgAABAgQIECBAgACBMgQEoGUoKoMAAQIECBAgQIAAAQIECBAgQIAAgYYUEIA2ZLeoFAECBAgQIECAAAECBAgQIECAAAECZQgIQMtQVAYBAgQIECBAgAABAgQIECBAgAABAg0pIABtyG5RKQIECBAgQIAAAQIECBAgQIAAAQIEyhAQgJahqAwCBAgQIECAAAECBAgQIECAAAECBBpSoGdD1kqlCBAgQKDpBbp98XNp1Tq1YubJp9WpZMUSIECAAAECBAgQIECAQNUEjACtWo9qDwECBAgQIECAAAECBAgQIECAAAECNQEBaI3CCgECBAgQIECAAAECBAgQIECAAAECVRMQgFatR7WHAAECBAgQIECAAAECBAgQIECAAIGagAC0RmGFAAECBAgQIECAAAECBAgQIECAAIGqCQhAq9aj2kOAAAECBAgQIECAAAECBAgQIECAQE1AAFqjsEKAAAECBAgQIECAAAECBAgQIECAQNUEBKBV61HtIUCAAAECBAgQIECAAAECBAgQIECgJiAArVFYIUCAAAECBAgQIECAAAECBAgQIECgagIC0Kr1qPYQIECAAAECBAgQIECAAAECBAgQIFATEIDWKKwQIECAAAECBAgQIECAAAECBAgQIFA1AQFo1XpUewgQIECAAAECBAgQIECAAAECBAgQqAkIQGsUVggQIECAAAECBAgQIECAAAECBAgQqJqAALRqPao9BAgQIECAAAECBAgQIECAAAECBAjUBASgNQorBAgQIECAAAECBAgQIECAAAECBAhUTUAAWrUe1R4CBAgQIECAAAECBAgQIECAAAECBGoCAtAahRUCBAgQIECAAAECBAgQIECAAAECBKomIACtWo9qDwECBAgQIECAAAECBAgQIECAAAECNQEBaI3CCgECBAgQIECAAAECBAgQIECAAAECVRMQgFatR7WHAAECBAgQIECAAAECBAgQIECAAIGagAC0RmGFAAECBAgQIECAAAECBAgQIECAAIGqCQhAq9aj2kOAAAECBAgQIECAAAECBAgQIECAQE1AAFqjsEKAAAECBAgQIECAAAECBAgQIECAQNUEBKBV61HtIUCAAAECBAgQIECAAAECBAgQIECgJiAArVFYIUCAAAECBAgQIECAAAECBAgQIECgagIC0Kr1qPYQIECAAAECBAgQIECAAAECBAgQIFATEIDWKKwQIECAAAECBAgQIECAAAECBAgQIFA1AQFo1XpUewgQIECAAAECBAgQIECAAAECBAgQqAkIQGsUVggQIECAAAECBAgQIECAAAECBAgQqJqAALRqPao9BAgQIECAAAECBAgQIECAAAECBAjUBASgNQorBAgQIECAAAECBAgQIECAAAECBAhUTUAAWrUe1R4CBAgQIECAAAECBAgQIECAAAECBGoCAtAahRUCBAgQIECAAAECBAgQIECAAAECBKomIACtWo9qDwECBAgQIECAAAECBAgQIECAAAECNQEBaI3CCgECBAgQIECAAAECBAgQIECAAAECVRMQgFatR7WHAAECBAgQIECAAAECBAgQIECAAIGagAC0RmGFAAECBAgQIECAAAECBAgQIECAAIGqCQhAq9aj2kOAAAECBAgQIECAAAECBAgQIECAQE1AAFqjsEKAAAECBAgQIECAAAECBAgQIECAQNUEBKBV61HtIUCAAAECBAgQIECAAAECBAgQIECgJiAArVFYIUCAAAECBAgQIECAAAECBAgQIECgagIC0Kr1qPYQIECAAAECBAgQIECAAAECBAgQIFATEIDWKKwQIECAAAECBAgQIECAAAECBAgQIFA1AQFo1XpUewgQIECAAAECBAgQIECAAAECBAgQqAkIQGsUVggQIECAAAECBAgQIECAAAECBAgQqJqAALRqPao9BAgQIECAAAECBAgQIECAAAECBAjUBASgNQorBAgQIECAAAECBAgQIECAAAECBAhUTUAAWrUe1R4CBAgQIECAAAECBAgQIECAAAECBGoCAtAahRUCBAgQIECAAAECBAgQIECAAAECBKomIACtWo9qDwECBAgQIECAAAECBAgQIECAAAECNQEBaI3CCgECBAgQIECAAAECBAgQIECAAAECVRMQgFatR7WHAAECBAgQIECAAAECBAgQIECAAIGaQM/aWkVXHnrooXT22WencePGpUMPPbSwlfPmzUtXXnlleuCBB9K0adPSiBEj0pgxY9LOO++cevToUXiOjQQIECBAgAABAgQIECBAgAABAgQINL5ApQPQmTNnpuOPPz5NmTIlrbvuuoW9MX369HTYYYel5557Lts/aNCgdPPNN2cf99xzTzrmmGNS7969C8+1kQABAgQIECBAgAABAgQIECBAgACBxhao9C3wp59+ehZ+vlkXHHfccVn4GSNEr7/++nTttdemyy+/PG244Ybp7rvvTmeeeeabnW4fAQIECBAgQIAAAQIECBAgQIAAAQINLFDZAPR3v/tduu2229KAAQPa5Z8wYUK6//7708orr5yNFO3fv3927DrrrJPOOOOM7Pb3m266KcVIUgsBAgQIECBAgAABAgQIECBAgAABAs0nUMkA9MUXX8wCzAgy99tvv3Z75c4778z27bjjjmmllVZqdVzcCj927Ng0f/78FCGohQABAgQIECBAgAABAgQIECBAgACB5hOoXAC6ePHibDTnnDlz0re//e0lgs2WXfToo49mL+P296IlAtBYxo8fX7TbNgIECBAgQIAAAQIECBAgQIAAAQIEGlygcpMgXXbZZenhhx9OBx10UNp0003T448/3m4XPP/889m+9m6Tz7fnEyS1V9DPfvaztHDhwsLdEa4OGzascJ+N5Qr07PnPL+e+ffumCMIt/xRYZZVVmoYi78OocK9evVIz1T3q3Gz1jTo368K6WXvu3/XWh/+2qOda/Czt1q2bn0/1RK5z2S3/b4zHNpmcs87gdSq+R48eWcnRh96n1gm5zsXmfRiXacb3qXXmaari4/9FfdhUXaayBEoRqFQAGmHnBRdckEaOHJkOPvjgtwSaPXt2dkwedLY9YbXVVss25ce13Z+//sEPfpDmzZuXv2z1+eSTT06jR49utc2L+gr069evvheoQ+mz6lBmXmT+dZy/bpbPffr0SfFR9sK6bNHOKa9Zv647R6sxr6oPV2y/8F6x3vW6mj8c1Et2xZXbjO9TV5xO81wp/hDhjxHN019FNY1Aux6/a8S14m5UCwECjSdQmQA0Ashjjz02m7gobn1v+dfyIvb4y+vcuXOzXauuumrRISl/g9JeuFl4UoU2zjrogLq1pt8vLq1b2QomQIAAAQIECBAgQIAAAQIECBAgkAtUJgA9++yz07PPPpsOP/zwtP766+fta/dz9+7ds9nf468z7QWc+fa3+uteBK+LFi0qvFaMRp0+fXrhvkbfWM8vjnqYROgdofWMGTOa7taiZrOu19duy1tR4g8U+R8pyrwe6zI12y+r3qPN6vEzpP3WdM09/fv3r2vD9WFdeWuFx+228f/jzJkza9usNJdA9GE+Sin6sb33nM3Vqq5X2xhtFoMumvF9atfrreIW579rxN74PdEov2KnZtga34sLFiyoy+8a0f64xT5+dlsIEGgsgXpmASuspffee2+6+uqr05Zbbpk+8pGPLPV1hwwZkuL5nu39UpBvf6vbjfbcc892rzlt2rSm/c+xeFxsu01dph31eMOQ/3IQoVmz/XLQbNbL1NnLcPAbb7xRe05dPFe3Hl8nrJehQ5bj0HoHoPX42liO5lby1HoHoPpwxXzZxB+WInjhvWK863GVln+Ij9Alfmm3NJ9AfC9G6BJ92N7cAc3Xqq5V4/hezO8QrNf71K4l2nmtjX6sZx/mv5d2XgtdmQCBIoFKBKDXXHNN1rYJEyakPfbYo1U781Gcd911V9p9991T/EJ3ySWXZMe8VQAaf6GNZeDAgdln/xAgQIAAAQIECBAgQIAAAQIECBAg0FwClQhAY2RDfETYmQeeeTfEiLJY4i88MaIzjsuXNdZYI1t96qmn0tZbb51vrn2O7bFssskmtW1WCBAgQIAAAQIECBAgQIAAAQIECBBoHoFKBKAnnnhiu+K/+c1vUszSvtNOO6Vjjjmm1XGx7ZZbbkm33npr2n///Vvti0mSbr/99mzbmDFjWu3zggABAgQIECBAgAABAgQIECBAgACB5hDo3hzVrE8tY9TnsGHD0sSJE9NNN93U6iKXXnppeuWVV7IJlcaNG9dqnxcECBAgQIAAAQIECBAgQIAAAQIECDSHQCVGgHaUOmZnO/TQQ9N3vvOdFKNIYzKlESNGpEceeSRbj4eVH3nkkdksbh29hvMIECBAgAABAgQIECBAgAABAgQIEOg8gS49AjTYd9hhh+wW+aFDh6Y77rgjnXfeeVn4GSNDTzvttDR69OjO6x1XJkCAAAECBAgQIECAAAECBAgQIEBguQQqPwJ07733TvHxZsvmm2+errjiiuyW9+eeey7F5EgRiHbv3uXz4Tdjs48AAQIECBAgQIAAAQIECBAgQIBAwwtUPgBdlh4YPHhwig8LAQIECBAgQIAAAQIECBAgQIAAAQLVEDDEsRr9qBUECBAgQIAAAQIECBAgQIAAAQIECBQICEALUGwiQIAAAQIECBAgQIAAAQIECBAgQKAaAgLQavSjVhAgQIAAAQIECBAgQIAAAQIECBAgUCDgGaAFKDYRIECAQGMLbP7gw3Wp4ENbbFaXchVKgAABAgQIECBAgAABAp0nYARo59m7MgECBAgQIECAAAECBAgQIECAAAECdRYQgNYZWPEECBAgQIAAAQIECBAgQIAAAQIECHSegAC08+xdmQABAgQIECBAgAABAgQIECBAgACBOgsIQOsMrHgCBAgQIECAAAECBAgQIECAAAECBDpPQADaefauTIAAAQIECBAgQIAAAQIECBAgQIBAnQUEoHUGVjwBAgQIECBAgAABAgQIECBAgAABAp0nIADtPHtXJkCAAAECBAgQIECAAAECBAgQIECgzgIC0DoDK54AAQIECBAgQIAAAQIECBAgQIAAgc4TEIB2nr0rEyBAgAABAgQIECBAgAABAgQIECBQZwEBaJ2BFU+AAAECBAgQIECAAAECBAgQIECAQOcJCEA7z96VCRAgQIAAAQIECBAgQIAAAQIECBCos4AAtM7AiidAgAABAgQIECBAgAABAgQIECBAoPMEBKCdZ+/KBAgQIECAAAECBAgQIECAAAECBAjUWUAAWmdgxRMgQIAAAQIECBAgQIAAAQIECBAg0HkCAtDOs3dlAgQIECBAgAABAgQIECBAgAABAgTqLCAArTOw4gkQIECAAAECBAgQIECAAAECBAgQ6DwBAWjn2bsyAQIECBAgQIAAAQIECBAgQIAAAQJ1FuhZ5/IVT4AAAQIECBAgQIAAAQIECBAg0AQC8+fPT3fccUeaOHFieumll9Lw4cPTO97xjuxj1VVXfcsWTJ48Of3lL39Jjz32WJo3b14aPXp09rH++uu3e+7TTz+dXn755dSjR4+0xRZbtHvc+PHj09y5c9OAAQPSxhtvXDtu0qRJacqUKal///5p5MiR2fp1112Xfd5mm23Su9/97tqxLVf+8Ic/pP/93/9NUeehQ4emESNGpJ122in16tWr5WFLrL/22mvp4Ycfzj5iPdo4ZsyYtN566y1xrA2NIyAAbZy+UBMCBAgQIECAAAECBAgQIECAQKcIXHzxxemoo45KL7zwwhLX79evX/rBD36QDjnkkCX2xYYZM2akY489Np155plpwYIFSxzzkY98JJ1zzjlp9dVXX2JfnPfzn/88rbLKKmnWrFlL7M837LnnninC0l133TXdcMMN+eZ08sknp3PPPTd94AMfSMcdd1x63/vel2bOnFnbH+V/+9vfrr2++eab0+GHH56eeOKJ2rZ8ZYMNNkinnHJK+tjHPpZvqn1+44030hlnnJGOPvroLNyt7fjXyv7775/OPvvsLKBtu8/rzhcQgHZ+H6gBAQIECBAgQIAAAQIECBAgQKDTBK6++up08MEHp8WLF6eBAwem9773vWnQoEHp/vvvz0Y6RjB56KGHpkcffTQLQltWNMLGzTffPD311FPZ5jXWWCNttdVWWaAZ58cIzauuuir9/ve/T7/97W/T2LFjW55e2vqcOXNShJAtw88o/IMf/GDtGj/5yU/Sl770paydMeI0Rm6OGjUq3XPPPdmo1whY99lnn2w06t577107L8r+0Ic+lG6//fZs25prrpnGjRuXjTp98MEH04QJE9Jll12WYlTp7373u2wkau1kKw0hIABtiG5QCQIECBAgQIAAAQIECBAgQIBA5wj813/9VxYKxi3vcat5jMbMlxgRGreSP/PMM+n888/PRlnGiNB8OeKII2rhZ4SLp512Wurdu3e+OxsV+v/+3//Lbkn/9Kc/nR566KHUs2f5cVQErLEccMAB6TOf+Uw2mvTee++tBa5/+9vfUtQ1Qt511lkn/fKXv0zbb799rZ733Xdfev/7358FqJ/85CezkaSrrbZatv/UU0+thZ+f//znU7xuaXT55Zdno2OfffbZLGCNENTSWAImQWqs/lAbAgQIECBAgAABAgQIECBAgMAKE5g6dWqKcDCWCDBbBnuxbe21107HH3986tOnT3rb296WYsRjvvzP//xPuuCCC7KX++23XxZ2tgw/Y8eXv/zldMIJJ2TH/PWvf00//OEPs/V6/LPddtuliy66KO2www7ZrfJxS3y+RBviuaSxtA0/Y1uM6DzyyCNjNc2ePTsbtRrrzz33XHZbfKzvsssuKUaRtjXad999009/+tM4JIXJtddem637p3EEBKCN0xdqQoAAAQIECBAgQIAAAQIECBBYoQIxyjEP9K688sr06quvLnH9GFX5+uuvZ0HpjjvuWNufP4szbiePZ222t3zxi1/MRl3G/ksuuaS9w5Z7+xe+8IXUvfuSUdfChQtrzw19z3ve02rkZ8uLxgjRCGwjpI3b+mP5xS9+kbU91t+sjWEUt8bHEudYGkug/DHHjdU+tSFAgAABAgQIECBAgAABAgQIEGhHIEZsxjM/4/mcf/zjH7MZ1vfYY4+08847Z7eExzNBu3Xrln20LSJme49l2LBhaaONNmq7u/Z6pZVWSjE681e/+lX2rM2YUCjKLHuJmdyLlqjn9OnTs13bbrtt0SHZtr59+6Yf/ehHrfZPnDgxex1tiNnsp02b1mp/yxfxPNGXXnopa2PL7dY7X0AA2vl9oAYECBAgQIAAAQIECBAgQIAAgU4TiNvYd99992zSo1deeSX97Gc/yz5iZOfWW2+dIhCNW9zXXXfdVnVsGYC22lHwImZYjyVGksZt5eutt17BUcu3qb0Q9sUXX6wVvKzXzWeLnzt3booJnpZmefLJJ1O9Qt6lub5jlhRYclzwksfYQoAAAQIECBAgQIAAAQIECBAgUFGBCPbuuOOO7FmdI0eOrLVy0aJF2ajQb3zjG9kIz/w5l3FABHyTJ0/Ojl199dVr57S3ErPK50uErGUvMbFS//79C4ttOTN8y3oUHtxm4z/+8Y82W976ZcwaH89WtTSOgAC0cfpCTQgQIECAAAECBAgQIECAAAECnSIQt3/HbPAxIdLf//73dNZZZ2WjQvPng86fPz/FDOgxCVAscQt7Ppry+eeff8s6twwShwwZssTxMTv7my0RKnZ0yesZ57ccDbo05cXET7FssskmacaMGUv9UdTGpbmeY+ojIACtj6tSCRAgQIAAAQIECBAgQIAAAQJNKbDhhhummFDouuuuy0YynnTSSbV2XHHFFbX1/JmbkyZNqm1rbyU/JoLc1kpcAABAAElEQVTTliNG80mLFixY0N6p2WjTN3v2Zrsn/mtHy1vjWwaxReddeOGFWfh7++23Z7vzNsazQHv16pVWXXXVpfqoxzNOi+pr29IJCECXzslRBAgQIECAAAECBAgQIECAAIHKCcTkR7vuumsaPnx4Gj9+/BLti8l/jjrqqLTllltm++6///4skIwX+e3y8UzPmECpvWXKlCnptttuy3bHM0WjzHxZeeWVs9WYqf21117LN7f6/PDDD6d58+a12rYsLwYMGFALXW+66aZa/duWEaNQjzzyyPSlL30pnXfeednut7/97dnnqN+NN97Y9pTa6zg3Zpjfcccd0+GHH17bbqUxBASgjdEPakGAAAECBAgQIECAAAECBAgQWOECq622WopQ8Omnn07HHHNMu9ePZ37G8q53vas2g3sEhTGLfCwRkkZIWLR873vfS7Nnz8527bPPPq0OiZnV8+Wcc87JV2uf49b3uPV+eZeoaywRpl5zzTWFxV188cUpfz7pXnvtlR3z6U9/OoVRLEcccUStHdmGFv/EyNE777wz3X333TWTFrutdrKAALSTO8DlCRAgQIAAAQIECBAgQIAAAQKdJbD99tunuOU9lggG49mfLW9Hj+defuUrX0kPPvhgdsyee+6ZfY5/4tbyr33ta9nrP/zhD+m9731vihnQ8+XVV19NBx98cO25oWPGjEmf+9zn8t3Z53333TfFbPOxRFB6ySWXpLjdPUZ8xm3ocb0//elPqU+fPtkxHf3n61//eu2ZpQcccEA2y33LwPbSSy/NRn5G+f/xH/+RPvzhD2eXWnPNNbN6xYtnnnkmjR07NsUo2Hx56qmn0ve///102GGHZZtitGketubH+Nz5Aj07vwpqQIAAAQIECBAgQIAAAQIECBAg0BkC8QzOCD632WabFLOlR3gXozmHDRuWjeiM0G/u3LlZ1T772c+m+Gi5HH300Smej3nllVem3//+91koutZaa6WYVKllGLrpppumq666aokgc911183KjMmV4joHHnhgihnd43mb+cRHcc2Ykf6CCy5oeellWo9b7SPkjBGoL7zwQvrP//zPLLSMZ3zGLfzTp0/Pyovnk0Y9W96m/8UvfjFNmDAhnX/++dnncePGpYEDB6aY6Cjani/R5htuuKEWtObbfe58ASNAO78P1IAAAQIECBAgQIAAAQIECBAg0GkCo0aNykZZ7rfffikC0bhd/dFHH02PP/54itnfYwb0n/70p9lHBJMtl5glPiZG+uUvf5k22GCDbNfkyZNr4efQoUPTt771rWzUZDxntGg5++yzU9z+nk+OFCMzI/yMGdiPPfbY7Lr5KNGi85d223bbbZceeeSRtP/++2cBbQSu8TrCzyj/M5/5TNbuCH9bLhHIxjNBb7nllhRBbhjFKNU8/IxzP/GJT6SHHnooC5Jbnmu9MQS6/d8zHP75EIfGqE/lahHfEPlfSpqtcase9fW6VXnmyaeVXnYMhx80aFCKhyvHX4aaaWk263rZxl/Y4q9oscyaNSv762PZ12JdtmhxeXGbyOufOrB4ZwlbN/pYfcp+aIvNSqhdNYqIN6pr3XBL3RrDum60rQqO51XF/48vv/xyq+1eNI9A//79s1/QosZTp05tdUti87RCTSMsiFFC8b3Y8nZLMs0jEM84HDx4cFbhCIbilmBLcwpEwBa/o8dIx3os+e+l9Sh7RZUZX98x6vPZZ5/Nfj/bbLPNUoScS7vEREYRKsbneL5nTCIUAeLSLnHtGG0ZoWvbIHJpy1ia42LSor///e/piSeeSOuss042oVOM4Fya5fXXX0+PPfZYNpJ0vfXWyyaQitnhLY0rsPRfgY3bBjUjQIAAAQIECDS1wOYPPly3+v9j5/fVrWwFEyBAgAABAtUTiD+gvvOd78w+OtK6+ONdjLTs6BKhacuJkTpazludF6M4N9544+zjrY5tuz+C0i222CL7aLvP68YUcAt8Y/aLWhEgQIAAAQIECBAgQIAAAQIECBAgUIKAALQEREUQIECAAAECBAgQIECAAAECBAgQINCYAgLQxuwXtSJAgAABAgQIECBAgAABAgQIECBAoAQBAWgJiIogQIAAAQIECBAgQIAAAQIECBAgQKAxBQSgjdkvakWAAAECBAgQIECAAAECBAgQIECAQAkCAtASEBVBgAABAgQIECBAgAABAgQIECBAgEBjCghAG7Nf1IoAAQIECBAgQIAAAQIECBAgQIAAgRIEBKAlICqCAAECBAgQIECAAAECBAgQIECAAIHGFBCANma/qBUBAgQIECBAgAABAgQIECBAgAABAiUICEBLQFQEAQIECBAgQIAAAQIECBAgQIAAAQKNKdCzMaulVgQIECBAgMDSCqx61NeX9tClPm52HPmxA5f6eAcSIECAAAECBAgQIECgUQWMAG3UnlEvAgQIECBAgAABAgQIECBAgAABAgSWW0AAutyECiBAgAABAgQIECBAgAABAgQIECBAoFEFBKCN2jPqRYAAAQIECBAgQIAAAQIECBAgQIDAcgsIQJebUAEECBAgQIAAAQIECBAgQIAAAQIECDSqgEmQGrVn1IsAAQIECBAgQIAAAQIECBAgUGeByZMn1/kK9Sl+rbXWqk/BSq2kgBGglexWjSJAgAABAgQIECBAgAABAgQIECBAIASMAPV1QIAAAQIECBAgQIAAAQIECBDowgKrHvX1pmn9zJNPa5q6qmjjCBgB2jh9oSYECBAgQIAAAQIECBAgQIAAAQIECJQsIAAtGVRxBAgQIECAAAECBAgQIECAAAECBAg0joAAtHH6Qk0IECBAgAABAgQIECBAgAABAgQIEChZoKEC0DfeeKPk5imOAAECBAgQIECAAAECBAgQIECAAIGuLFDXSZAmTZqUXnzxxTR//vy0ePHimnMEnQsXLkyLFi1Ks2fPTi+99FK64YYb0jbbbJOOPvro2nFWCBAgQIAAAQIECBAgQIAAAQIECBAgsDwCdQlA//SnP6Wjjjoq3XXXXctUt6222mqZjncwAQIECBAgQIAAAQIECBAgQIAAAQIE3kyg9AB02rRpaa+99spGfr7ZhdvuW3vttdPb3/72tpu9JkCAAAECBAgQIECAAAECBAgQIECAQIcFSg9ATzzxxFr4udNOO6U99tgjrbzyyukzn/lM6tOnT7rggguy296feeaZ9Otf/zo9+eSTafjw4elvf/tb6tWrV4cb4kQCBAgQIECAAAECBAgQIECAAAECBAi0FSg9AH3wwQeza3zgAx9It9xyS+16p5xyShZ2brzxxmns2LHZ9iOPPDLtvPPO6b777kunn356dtt87QQrBAgQIECAAAECBAgQIECAAAECBAgQWE6B0meBnzhxYlalww8/vFXV3vWud2Wv77jjjtr2AQMGpFtvvTVtuOGG6dhjj01PP/10bZ8VAgQIECBAgAABAgQIECBAgAABAgQILK9AqQHoggUL0vPPP5/VacSIEa3qNnLkyOz1+PHjW23v169f2mWXXdKcOXPS1Vdf3WqfFwQIECBAgAABAgQIECBAgAABAgQIEFgegVID0HiG5+DBg7P69OzZ+u769gLQOHjHHXfMznnkkUeyz/4hQIAAAQIECBAgQIAAAQIECBAgQIBAGQKlBqBRoXwm97a3s2+yySZZfWOyo/nz57eqe9++fbPXjz76aKvtXhAgQIAAAQIECBAgQIAAAQIECBBodIHLLrssbbnlltkjHpenrmWV09E6RBviY9asWR0toiHPq1sA+tOf/rRVg2MEaIwKXbhwYbr77rtb7bv++uuz16uuumqr7V4QIECAAAECBAgQIECAAAECBAgQaHSBl156KcXE4JMmTVquqpZVTkcrEW2Ij8jvqrSUHoAeeOCBqVu3bumKK65IH/3oR9MDDzyQecXt8dtuu222fthhh6UXXnghvfHGG+m3v/1tuvLKK7PtG220UZVstYUAAQIECBAgQIAAAQIECBAgQKALCGy++ebpS1/6Utppp52Wq7VllbNclajgya0f1FlCA3fYYYesw88888x01VVXpXvuuScLO6PoI444It11110pZopfd9110+qrr54i2c6XCE8tBAgQIECAAAECBAgQIECAAAECBJpJ4N3vfneKj+VdyipneetRtfNLD0AD6KSTTkqLFy9OP//5z9OGG25YM9t9993TF77whXT22Wdn+1uGn4ceemjabrvtasdaIUCAAAECBAgQIECAAAECBAgQIFBPgbhDefz48enZZ59NcWfy6NGj05AhQ5a45NSpU7NBfOuss06KuWxuv/32LNuKEZ99+vRJr7zySnrxxRfTwIED09prr73E+X/5y19STP698sorp6233jqtt9566dVXX02TJ09Oa665Zu2a7ZXz2GOPpe7du6d4xGTcUf3444+n++67Lw0aNChtscUWhdfMKzF37tz0xBNPZOe89tprWVYXc/istdZa+SGV/1z6LfAhFl8IP/7xj9Pzzz+fTjjhhBpi3Bp/1llnpQsvvDDtscce2SjQ97znPen8889P5513Xu04KwQIECBAgAABAgQIECBAgAABAgTqJTBt2rT0xS9+MQsid9lll/TZz342u3196NCh6Rvf+EaaN29eq0tHdjVq1Kh0+eWXpx133DHFObvttluWbcWxF110Ubb/W9/6VqvzZsyYkR0fIeXBBx+c9tlnn7T++uunz3/+87Vzzj333No57ZUzduzYbIRpZG0R0sZk41Fe5Gtve9vb0tFHH50WLVpUKydf+e///u8sYN1ss83Sxz/+8RQDEN/73vdm58Qt+3PmzMkPrfTnuowAzcVWW221FLfEt10+9alPpfiwECBAgAABAgQIECBAgAABAgQIEFiRAjGCMuatiVGcMVrzy1/+cjaycsKECemHP/xh+v73v5/uv//+bH8M5mu5nHHGGdmjHSMMjfAwgs0YAVq0xHU+8YlPZJOBv/Od78xCzxixeeONN6aYPHxZJwOPmdljfp0o99hjj80eLRltiHl4TjzxxLTBBhukQw45pFaVOOaYY47JRpdGwBt3Xr/88sspJiOPCcpjkGKEsV//+tdr51R1pa4BaFu0GC4cw3Wjsy0ECBAgQIAAAQIECBAgQIAAAQIEVrRABH8RHMZjG2PumjXWWCOrwp577pkOOuigNG7cuHTnnXdmj3ZsO4Av5rX50Y9+lIWmcVLcXt7eEkFqTP691VZbZeXFHdOxxCjQmOzoq1/9anunFm6PADQC24ceeii7+zoO+tznPpeNZI3HTcZozzwAjXpFO2P5yU9+kj72sY9l6/FPXDdGv8Y5l1xySZcIQOtyC3wuGml5DP390Ic+lOIZCTHp0eDBg7NkPBLr0047LcWQYwsBAgQIECBAgAABAgQIECBAgACBFSEQoy9jiRGSefiZXzfyq+9+97vZy9NPPz3fXPscz82MW8fzZaWVVspXl/h89dVXZ9tidGYefuYHRRkbb7xx/nKpP8et7m3LiuA2lqeffrpWzuzZs9P3vve9bGRoy/AzP2CvvfbKVuM5pF1hqcsI0EiZY2jtxRdfnA3LbQs5f/78LGGPlP3UU0/NEucYemwhQIAAAQIECBAgQIAAAQIECBAgUC+BhQsXZrewR/kf/OAHCy+Tb4/RnvFczR49etSOi0mI2t4WX9vZYiXOi8mV4tjtt9++xZ5/rkaZ8SzRmJxoWZaYvKjtEhMqxRJ5W77EAMR4zmjLJQYhxvViMqa4DT6W8OgKS+kBaHRwDOW97rrrMr+45T1mtxo2bFj2YNgYrhsza/31r3/NkukpU6Zkx//mN79JH/7wh7uCuTYSIECAAAECBAgQIECAAAECBAh0gsCTTz6ZFixYkPr165fdpVxUhRgF2qtXryxQfOaZZ9Lw4cNrh7Vcr20sWPnb3/6WPSM0Rpi294zQeP7msi5FM7f37t07KyaeDdpyiYzuqquuym6NjzA2ZqnPl549S48E86Ib8nPprf31r39dCz9jJqqTTz45m5mqbesXL16crrzyymxmrUmTJqUDDzwwC0YHDBjQ9lCvCRAgQIAAAQIECBAgQIAAAQIECCy3QASbscTIxwgMi0Zzxr58ZGR+fH7h9sLMfH/+OQ8lYxb49pYYJLisSww0XJol2vbJT34yXXbZZdnhMVN83Aofs8FvueWWKSYu32abbZamqEocU3oAmj9g9f3vf3+KUZ0thwm3FIsO+/jHP55Gjx6dPVw2viDOO++8dOSRR7Y8zDoBAgQIECBAgAABAgQIECBAgACBUgRi1GWEmvH4xpdeeikNHTp0iXLjzuV8NGXMZ9ORJWZkj1GW+XXWXHPNJYpp+czOJXYu54Z4/miEn/37988GKu6www6tSozMLpYYoNgVlqWLjZdBIm5tj+WEE05oN/xsWVw8uyCeFxrL73//+5a7rBMgQIAAAQIECBAgQIAAAQIECBAoTSAG6m2yySZZeZdffnlhufn2sWPHpjeb5Kjw5H9tjPAznvEZS8zO3naJ8DVmiK/Xctddd2VFx8TkbcPP2BEzyceSj3TNXlT4n1ID0ECLZDuGD8fIzqVdYuhtLC2fRbC05zqOAAECBAgQIECAAAECBAgQIECAwNIK5LO8x+zs8WzMlssDDzyQTjvttGzTl7/85Za7lnk9ZpmPJR4PecMNN9TOnzp1ajrggAPS66+/XttW9krc4h5LDDZsG3LeeOON6ZRTTsn2R47XFZZSA9BItzfffPNsmHA+EnRpEOMZoLGMGTMm++wfAgQIECBAgAABAgQIECBAgAABAvUQ2GuvvbLnYb788svZxN2f+tSn0kknnZQ9M3PbbbdNr732WvrhD3+YhZTLc/14xubxxx+f4lmfMRJzxIgR2XM3Y9b2+++/P73jHe/Iiq/HhEQxQXk8rzRu5x83blw67rjj0ve///202267pb333jvL4GL/7Nmzs/YuTzub4dxSA9Bo8Pbbb5+1O26Bz5+X8GYQc+bMSZdcckl2SD40+M2Ot48AAQIECBAgQIAAAQIECBAgQIDA8gjEJN7nnHNONhv8z3/+8/Rf//Vf6Ve/+lWK297jlvXDDz98eYqvnXv00Uena6+9NsVE4THiM2aV33XXXbMAdIsttsiOy0dr1k4qYWXUqFHZDPDxLNK//OUv6Tvf+U42EXmMcD311FPTn/70p7TddttlV4rnhVZ9KX0SpG9/+9vptttuS4F3yCGHZENqhwwZUug4ZcqU9LnPfS498sgj6YMf/GDab7/9Co+zkQABAgQIECBAgAABAgQIECBAgECZApFJxcfkyZOzYDIe59i3b9/CS3zzm99M8dHe8tWvfjXFR9Gy++67p/hou0yfPj3bNGDAgNqu9sqZOXNm7Zi2K8OGDSschBijPXfeeeesbS+88ELacMMN01prrVU7/dZbb62t5ytLM5gxP7aZPpcegE6cODF95Stfyb6ALrzwwiw933///dPIkSNT3iHPPfdceuyxx9Kll15ae95BDAPOn8HQFnDfffdNm266advNXhMgQIAAAQIECBAgQIAAAQIECBBYLoEIBVsGg8tVWIuT41b7f/zjH9kzRdve9RzZWD5RUYw6rdcSkz4NHz48+6jXNZqh3NID0BhSGw9TzZd4lsD555+fv2z381lnndXuvs0220wA2q6OHQQIECBAgAABAgQIECBAgAABAo0mECMur7nmmnTEEUekiy66qJZtTZgwIbvFfsaMGel973tfNjKz0epetfqUHoBWDUh7CBAgQIAAAQIECBAgQIAAAQIECCyrQDwm8vrrr8+ewRnP5Fx77bXT4sWL04svvpgVFfPoREBqqb9A6QFoPDB24cKFpdZ8lVVWKbU8hREgQIAAAQIECBAgQIAAAQIECBCop0D//v1TTDp07rnnpt/+9rdp0qRJKSY8igmR3vWud6WPfvSjSeZVzx74d9mlB6D9+vX7d+nWCBAgQIAAAQIECBAgQIAAAQIECHRRgcjJvva1r2UfXZSgIZrdvexanHfeedkkSPE8g6VdjjvuuLTNNtukk046aWlPcRwBAgQIECBAgAABAgQIECBAgAABAgTeUqD0APTaa69NP/rRj9LTTz/9lhfPD7jzzjvTvffem5YlNM3P9ZkAAQIECBAgQIAAAQIECBAgQIAAAQLtCZR+C3x7FyravmjRojRx4sT08MMPZ7v79u1bdJhtBAgQIECAAAECBAgQIECAAAECBAgQ6JDAcgWgu+22W7rttttaXXjBggXZ67322it17/7mA0zj2Jj9Kl+23HLLfNVnAgQIECBAgAABAgQIECBAgAABAgQILLfAcgWgp59+eho9enTKQ8+WtSna1nJ/2/VRo0alPffcs+1mrwkQIECAAAECBAgQIECAAAECBAgQINBhgeUKQN/+9renc845J/35z3+uVeCmm25Kzz77bNpll13SeuutV9tetNKrV6+0yiqrpA022CB9/OMfTwMHDiw6zDYCBAgQIECAAAECBAgQIECAAAECBAh0SGC5AtC44qc//ensI7963BYfAegXvvCFFOsWAgQIECBAgAABAgQIECBAgACBxhWYefJpjVs5NSNQgsByB6Bt63DggQembbbZJo0cObLtLq8JECBAgAABAgQIECBAgAABAgQIECCwQgVKD0D33XffFdoAFyNAgAABAgQIECBAgAABAgQIECBAgEB7Am8+TXt7Z9lOgAABAgQIECBAgAABAgQIECBAgACBJhAofQRoE7RZFQkQIECAAAECBAgQIECAAAECBP4lsPmDDzeNxUNbbNY0dVXRxhEwArRx+kJNCBAgQIAAAQIECBAgQIAAAQIECBAoWUAAWjKo4ggQIECAAAECBAgQIECAAAECBAgQaBwBAWjj9IWaECBAgAABAgQIECBAgAABAgQIECBQsoAAtGRQxREgQIAAAQIECBAgQIAAAQIECBAg0DgCAtDG6Qs1IUCAAAECBAgQIECAAAECBAgQIECgZAGzwJcM2ra47t27p969e7fd3OVf18OkZ89/fjn36tUr9ejRo8sb5wD1sM7LLvtz3odRbjN+7zSTddl9V5Xy9OGK60nWK866W7du3ousOO7SrxT/H+ZLvMeJ/rQ0n0D+Hif6sGWfNl9Lum6N8z4Mgfhdw/9jzfu1ED9H69mHvseb92tDzastIACtc//Gm5z+/fvX+Sr1KX5WfYrNSh08ePD/Z+9OwOyoyoQBn6wkYUlCwg6yRkCGENYgA0QFAYcZHVRERVSQxUHFAfkBUYOyRFBkcwEVR0RBliBCxCAwwIMIimCGXWQT2QTJAgFC9p+vpJruzu2+3UlXp6vqPc/T6XtPnXuqzvvdm3v7u6fqFNb7yJEjC+u7qI7Lal2UR/Q7bNiw7Ken98G6p0Wr1V+R/zcVKVXk87qo4y6rdVEeRfYbf+TxLlK49/ou62fK3hPq+3saMWJE3z9IR9hUYMiQISl+lPIKDB06NMVPEWXOnDlFdKtPAgSWUUACdBkBmz183rx56eWXy/in6esJqGaDW4btzz///DI8uvFD41vY+FD5wgsvpEWLFjVu1Edry2ZdFOMKK6zQ8oXBK6+8kuKnpwvrnhZt3N/o0aMbb+jjtUX839QbQy7yeV3U8ZfVuiiPIvtdsGBBmjFjRpG70HeBAiuvvHLLH+kzZ85M8+fPL3Bvui5KIGYPrrrqqtlrMV6TSvkEYmJLPtHi1VdfLe3feOWT7/kjji8F586dW1gM47lSVHK15zX0SKA+AhKgBcd68eLFaeHChQXvpXzdF2GSJz3jdxH9l0/5n0dcJos8hnHkZXztlMm6rM/noo9bDIsWfrN/1m9a9MYt3r2hXMw+4v0wLxFHscw1yvU7PyVWDMsVt9ZH2/oSW2X8nNp6LHW/HfEr8m/G1pdLqLu18RPoSwJvXlSoLx2VYyFAgAABAgQIECBAgAABAgQIECBAgEAPCEiA9gCiLggQIECAAAECBAgQIECAAAECBAgQ6JsCToHvm3FxVAQIECBAgAABAgUIbH3X3QX0+s8up227VWF965gAAQIECBAgQGDpBcwAXXo7jyRAgAABAgQIECBAgAABAgQIECCwVAIXXnhh2m677dKkSZOW6vHL+qCHH354Wbto+viHHnooG+P73//+pm2LbCABWqSuvgkQIECAAAECBAgQIECAAAECBAg0EPj73/+e7rrrrvTEE0802FpcVSzK98UvfjFts802xe3kjZ5feeWVbIz33Xdf4fvqbAdOge9MxzYCBAgQIECAAAECBAgQIECAAAECBQiMGzcuHXbYYWmXXXYpoPeOu5w1a1Y69dRT0+DBgztuVLEtEqAVC6jhECBAgAABAgQIECBAgAABAgQI9H2BPfbYI8WPUryABGjxxvZAgAABAgQIECBAgAABAgQIECDQRwSmT5+e4vTztdZaK6266qrpqaeeSrfddlt67bXX0tZbb5223HLLJY70hRdeSM8991xaZ5110rBhw9KNN96YFi1alHbbbbe0wgortGn/zDPPpHvuuSf97W9/S5tsskkaO3ZsGj16dJs2cSfvM44hjqV9iZma//d//5fiWp0bbbRRihmjo0aNat+szf04pnhMnFoffW6//fZpjTXWaGnz7LPPpsceeyy7v3jx4nT//fdntzfffPPUv/+bV8qMfv7yl7+ku+++Oy1YsCDb92abbZYGDBjQ0lejG9HfH//4x7TVVltl427UZnnUSYAuD3X7JECAAAECBAgQIECAAAECBAgQWC4CF1xwQTr66KPT2WefnSX5vvvd77Y5jr333jtdccUVbRKbP/zhD9Pxxx+fzj333PSTn/wk/f73v88es9pqq6Unn3wyaztz5sz0la98JZ133nkprrOZl0gafuELX0gnnnhimz7/53/+Jx177LHp0EMPTd///vfz5tljY2GkaB/Jx7xEP1/60peyfQwcuGRK76ijjko//vGPUyRO8zJkyJAU9SeffHLq169fOumkk7IxxPb58+enf/mXf8mazp49O6200krZ7Uhi7r///lnyM6t4459oe9FFFzVMbEZ9HFvr65m+7W1vy/bXuo/ldXtJreV1JPZLgAABAgQIECBAgAABAgQIECBAoJcEIskYszo///nPp//8z//MZoWecsop6Zprrkmf+cxn0vnnn7/EkZxxxhnZjMxIBs6ZMydtu+22WVIzZlN+8IMfzGaGrr322umII45Im266aXrggQfSWWedlb7xjW+kO+64I9seicjOyoc+9KH0i1/8Iq2++upZ4jRmZ8Ys0HPOOSdLikaCcvLkyW26iOTjmWeemc32PO2009KYMWPS9ddfny6++OJslfmYpTpx4sS07777pnXXXTdLVkZC9Vvf+lbWTz6LNWZv7rrrrtls2A984AMpVm+PmaHXXnttlvjdYYcd0u9+97ts3PkBRDL4oIMOyhK3//3f/51d0/TRRx9NYfXRj340b7Zcf0uALld+OydAgAABAgQIECBAgAABAgQIEFgeApH8jORkJEDzErM/Y3X0H/3oR2nChAnpgAMOyDdlvyMRGTNHI8EZJU6bj/Kd73wnS25uvPHG2en0kbyMEonVT3ziE2n8+PHp5ptvTjH79MADD8y2NfonEo2R/FxzzTWzU9lbn74e/Wy33XbZ7NTrrruu5fqhl19+eZbkjMdMmzYte2z0vc8++6Sdd945m80ZCdiYhfrOd74zm8EZCdNIgLYeeyRxY1wxpq997WtZwjQ/xg9/+MPZpQFi5my0iSRolJg5GonSefPmpZ///Ocp2uUlkp9xSYF//OMfedVy+/3myf3L7RDsmAABAgQIECBAgAABAgQIECBAgEDvCkSyMmZ6ti4rr7xy+uIXv5hVXX311a03Zbfjupqf+9znWurjFPMocdp7lDhtPU9+ZhWv/xPXDf3qV7+a3c1nXObb2v/++te/nlVF+9bJz6iM63/mMypjdmVepk6dmt085phjWpKf+baPfOQj6b/+67/SZz/72TanxufbW/+OBG3M5ozjjQRp+xKn6sd443qp+SUA4lqjcc3TuOZn6+RnPDb6OfLII9t3s1zumwG6XNjtlAABAgQIECBAgAABAgQIECBAYHkK7L777qnRtTRj1mSUmE3ZvsRp7e1PYY/rdMbM0Ch77rln+4e0qY92cX3QmH3ZqMQp81FieyQk25d8saQ///nPLZti0aMocep6+xLH+r3vfa99dcP7+b432GCD9Nvf/rZhmzj1/84770yx/x133LHlOqG5WfsHxSr3ce3U5V0kQJd3BOyfAAECBAgQIECAAAECBAgQIECg1wXWX3/9hvtcb731svpYLf3ll19uWRwoKmM19vYlrncZCwrFIkIdrdIesyEHDRqUnSoeCwU16icWL4qV4aMccsgh7XfT5n4svDR37tzs+pz5Su5vectb2rTp7p08iRunt8ep8p2VRx55JNscq8RHieuKNiod1TdqW2SdBGiRuvomQIAAAQIECBAgQIAAAQIECBAolcCrr76aHW+c7j106NA2x54vFtS6MhKbUWImaFxHs/0M0XxbvqJ73j57UKt/4rF5OfXUU9PgwYPzu53+zlecjyTsspR8/5H8/I//+I9Ou4pre0YZNmxY9rujfXc01uxBvfiPBGgv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IECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FJEDrGXejJkCAAAECBAgQIECAAAECBAgQIFALAQnQWoTZIAkQIECAAAECBAgQIECAAAECBAjUU0ACtJ5xN2oCBAgQIECAAAECBAgQIECAAAECtRCQAK1FmA2SAAECBAgQIECAAAECBAgQIECAQD0FBtZz2EZNgACBnhfY+q67e77T13uctu1WhfSrUwIECBAgQIAAAQIECBAgUAcBM0DrEGVjJECAAAECBAgQIECAAAECBAgQIFBTAQnQmgbesAkQIECAAAECBAgQIECAAAECBAjUQUACtA5RNkYCBAgQIECAAAECBAgQIECAAAECNRWQAK1p4A2bAAECBAgQIECAAAECBAgQIECAQB0EJEDrEGVjJECAAAECBAgQIECAAAECBAgQIFBTAQnQmgbesAkQIECAAAECBAgQIECAAAECBAjUQUACtA5RNkYCBAgQIECAAAECBAgQIECAAAECNRWQAK1p4A2bAAECBAgQIECAAAECBAgQIECAQB0EJEDrEGVjJECAAAECBAgQIECAAAECBAgQIFBTAQnQmgbesAkQIECAAAECBAgQIECAAAECBAjUQUACtA5RNkYCBAgQIECAAAECBAgQIECAAAECNRWQAK1p4A2bAAECBAgQIECAAAECBAgQIECAQB0EJEDrEGVjJECAAAECBAgQIECAAAECBAgQIFBTAQnQmgbesAkQIECAAAECBAgQIECAAAECBAjUQUACtA5RNkYCBAgQIECAAAECBAgQIECAAAECNRWQAK1p4A2bAAECBAgQIECAAAECBAgQIECAQB0EJEDrEGVjJECAAAECBAgQIECAAAECBAgQIFBTAQnQmgbesAkQIECAAAECBAgQIECAAAECBAjUQUACtA5RNkYCBAgQIECAAAECBAgQIECAAAECNRWQAK1p4A2bAAECBAgQIECAAAECBAgQIECAQB0EJEDrEGVjJECAAAECBAgQIECAAAECBAgQIFBTAQnQmgbesAkQIECAAAECBAgQIECAAAECBAjUQUACtA5RNkYCBAgQIECAAAECBAgQIECAAAECNRWQAK1p4A2bAAECBAgQIECAAAECBAgQIECAQB0EBtZhkMZIgAABAgQIECBAgEDvCmx9192F7XDatlsV1reOCRAgQIAAgeoJmAFavZgaEQECBAgQIECAAAECBAgQIECAAAECbwhIgHoqECBAgAABAgQIECBAgAABAgQIECBQWQEJ0MqG1sAIECBAgAABAgQIECBAgAABAgQIEJAA9RwgQIAAAQIECBAgQIAAAQIECBAgQKCyAhKglQ2tgREgQIAAAQIECBAgQIAAAQIECBAgIAHqOUCAAAECBAgQIECAAAECBAgQIECAQGUFJEArG1oDI0CAAAECBAgQIECAAAECBAgQIEBAAtRzgAABAgQIECBAgAABAgQIECBAgACBygoMrNrIpk+fni677LL06KOPpueeey6tvvrqacMNN0z77bdfWm211RoOd+7cuWny5MnpzjvvTDNnzkxjxoxJ48aNS3vttVcaMGBAw8eoJECAAAECBAgQIECAAAECBAgQIECg7wtUKgF68803p0mTJqU5c+ZkictRo0alu+66K91xxx3p6quvTscee2zabbfd2kRl1qxZ6fDDD09PPvlkVr/qqquma6+9Nvu57bbb0gknnJAGDx7c5jHuECBAgAABAgQIECBAgAABAgQIECBQDoHKnAL/9NNPtyQ/DzzwwPSb3/wmXXHFFdnvj3/841lS9NRTT21JdObhOemkk7K68ePHp1/96lfpqquuSpdccknaeOON0y233JLOOeecvKnfBAgQIECAAAECBAgQIECAAAECBAiUTKAyCdApU6ZkSc7dd989HXTQQWmFFVbIQhG/DznkkPSOd7wjvfbaayna5eWBBx7IZocOHTo0nXzyyWn48OHZpnXWWSedccYZ2SzSqVOnptmzZ+cP8ZsAAQIECBAgQIAAAQIECBAgQIAAgRIJVCYBOm3atIx9l112acgfMzyjPPLIIy3b45T5KBMmTEhDhgzJbuf/xKnwO+ywQ5o3b16KJKhCgAABAgQIECBAgAABAgQIECBAgED5BCqTAI1T1S+99NK04447NozCjBkzsvoRI0a0bL///vuz23lytGXDGzciARrlnnvueaPGLwIECBAgQIAAAQIECBAgQIAAAQIEyiRQmUWQ4lT3tddeu6H9ggULWmZxbrHFFi1t4rqhUVonRVs2tqrPF0hqva317e9973sp9tGo7Lzzztmq8o221blupZVW6vHhDxz4z6fziiuumBYtWtTj/Ze1wyKsi7LIYxj9Dxo0KJXp2IsyiX7L6NCvX78iSQrru4zWhWEU3DHrgoFbdd+/f/9S/j/SagiluVnE8zreD/MSl23KL/OU19X1dxHWRVoOGDAg637YsGE+pxYJXWDfeQxjFz6nFgjdC13H+2IsdFzU/yOLFy/uhVHYBQEC3RWoTAK0s4F///vfT0899VSKa3vuvffeLU1feeWV7HZHCdBVVlkl2563a3lguxvnnXdemjt3brvaf95db7310jbbbNNwW1+vfLnAA1x55ZUL6z0SoGUrZbUu0jn+wCvij7wirYvyKPL1UtQxl7Xfslp7Xpf1Gdc7xx1/6JX1ud07Qj23l6Kdy/gZp+d02/ZUtHXbvfXcPTHsOcvl2VMkz+JHKa9AkTGcM2dOeWEcOYEKC1Q+ARorusdPfPg//vjjW671GTMEY1GkKB19gMq/EeoouVnh54WhESBAgAABAg0EXv7E/g1qe6Bq3wN6oBNdECBAgAABAgQIECDQSKDSCdAf/OAH6ac//WmW/Pzyl7+cxo4d22IQCdE4jSi+nekowZnXN/t27/TTT+/wFPhNNtkkzZw5s2W/Zbrx5glXPX/URZjkp6K8+OKLpTu1qGzWPf+M+GePeQzjXrw28y8penJ/RVr35HG27quI10vr/ou4PXz48CK6LbzPMloHiud14U+Nlh2U0XrhwoXppZdeahmDG8UJFPF/SJwynZ8REXGMeCqpdJ+v4/TpOLtMDMv77I1LNeUTZ+Izqll+5Y1lvBbnz59fWAzzXEN5hRw5gWoKVDIBGv+ZnXrqqem6667LTk2YOHFittJ7+xCOHj06xfU9Z8+e3X5Tdj+vb3aqyh577NHw8VEZH4SLSOJ0uMMe3FDkH3lFmOTXWonEddn+OCibdQ8+zTrsKmJYxPOkSOsOB7OMG4pwWMZDavrwsiZAy2gdwfC8bvqU7LEGZbSO98eyPrd7LHC91FERznnyM4Ywb9687I/2XhpOn95NEdZFDji/lmt8Tu1o7YAi96/vZRdoPSmmqM+py36UeuiKQCSy43VY1P8jrf/f7srxaEOAQO8IVGYV+JwrkpZf+MIXsuRnfLNz1llnNUx+RvtIgEbJE53ZnVb/5LMlRo4c2arWTQIECBAgQIAAAQIECBAgQIAAAQIEyiJQqQRoJCw/97nPpWnTpqV11103xeJEW265ZYexWH311bNtjz32WMM2ef3mm2/ecLtKAgQIECBAgAABAgQIECBAgAABAgT6tkBlEqBxetdxxx2XHn300bTppptmyc9Ygb2zsttuu2Wbb7jhhiWaxSJJN954Y1Y/bty4JbarIECAAAECBAgQIECAAAECBAgQIECg7wtUJgE6ZcqUdO+992antX/zm99MXbn+3I477pg22GCD9PDDD6epU6e2idZFF12Upk+fntZff/00fvz4NtvcIUCAAAECBAgQIECAAAECBAgQIECgHAKVWAQpLiYep7tHeeGFF9I+++zToX6syn7++edn2/v165cOOeSQFIskTZo0Kd1+++1pzJgxWSI1bsfFyo855pgU7RQCBAgQIECAAAECBAgQIECAAAECBMonUIkE6OOPP95mIaPOVgBvv+rirrvums4888wsAXrTTTel+IkSM0OPPPLINHbs2PJF1RETIECAAAECBAgQIECAAAECBAgQIJAJVCIButlmm6Xf/va3Sx3SrbfeOl1++eXZKe9PPvlkisWR1lxzzdS/f2WuELDUNh5IgAABAgQIECBAgAABAgQIECBAoMwClUiA9lQARo0aleJHIUCAAAECBAgQIECAAAECBAgQIECgGgKmOFYjjkZBgAABAgQIECBAgAABAgQIECBAgEADAQnQBiiqCBAgQIAAAQIECBAgQIAAAQIECBCohoAEaDXiaBQECBAgQIAAAQIECBAgQIAAAQIECDQQkABtgKKKAAECBAgQIECAAAECBAgQIECAAIFqCEiAViOORkGAAAECBAgQIECAAAECBAgQIECAQAMBCdAGKKoIECBAgAABAgQIECBAgAABAgQIEKiGgARoNeJoFAQIECBAgAABAgQIECBAgAABAgQINBCQAG2AoooAAQIECBAgQIAAAQIECBAgQIAAgWoISIBWI45GQYAAAQIECBAgQIAAAQIECBAgQIBAAwEJ0AYoqggQIECAAAECBAgQIECAAAECBAgQqIaABGg14mgUBAgQIECAAAECBAgQIECAAAECBAg0EJAAbYCiigABAgQIECBAgAABAgQIECBAgACBaghIgFYjjkZBgAABAgQIECBAgAABAgQIECBAgEADAQnQBiiqCBAgQIAAAQIECBAgQIAAAQIECBCohoAEaDXiaBQECBAgQIAAAQIECBAgQIAAAQIECDQQkABtgKKKAAECBAgQIECAAAECBAgQIECAAIFqCEiAViOORkGAAAECBAgQIECAAAECBAgQIECAQAMBCdAGKKoIECBAgAABAgQIECBAgAABAgQIEKiGgARoNeJoFAQIECBAgAABAgQIECBAgAABAgQINBCQAG2AoooAAQIECBAgQIAAAQIECBAgQIAAgWoISIBWI45GQYAAAQIECBAgQIAAAQIECBAgQIBAAwEJ0AYoqggQIECAAAECBAgQIECAAAECBAgQqIaABGg14mgUBAgQIECAAAECBAgQIECAAAECBAg0EJAAbYCiigABAgQIECBAgAABAgQIECBAgACBaghIgFYjjkZBgAABAgQIECBAgAABAgQIECBAgEADAQnQBiiqCBAgQIAAAQIECBAgQIAAAQIECBCohoAEaDXiaBQECBAgQIAAAQIECBAgQIAAAQIECDQQkABtgKKKAAECBAgQIECAAAECBAgQIECAAIFqCEiAViOORkGAAAECBAgQIECAAAECBAgQIECAQAMBCdAGKKoIECBAgAABAgQIECBAgAABAgQIEKiGgARoNeJoFAQIECBAgAABAgQIECBAgAABAgQINBCQAG2AoooAAQIECBAgQIAAAQIECBAgQIAAgWoISIBWI45GQYAAAQIECBAgQIAAAQIECBAgQIBAAwEJ0AYoqggQIECAAAECBAgQIECAAAECBAgQqIaABGg14mgUBAgQIECAAAECBAgQIECAAAECBAg0EJAAbYCiigABAgQIECBAgAABAgQIECBAgACBaghIgFYjjkZBgAABAgQIECBAgAABAgQIECBAiiRS0wAAOyVJREFUgEADAQnQBiiqCBAgQIAAAQIECBAgQIAAAQIECBCohoAEaDXiaBQECBAgQIAAAQIECBAgQIAAAQIECDQQkABtgKKKAAECBAgQIECAAAECBAgQIECAAIFqCEiAViOORkGAAAECBAgQIECAAAECBAgQIECAQAMBCdAGKKoIECBAgAABAgQIECBAgAABAgQIEKiGgARoNeJoFAQIECBAgAABAgQIECBAgAABAgQINBCQAG2AoooAAQIECBAgQIAAAQIECBAgQIAAgWoISIBWI45GQYAAAQIECBAgQIAAAQIECBAgQIBAAwEJ0AYoqggQIECAAAECBAgQIECAAAECBAgQqIaABGg14mgUBAgQIECAAAECBAgQIECAAAECBAg0EJAAbYCiigABAgQIECBAgAABAgQIECBAgACBaghIgFYjjkZBgAABAgQIECBAgAABAgQIECBAgEADgYEN6lQRIECAAAECBAgQWK4CKx93dDH73/eAYvrVKwECBAgQIECAQJ8VMAO0z4bGgREgQIAAAQIECBAgQIAAAQIECBAgsKwCEqDLKujxBAgQIECAAAECBAgQIECAAAECBAj0WQEJ0D4bGgdGgAABAgQIECBAgAABAgQIECBAgMCyCrgG6LIKejwBAgQIECBAgAABAgSaCKz5q2ubtFj6zdO23WrpH+yRBAgQIECgBgJmgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKiABWtfIGzcBAgQIECBAgAABAgQIECBAgACBGghIgNYgyIZIgAABAgQIECBAgAABAgQIECBAoK4CEqB1jbxxEyBAgAABAgQIECBAgAABAgQIEKiBgARoDYJsiAQIECBAgAABAgQIECBAgAABAgTqKjCwrgM3bgIECBAgQIAAAQIECBConsDWd91d2KD+/u97Fda3jgkQIECgOAEzQIuz1TMBAgQIECBAgAABAgQIECBAgAABAstZQAJ0OQfA7gkQIECAAAECBAgQIECAAAECBAgQKE5AArQ4Wz0TIECAAAECBAgQIECAAAECBAgQILCcBSRAl3MA7J4AAQIECBAgQIAAAQIECBAgQIAAgeIEJECLs9UzAQIECBAgQIAAAQIECBAgQIAAAQLLWUACdDkHwO4JECBAgAABAgQIECBAgAABAgQIEChOYGBxXes5BPr3759WWGEFGO0EijAZNGhQtpfBgwenRYsWtdtjfe8WYV2U5sCBb/6XNGDAAK+dN6DLFMOinhu91S/r3pJOXt+9R5369evHu5e8i/g/JN4P8xKfceKzpVK+/0Naf8YpIn5FPPeKOM6q9OlzarkjGe+LRcbQ/9Plfn44+uoKvJltqO4Yl+vIIik3fPjw5XoMS7vzl5f2gV143KqrrtqFVkvXZMSIEUv3wOX4qLJaF0k2dOjQFD89XYq07uljzfsr8vWS78PvfwqU1drzuveewWW0jj/yyvjcLqN10c6rrLJK7z3Z+/ieirbu48Nf4vB4LEFSaMWQIUNS/CjlFSjqb40QmTNnTnlhHDmBCgtIgBYc3Llz56bZs2cXvJdiul+xmG6zXp977rke7z1mRYwcOTL94x//KN0M0LJZ93jw3ugwZi/kCeyXX345vfLKKz2+qyKte/xg3+iwiNdLUcea97vaaqvlN0v1u4zWAex53XtPszJaL1iwIE2fPr33kHpoT2W0LuL/kEh65l8IzpgxI82fP7+HhMvdTRHWRYrkZyoVtY+yeRTl0Fv9vvrqq6X9G6+3jPryfkaPHp1ee+21FH9vFFHi9Z7/v11E//okQGDpBCRAl86tW49yOvaSXEWYLF68ONtR/C6i/yVHUY6aMlnkMcxly3Ts+TEX8ZtDEaqN+2Td2KWIWtZFqHbcJ++ObXpySxHOrd8bo/8i9tGTBr3VV9kcij7eovvvrbiWZT/+3ihLpBofZ8RPDBvbqCVQZQEXEapydI2NAAECBAgQIECAAAECBAgQIECAQM0FJEBr/gQwfAIECBAgQIAAAQIECBAgQIAAAQJVFpAArXJ0jY0AAQIECBAgQIAAAQIECBAgQIBAzQUkQGv+BDB8AgQIECBAgAABAgQIECBAgAABAlUWkACtcnSNjQABAgQIECBAgAABAgQIECBAgEDNBSRAa/4EMHwCBAgQIECAAAECBAgQIECAAAECVRaQAK1ydI2NAAECBAgQIECAAAECBAgQIECAQM0FJEBr/gQwfAIECBAgQIAAAQIECBAgQIAAAQJVFpAArXJ0jY0AAQIECBAgQIAAAQIECBAgQIBAzQUkQGv+BDB8AgQIECBAgAABAgQIECBAgAABAlUWkACtcnSNjQABAgQIECBAgAABAgQIECBAgEDNBSRAa/4EMHwCBAgQIECAAAECBAgQIECAAAECVRaQAK1ydI2NAAECBAgQIECAAAECBAgQIECAQM0FJEBr/gQwfAIECBAgQIAAAQIECBAgQIAAAQJVFpAArXJ0jY0AAQIECBAgQIAAAQIECBAgQIBAzQUkQGv+BDB8AgQIECBAgAABAgQIECBAgAABAlUWkACtcnSNjQABAgQIECBAgAABAgQIECBAgEDNBSRAa/4EMHwCBAgQIECAAAECBAgQIECAAAECVRaQAK1ydI2NAAECBAgQIECAAAECBAgQIECAQM0FJEBr/gQwfAIECBAgQIAAAQIECBAgQIAAAQJVFpAArXJ0jY0AAQIECBAgQIAAAQIECBAgQIBAzQUG1nz8hk+AAAECBAgQIECg1gIrH3d0MePf94Bi+tUrAQIECBAgQKCbAmaAdhNMcwIECBAgQIAAAQIECBAgQIAAAQIEyiMgAVqeWDlSAgQIECBAgAABAgQIECBAgAABAgS6KSAB2k0wzQkQIECAAAECBAgQIECAAAECBAgQKI+ABGh5YuVICRAgQIAAAQIECBAgQIAAAQIECBDopoAEaDfBNCdAgAABAgQIECBAgAABAgQIECBAoDwCEqDliZUjJUCAAAECBAgQIECAAAECBAgQIECgmwISoN0E05wAAQIECBAgQIAAAQIECBAgQIAAgfIISICWJ1aOlAABAgQIECBAgAABAgQIECBAgACBbgpIgHYTTHMCBAgQIECAAAECBAgQIECAAAECBMojIAFanlg5UgIECBAgQIAAAQIECBAgQIAAAQIEuikgAdpNMM0JECBAgAABAgQIECBAgAABAgQIECiPgARoeWLlSAkQIECAAAECBAgQIECAAAECBAgQ6KaABGg3wTQnQIAAAQIECBAgQIAAAQIECBAgQKA8AhKg5YmVIyVAgAABAgQIECBAgAABAgQIECBAoJsCEqDdBNOcAAECBAgQIECAAAECBAgQIECAAIHyCEiAlidWjpQAAQIECBAgQIAAAQIECBAgQIAAgW4KSIB2E0xzAgQIECBAgAABAgQIECBAgAABAgTKIyABWp5YOVICBAgQIECAAAECBAgQIECAAAECBLopIAHaTTDNCRAgQIAAAQIECBAgQIAAAQIECBAoj4AEaHli5UgJECBAgAABAgQIECBAgAABAgQIEOimgARoN8E0J0CAAAECBAgQIECAAAECBAgQIECgPAISoOWJlSMlQIAAAQIECBAgQIAAAQIECBAgQKCbAhKg3QTTnAABAgQIECBAgAABAgQIECBAgACB8ghIgJYnVo6UAAECBAgQIECAAAECBAgQIECAAIFuCkiAdhNMcwIECBAgQIAAAQIECBAgQIAAAQIEyiMgAVqeWDlSAgQIECBAgAABAgQIECBAgAABAgS6KSAB2k0wzQkQIECAAAECBAgQIECAAAECBAgQKI+ABGh5YuVICRAgQIAAAQIECBAgQIAAAQIECBDopoAEaDfBNCdAgAABAgQIECBAgAABAgQIECBAoDwCEqDliZUjJUCAAAECBAgQIECAAAECBAgQIECgmwISoN0E05wAAQIECBAgQIAAAQIECBAgQIAAgfIISICWJ1aOlAABAgQIECBAgAABAgQIECBAgACBbgpIgHYTTHMCBAgQIECAAAECBAgQIECAAAECBMojMLA8h+pICRAgQIAAAQIECBAgQIAAgb4isPVddxd2KNO23aqwvnVMgED9BMwArV/MjZgAAQIECBAgQIAAAQIECBAgQIBAbQQkQGsTagMlQIAAAQIECBAgQIAAAQIECBAgUD8BCdD6xdyICRAgQIAAAQIECBAgQIAAAQIECNRGQAK0NqE2UAIECBAgQIAAAQIECBAgQIAAAQL1E7AIUv1ibsQ1EyjqwuQuSl6zJ5LhEiBAgAABAgQIECBAgACBkgqYAVrSwDlsAgQIECBAgAABAgQIECBAgAABAgSaC0iANjfSggABAgQIECBAgAABAgQIECBAgACBkgpIgJY0cA6bAAECBAgQIECAAAECBAgQIECAAIHmAhKgzY20IECAAAECBAgQIECAAAECBAgQIECgpAISoCUNnMMmQIAAAQIECBAgQIAAAQIECBAgQKC5gARocyMtCBAgQIAAAQIECBAgQIAAAQIECBAoqYAEaEkD57AJECBAgAABAgQIECBAgAABAgQIEGguIAHa3EgLAgQIECBAgAABAgQIECBAgAABAgRKKiABWtLAOWwCBAgQIECAAAECBAgQIECAAAECBJoLSIA2N9KCAAECBAgQIECAAAECBAgQIECAAIGSCkiAljRwDpsAAQIECBAgQIAAAQIECBAgQIAAgeYCEqDNjbQgQIAAAQIECBAgQIAAAQIECBAgQKCkAhKgJQ2cwyZAgAABAgQIECBAgAABAgQIECBAoLmABGhzIy0IECBAgAABAgQIECBAgAABAgQIECipgARoSQPnsAkQIECAAAECBAgQIECAAAECBAgQaC4gAdrcSAsCBAgQIECAAAECBAgQIECAAAECBEoqIAFa0sA5bAIECBAgQIAAAQIECBAgQIAAAQIEmgtIgDY30oIAAQIECBAgQIAAAQIECBAgQIAAgZIKSICWNHAOmwABAgQIECBAgAABAgQIECBAgACB5gISoM2NtCBAgAABAgQIECBAgAABAgQIECBAoKQCEqAlDZzDJkCAAAECBAgQIECAAAECBAgQIECguYAEaHMjLQgQIECAAAECBAgQIECAAAECBAgQKKmABGhJA+ewCRAgQIAAAQIECBAgQIAAAQIECBBoLiAB2txICwIECBAgQIAAAQIECBAgQIAAAQIESiogAVrSwDlsAgQIECBAgAABAgQIECBAgAABAgSaC0iANjfSggABAgQIECBAgAABAgQIECBAgACBkgpIgJY0cA6bAAECBAgQIECAAAECBAgQIECAAIHmAhKgzY20IECAAAECBAgQIECAAAECBAgQIECgpAISoCUNnMMmQIAAAQIECBAgQIAAAQIECBAgQKC5gARocyMtCBAgQIAAAQIECBAgQIAAAQIECBAoqYAEaEkD57AJECBAgAABAgQIECBAgAABAgQIEGguIAHa3EgLAgQIECBAgAABAgQIECBAgAABAgRKKiABWtLAOWwCBAgQIECAAAECBAgQIECAAAECBJoLSIA2N9KCAAECBAgQIECAAAECBAgQIECAAIGSCkiAljRwDpsAAQIECBAgQIAAAQIECBAgQIAAgeYCEqDNjbQgQIAAAQIECBAgQIAAAQIECBAgQKCkAhKgJQ2cwyZAgAABAgQIECBAgAABAgQIECBAoLmABGhzIy0IECBAgAABAgQIECBAgAABAgQIECipgARoSQPnsAkQIECAAAECBAgQIECAAAECBAgQaC4gAdrcSAsCBAgQIECAAAECBAgQIECAAAECBEoqIAFa0sA5bAIECBAgQIAAAQIECBAgQIAAAQIEmgtIgDY30oIAAQIECBAgQIAAAQIECBAgQIAAgZIKSICWNHAOmwABAgQIECBAgAABAgQIECBAgACB5gISoM2NtCBAgAABAgQIECBAgAABAgQIECBAoKQCEqAlDZzDJkCAAAECBAgQIECAAAECBAgQIECguYAEaHMjLQgQIECAAAECBAgQIECAAAECBAgQKKmABGhJA+ewCRAgQIAAAQIECBAgQIAAAQIECBBoLiAB2txICwIECBAgQIAAAQIECBAgQIAAAQIESiogAVrSwDlsAgQIECBAgAABAgQIECBAgAABAgSaCwxs3qT6LebOnZsmT56c7rzzzjRz5sw0ZsyYNG7cuLTXXnulAQMGVB/ACAkQIECAAAECBAgQIECAAAECBAhUVKD2CdBZs2alww8/PD355JNZiFddddV07bXXZj+33XZbOuGEE9LgwYMrGn7DIkCAAAECBAgQIECAAAECBAgQIFBtgdqfAn/SSSdlyc/x48enX/3qV+mqq65Kl1xySdp4443TLbfcks4555xqPwOMjgABAgQIECBAgAABAgQIECBAgECFBWqdAH3ggQfSHXfckYYOHZpOPvnkNHz48CzU66yzTjrjjDOy09+nTp2aZs+eXeGngKERIECAAAECBAgQIECAAAECBAgQqK5ArROgN998cxbZCRMmpCFDhrSJcpwKv8MOO6R58+alSIIqBAgQIECAAAECBAgQIECAAAECBAiUT6DWCdD7778/i1ic/t6oRAI0yj333NNoszoCBAgQIECAAAECBAgQIECAAAECBPq4QK0XQXr66aez8IwYMaJhmPL6fIGkho1erzz99NPTggULGm7efffd02abbdZwW50rV1555R4f/oABA7I+V1pppbRo0aIe71+HbQWKjGHsKRYfK2IfbUdRjntldOjXr185cNsdZRmt2w2hNHdZ916o+vfv7//TXuIu4nk9aNCglqMfNmyYzzhvaBRh3QJdwI38c2oBXWddls2jKIfe6tfn1N6RLup5He+LRcZw8eLFvQNkLwQIdEug1gnQV155JcPKE53t5VZZZZWsKm/Xfnt+/8ILL0xz587N77b5vemmm6btttuuTV1Z7rxc4IFGkrKoEn8clK0UaV2URZExjGOODyXx09OFdU+LVqu/op/XRWl5Xhclu2S/ZbSOP/TK+Nwuo3XRzkV9xmG95Gu9bDVFP/fK5lH08cYXE62/nCh6f3Xtv8jndVF/a0Ss5syZU9eQGTeBPi3Q7/VvJ2r59UTMEIxrf0aZPHlyWmONNZYIVCySdNhhh6VIkE6ZMmWJ7XnF2LFjO0yAnnrqqWmfffbJm/pNgAABAgQIECBAgAABAgQIVFQgEqCx0LJCgEDfEqjtDNCYDRH/KcV/Th3N3szrm81C++53v9vh6UjrrrtumjFjRt+KekWPJr6FjdMkZs2a1WE8Kjr0ygwrj2EMKF6bvj0tb2jji6PXXnst+ynvKOp95CNHjkxxKYOFCxemF198sd4YJR59zBiM/1vFsLxBXHHFFdMKK6yQDeCll17q8LJL5R1hPY48ToEfPnx49lqM/1eV8gkMHDgw5WcIxmecV199tXyDcMSZQLwWY7Hjov7WyHMNuAkQ6FsCtU2ARhhGjx6d4vqes2fPbhiVvD4+eHZWdtlllw43z5w5UwKgQ51iNsSbmQ+WxdgW3Wvr60ZGDPMvIYrer/6LERDDYlx7u9c4UcRrsbfVe25/kTiLP9rFsOdMe7unIUOGtOwyPuPMnz+/5b4b5RHIT5eOGHa0dkB5RlPPI2194qTPOOV+DkQsi4xh/qVVuZUcPYHqCdR6FfhIgEbJE53twxvfskeJWTAKAQIECBAgQIAAAQIECBAgQIAAAQLlE6h1AnT11VfPIvbYY481jFxev/nmmzfcrpIAAQIECBAgQIAAAQIECBAgQIAAgb4tUOsE6G677ZZF54YbblgiSrFI0o033pjVjxs3bontKggQIECAAAECBAgQIECAAAECBAgQ6PsCtU6A7rjjjmmDDTZIDz/8cJo6dWqbaF100UVp+vTpaf3110/jx49vs80dAgQIECBAgAABAgQIECBAgAABAgTKIVDrRZBiwZVDDjkkTZw4MU2aNCndfvvtacyYMenee+/NbsfFyo855phsFdxyhNNREiBAgAABAgQIECBAgAABAgQIECDQWqDWM0ADYtddd01nnnlmWnPNNdNNN92UfvCDH2TJz5gZevrpp6exY8e29nKbAAECBAgQIECAAAECBAgQIECAAIESCdR6Bmgep6233jpdfvnl2SnvTz75ZIrFkSIh2r9/7fPDOZHfBAgQIECAAAECBAgQIECAAAECBEopIAHaKmyjRo1K8aMQIECAAAECBAgQIECAAAECBAgQIFANAVMcqxFHoyBAgAABAgQIECBAgAABAgQIECBAoIGABGgDFFUECBAgQIAAAQIECBAgQIAAAQIECFRDQAK0GnE0CgIECBAgQIAAAQIECBAgQIAAAQIEGghIgDZAUUWAAAECBAgQIECAAAECBAgQIECAQDUEJECrEUejIECAAAECBAgQIECAAAECBAgQIECggYAEaAMUVQQIECBAgAABAgQIECBAgAABAgQIVENAArQacTQKAgQIECBAgAABAgQIECBAgAABAgQaCEiANkBRRYAAAQIECBAgQIAAAQIECBAgQIBANQQkQKsRR6MgQIAAAQIECBAgQIAAAQIECBAgQKCBgARoAxRVBAgQIECAAAECBAgQIECAAAECBAhUQ0ACtBpxNAoCBAgQIECAAAECBAgQIECAAAECBBoISIA2QFFFgAABAgQIECBAgAABAgQIECBAgEA1BCRAqxFHoyBAgAABAgQIECBAgAABAgQIECBAoIGABGgDFFUECBAgQIAAAQIECBAgQIAAAQIECFRDQAK0GnE0CgIECBAgQIAAAQIECBAgQIAAAQIEGghIgDZAUUWAAAECBAgQIECAAAECBAgQIECAQDUEJECrEUejIECAAAECBAgQIECAAAECBAgQIECggYAEaAMUVQQIECBAgAABAgQIECBAgAABAgQIVENAArQacTQKAgQIECBAgAABAgQIECBAgAABAgQaCEiANkBRRYAAAQIECBAgQIAAAQIECBAgQIBANQQkQKsRR6MgQIAAAQIECBAgQIAAAQIECBAgQKCBgARoAxRVBAgQIECAAAECBAgQIECAAAECBAhUQ0ACtBpxNAoCBAgQIECAAAECBAgQIECAAAECBBoISIA2QFFFgAABAgQIECBAgAABAgQIECBAgEA1BCRAqxFHoyBAgAABAgQIECBAgAABAgQIECBAoIGABGgDFFUECBAgQIAAAQIECBAgQIAAAQIECFRDQAK0GnE0CgIECBAgQIAAAQIECBAgQIAAAQIEGghIgDZAUUWAAAECBAgQIECAAAECBAgQIECAQDUE+i1+vVRjKEZBgEDZBa6//vp05JFHZsM49NBD0xFHHFH2ITl+AqUV2GabbdK8efPS2muvna677rrSjsOBEyi7wFe+8pV05ZVXZsP42c9+lsaNG1f2ITl+AqUU+MMf/pA+9alPZcf+4Q9/OH35y18u5TgcNAECBOoqMLCuAzduAgT6nsCiRYvS/PnzswNbsGBB3ztAR0SgRgKR/IzXY/6arNHQDZVAnxJYuHBhy+sw3icVAgSWj0Drz6nxulQIECBAoFwCToEvV7wcLQECBAgQIECAAAECBAgQIECAAAEC3RCQAO0GlqYECBAgQIAAAQIECBAgQIAAAQIECJRLQAK0XPFytAQIECBAgAABAgQIECBAgAABAgQIdENAArQbWJoSIECAAAECBAgQIECAAAECBAgQIFAuAQnQcsXL0RIgQIAAAQIECBAgQIAAAQIECBAg0A2BfotfL91orykBAgQKE3juuefStGnTsv433njjNGbMmML2pWMCBDoXuP7661Oscjt06NA0YcKEzhvbSoBAYQL33Xdfeuqpp7L+x48fn0aOHFnYvnRMgEDHAjNmzEh33HFH1uAtb3lLetvb3tZxY1sIECBAoM8JSID2uZA4IAIECBAgQIAAAQIECBAgQIAAAQIEekrAKfA9JakfAgQIECBAgAABAgQIECBAgAABAgT6nIAEaJ8LiQMiQIAAAQIECBAgQIAAAQIECBAgQKCnBCRAe0pSPwQIECBAgAABAgQIECBAgAABAgQI9DkBCdA+FxIHRIAAAQIECBAgQIAAAQIECBAgQIBATwkM7KmO9EOAAIFcIFasve666/K7nf7eaaed0o477thpm9j46quvph/+8Iedttt7773TJpts0mkbGwnUSeCaa65JF154YfrqV7+aNt988w6H/uc//zldfvnl6Yknnkgrrrhi2nLLLdO73vWutNFGG3X4mI42zJ07N02ePDndeeedaebMmWnMmDFp3Lhxaa+99koDBgzo6GHqCVRaoKuvxenTp6fLLrssPfroo+m5555Lq6++etpwww3Tfvvtl1ZbbbVuGd10003pnnvu6fAxo0ePTvvvv3+H220gUEWB+Dx59NFHpzXXXDNNnDix4RCL+MzZk++zDQ9aJQECBAg0FZAAbUqkAQEC3RX461//mq688souPWzUqFFdSoA+8sgjWVKls04jySIB2pmQbXUSuPfee9Ppp5+eFixYkCIp2VGJZOXZZ5+dbV5ppZXSvHnz0p/+9KcsCXPqqaembbbZpqOHLlE/a9asdPjhh6cnn3wy27bqqquma6+9Nvu57bbb0gknnJAGDx68xONUEKiyQFdfizfffHOaNGlSmjNnTvZlQbw/3nXXXemOO+5IV199dTr22GPTbrvt1mWqq666Knt8Rw+ILzgkQDvSUV9FgcWLF6cTTzwxxWty4MCO/wzu6c+cPfk+W8W4GBMBAgR6S6Dj//l76wjshwCBygmMHTs2/b//9/86HFfMSPnNb36Thg0blt7xjnd02K71hocffji7u+2222Yz01pvy2/HTDOFAIGUpk2bliUbI/nZWYk/As8555wsKRnJyV122SVLmP7yl7/M6mOWzMUXX5zNlOmsn3zbSSedlCU/x48fn77yla+k4cOHp6effjp96UtfSrfccktLn3l7vwlUXaCrr8V4neTJzwMPPDBLTK6wwgrZlxcxizt+4guJt771rWm99dbrElv+vnnEEUek6Kt9WXnlldtXuU+gsgLxxUJ82fe73/2u6Rjz105PfObs6ffZpgevAQECBAh0KCAB2iGNDQQILK3AW97ylhQ/jUqc3vejH/0o2xQJkvXXX79RsyXq8g+jcVrue9/73iW2qyBA4J+Xivje976XYuZXlP79+6dFixZ1SPOTn/wkxYyYj33sY2nXXXfN2g0aNCjtu+++6ZlnnslmXUcy9NOf/nSHfeQbHnjggWym2tChQ9PJJ5+chgwZkm1aZ5110hlnnJHe//73p6lTp6bDDjssSbzkan5XVSBOoe3Oa3HKlCnZzM/dd989HXTQQS0skbg85JBD0t/+9rcUM0SjXcyyblaef/759NJLL6WYRRqvZ4VAnQXikiynnXZa+vvf/970fTGcevIzZ0++z9Y5hsZOgACBnhCwCFJPKOqDAIEuC8QMsRkzZqR99tkn7bzzzl1+XJyOFGXTTTft8mM0JFA3gYMPPjhLfsbs6ri2WVw7sKMSCZo4tTbKnnvuuUSzvO5Xv/pVNit0iQbtKiI5E2XChAktyc+s4vV/4lT4HXbYITu9PpKgCoGqC3TntRgWMVM0SszCblRiVnWU/L2wUZvWdXkCx3tmaxW36yhw/fXXpyOPPDJLfsbr6KijjmrKkL/OlvX109Pvs00PXAMCBAgQ6FRAArRTHhsJEOhJgfgQGtczi2TIoYce2uWu4zTexx57LLteU74oS8xs6ey6hl3uXEMCFRKIa3BG4vKCCy5I7373uzsd2YMPPpjN/ozTaddee+0l2m622WbZTM0XX3wxm322RIN2Fffff39Wkydq2m3OEqBR19miLO0f4z6Bsgp057UYY4xLUVx66aUdXhM7vjiMMmLEiOx3s3/aJ0DjfTTvo9ljbSdQJYF43sd73HHHHZe++c1vppEjR3Y6vJ78zNnT77OdHriNBAgQINBUwCnwTYk0IECgJwTi2kvf/e53s67i9L1YbKWrJVamnj9/fnZafVyPMBZYilPp4/TeONX+gAMOSHvssUdXu9OOQGUFfvzjH6c11lijS+OLaw5G6SyhEttmz56dXdcz//Kho86b9ZfvJ18gqaN+1BOogkB3Xosx3jjVvdEXEbEtEjL5zOktttgiqpqWPAEaj43ZbzHDdOHChdmXGttvv32K64LG6fEKgaoLxMJhH/jABzpd9Ki1QU9+5mz2vhj77c77bOvjdJsAAQIEui9gBmj3zTyCAIGlEIjZn5G0jD+4urOKbewq/0MuroF2/vnnp379+qW4MH18i//X11ecj9PqY3EIhUDdBbqa/AynV155JePKE5ON7FZZZZWsOm/bqE1el7fpqL/u9JX36TeBsgp057XYbIzf//7301NPPZXierp77713s+bZ9vwU3p/+9KdZ8jNmdMfpvK+99lq68cYb08c//vEun07fpR1qRKCPCowePbrLyc8YQk9+5mz2vhj7894YCgoBAgR6R8AM0N5xthcCtRe4+uqrM4P3ve993fogGg/K/5CLhVNildxx48ZlfcXiLTEb9KyzzkrXXHNNdupgV1eVzzrwD4EaC8S1yaJ0tiBRPlM7kiadlVhoKW/TUX95Xy5d0ZmkbQTaClxyySUpfuKMh+OPP36J6+u2bf3Pe5F0iUXMosQlMb7whS+kWJwsStR/7WtfS7FoWbyf/uAHP+j2e3LWkX8IVFSgJz9z9uT7bEW5DYsAAQK9KiAB2qvcdkagngLxbfpDDz2UBgwYsFQruH/iE5/IZo0OHz68zSmCMRM0VpaOWaCRCP3Zz36WJEDr+Rwz6u4LrLjiitmD5s2b1+GD82RlnJ7bWYnkTCRY4lIX+WPat8/rBw8e3H6T+wQINBCI5GTM4IzX15e//OU0duzYBq2WrIrXYlwu5oUXXsi+MIz3yrzEafYnnnhi2n///bOZbrEQ2k477ZRv9ptA7QV68jNnT77P1j4wAAgQINADAk6B7wFEXRAg0LnAlClTsgaxOvTSXHMsZpRtvvnmbZKfrfe4++67Z3cff/zxFDPRFAIEmgvEaYFRYkGxjkpc/zNK/kdcR+2iPu8vf0z7tnl9V/pq/1j3CdRJIK55HZd2ieRnfGEQCctmi5q19omEaSxutvXWW2eXjGm9LW7H6flbbrllVh0LDCoECLwp0JOfOfP3xZ56n33zKN0iQIAAgaURkABdGjWPIUCgywJxWux1112XtY/ZmkWU/FprMZMtv95SEfvRJ4EqCeR/mOWJyUZjy/9oa7Zqbjy2WX/d6avRsagjUAeBeD3GKevxvhnXBoxLvMSXhz1dVl999azLuDa3QoBA1wW685mz2fti7NV7Y9fttSRAgMCyCjgFflkFPZ4AgU4F4vS6SEqutdZaaauttuq0bUcbL7/88mwBpb322ittsMEGSzR77rnnsrr4Y7Gj6w8u8SAVBGoukCdAYlX2mHE2aNCgNiIvvvhimjFjRnb67ZgxY9psa3Qn7y9mlO24445LNMlnmsVsboUAgSUFIhESq7M/+uijad11103f+MY3spmcS7bsvCYuOfP73/8+DRs2LO27774NGz///PNZfexHIUDgTYGe/MyZvy/21Pvsm0fpFgECBAgsjYAZoEuj5jEECHRZ4L777svadiWB0lGnN9xwQ7rooovST37yk4ZNbr311qx+iy22aLhdJQECSwrEtQBjZeiXX345/eEPf1iiwU033ZQWLlyYtYlESrOy2267ZU3i9dq+xKUpYuXpKPkiZu3buE+gzgKxqN9xxx2XJT9jtfbzzjtvqZKfYThz5sx0/vnnp29/+9vpiSeeWII1vti4//77s3rvm0vwqKi5QE9+5uzp99mah8bwCRAgsMwCEqDLTKgDAgQ6E4iVZqNsuOGGnTXLtj377LPp+uuvT//7v//bpu073/nO7H4kUGJmTOty1113pfi2PsqnPvWp1pvcJkCgicBHPvKRrMWPf/zj1PpU+Jgd9vOf/zzb1n4GWUev05j1GTO0Y9GzqVOnttlzfIERp9quv/76afz48W22uUOAQEpxrex77703u5TEN7/5zRSL/nWl/O53v8veN+Ma2HmJLxlGjBiRIql6wQUXpAULFuSbUlyW5rTTTssWLPvXf/3X7AuOlo1uECCQlvYzZ6PXYnAuzfusMBAgQIBAMQJOgS/GVa8ECLwh8PTTT2e3Ntpoo6Ymd999dzrllFOy1eLz2WTxoEjA3HbbbWnatGnpk5/8ZNp2222zn0ceeSTFLLUohx9+eIpZMwoBAl0XiGsLxinpDz74YDr44IOzP/wiWRIzYCJhGQmSd73rXW067Oh1GitNH3LIIWnixIlp0qRJ6fbbb08x8zuSOnE7TrE/5phjGi7K0mYH7hComcDcuXOzGZ8x7Fi5fZ999ulQYJNNNslmd+YNzj777BRfSsRrL/+icciQIemEE05IRx11VPZaji8K99hjj+y99ZZbbklPPfVU1vbII4/Mu/GbAIE3BJb2M2ej12J0uTTvs4JBgAABAsUISIAW46pXAgReF4jTXuNUvCj5H2bZnW7+M2DAgOxaaDGL7OKLL07xx1z8RImVbuOaaY2uOdjN3WhOoHYC8dqK02TPPPPMbNGVeI1FifoPfvCD6bDDDsuuAdpVmF133TXrKxKg8eVE/gVFzAyNZMvYsWO72pV2BGojELM3W8/AjktPdFRaz+bsqE3Ub7fdduncc89NkZSJLzguvfTSrPnQoUPTnnvumSVHu3Jpi872YRuBKgr09GfOnn6fraK5MREgQKC3BPq9fnrM4t7amf0QIEBgWQXij79nnnkmzZo1K8Ws0pVWWmlZu/R4AgReF4jXVlxiIj4WxBcLK6644jK5xAzSWPghFoFYc801u5VIXaYdezABAm0EYkGzeC3GIoHx2u7f3xWw2gC5Q6ADgZ7+zNnT77MdHLZqAgQIEOhAQAK0AxjVBAgQIECAAAECBAgQIECAAAECBAiUX8BXwOWPoREQIECAAAECBAgQIECAAAECBAgQINCBgARoBzCqCRAgQIAAAQIECBAgQIAAAQIECBAov4AEaPljaAQECBAgQIAAAQIECBAgQIAAAQIECHQgIAHaAYxqAgQIECBAgAABAgQIECBAgAABAgTKLyABWv4YGgEBAgQIECBAgAABAgQIECBAgAABAh0ISIB2AKOaAAECBAgQIECAAAECBAgQIECAAIHyC0iAlj+GRkCAAAECBAgQIECAAAECBAgQIECAQAcCEqAdwKgmQIAAAQIECBAgQIAAAQIECBAgQKD8AhKg5Y+hERAgQIAAAQIECBAgQIAAAQIECBAg0IGABGgHMKoJECBAgAABAgQIECBAgAABAgQIECi/gARo+WNoBAQIECBAgAABAgQIECBAgAABAgQIdCAgAdoBjGoCBAgQIECAQC6w0UYbpX79+qX3ve99eVX2+7Of/WxW37//0n2kmjVrVvb46HvixIlt+naHAAECBAgQIECAAIGeEVi6T+s9s2+9ECBAgAABAgQqIbB48eJKjMMgCBAgQIAAAQIECFRRQAK0ilE1JgIECBAgQIAAAQIECBAgQIAAAQIEMgEJUE8EAgQIECBAgAABAgQIECBAgAABAgQqKyABWtnQGhgBAgQIECBAgAABAgQIECBAgAABAgMRECBAgAABAgQIpPTiiy+ma665Jj3yyCNp1KhRaeedd05bbrll6s4CR4sWLUr33ntvuv3229OMGTPS9ttvn/2MGDGiy8Rz5sxJ06ZNS3/84x/T/Pnz00477ZS23XbbtMIKK3S5Dw0JECBAgAABAgQIEHhTQAL0TQu3CBAgQIAAgZoKnHLKKem0005Ls2fPbiMwduzYLCnaprKDOzNnzkwf+tCH0g033NCmxYABA1L0f8wxx2QrvrfZ2O7Ob3/727TffvulZ599ts2WwYMHp3PPPTcddNBBberdIUCAAAECBAgQIECguYAEaHMjLQgQIECAAIEKCxx66KHphz/8YTbClVZaKb397W9Pq666ajaL85577knjx49Pr776alOBaPfwww+nNddcM+2yyy5p7ty56dZbb81mgh533HHpD3/4Q5o8eXKHM0p/8YtfpK9//etpwYIFKWaMRn/PPPNMNqN03rx56ZBDDslmge6///5Nj0UDAgQIECBAgAABAgTeFHAN0Dct3CJAgAABAgRqJjB16tSW5Gecrv7nP/85XXfddemSSy5Jf/3rX9OXvvSlLAk5a9aspjKR/DzggAPS448/ni677LJ01VVXZafTv/vd784ee+WVV2b1HXV0//33ZwnOX//61+mFF15I1157bYoEbNwfNGhQitPrjzjiiOx3R32oJ0CAAAECBAgQIEBgSQEJ0CVN1BAgQIAAAQI1EZg0aVI20iFDhqQpU6akddZZp2Xk/fr1SyeffHL62Mc+1lLX2Y24VueFF16Yoq+8jBw5Mv3yl79Mq6++elb11a9+NS1cuDDf3OZ3XGs0Eqfvec97Upw2n5e4f/DBB2d347qikShVCBAgQIAAAQIECBDouoAEaNettCRAgAABAgQqJBAzKmOxoSgHHnhgWmONNRqO7vjjj2967c544Iknntjw8cOGDUtHHnlktu2hhx5Kd911V8N2Y8aMSf/2b//WcFskV/Py/PPP5zf9JkCAAAECBAgQIECgCwISoF1A0oQAAQIECBConsCjjz6aXnnllWxgrROM7Ue6+eabp7XXXrt9dZv7MWNzwoQJbepa3/nXf/3XlruRBG1UNt1000bVWd1aa63Vsu2ll15que0GAQIECBAgQIAAAQLNBSRAmxtpQYAAAQIECFRQ4O67724Z1brrrttyu9GN9dZbr1F1S12cOj9wYMdrS7Z+fFwrtFF561vf2qg6q4vT4/OyePHi/KbfBAgQIECAAAECBAh0QeDNT9NdaKwJAQIECBAgQKAqAjEDNC+x6ntnJVZ276yMGjWqs83ZqvJ5g9mzZ+c32/weOnRom/vuECBAgAABAgQIECDQMwISoD3jqBcCBAgQIECgZAKtk5bPPPNMp0c/ffr0TrfH4kSdldb9b7zxxp01tY0AAQIECBAgQIAAgR4WkADtYVDdESBAgAABAuUQaH1a+hNPPNHpQTfbHgsTdbS6e3T817/+taX/TTbZpOW2GwQIECBAgAABAgQIFC8gAVq8sT0QIECAAAECfVBgl112SbFCe5TJkyd3eIQPPPBAevLJJzvcHhvmzJmTbrzxxg7bXHrppdm2fv36pVhUSSFAgAABAgQIECBAoPcEJEB7z9qeCBAgQIAAgT4kEMnP/fbbLzuiG264ocME5sSJE1NXFh468cQTs0Ro+yE++OCD6ac//WlWvf/++6f111+/fRP3CRAgQIAAAQIECBAoUEACtEBcXRMgQIAAAQJ9W+Db3/522nrrrbODfO9735tipmZ+KvuLL76YPvGJT6QrrriiS4O49dZb0/vf//70+OOPZ+0jaXr99denXXfdNeszFjmaNGlSl/rSiAABAgQIECBAgACBnhMY2HNd6YkAAQIECBAgUC6BFVdcMU2ZMiXtsMMOKRYq+vCHP5xWXnnltM4666S//OUvadGiRWn77bdPgwf///buGDWBKAgD8AQlWHsDbTyApRewsfIIezDPYCUIlpZaewFPYCUWSZ4QSKMQcBiEz8aFhX/2feXPrH7Gfr9/eLjJZBLj8Tg2m839ezQaRStQf/8cqWWuVqv4+7ujD8PcIECAAAECBAgQIEDgpQI2QF/KKYwAAQIECBB4N4FWdm6325jP5/dHv1wucTqdot/vx2KxiN1uF8Ph8Omx2nbner2OruuiXbct0FZ+9nq9mM1mcTgcYrlcPs1wkwABAgQIECBAgACBHIGPn9ezvnKipRIgQIAAAQIE3kvgfD7fy8+28dk2PweDwb8PcLvd4ng8xvV6jel0Gm3L1IcAAQIECBAgQIAAgToBBWidvckECBAgQIAAAQIECBAgQIAAAQIECCQLeAU+GVg8AQIECBAgQIAAAQIECBAgQIAAAQJ1AgrQOnuTCRAgQIAAAQIECBAgQIAAAQIECBBIFlCAJgOLJ0CAAAECBAgQIECAAAECBAgQIECgTkABWmdvMgECBAgQIECAAAECBAgQIECAAAECyQIK0GRg8QQIECBAgAABAgQIECBAgAABAgQI1AkoQOvsTSZAgAABAgQIECBAgAABAgQIECBAIFlAAZoMLJ4AAQIECBAgQIAAAQIECBAgQIAAgToBBWidvckECBAgQIAAAQIECBAgQIAAAQIECCQLKECTgcUTIECAAAECBAgQIECAAAECBAgQIFAnoACtszeZAAECBAgQIECAAAECBAgQIECAAIFkAQVoMrB4AgQIECBAgAABAgQIECBAgAABAgTqBBSgdfYmEyBAgAABAgQIECBAgAABAgQIECCQLKAATQYWT4AAAQIECBAgQIAAAQIECBAgQIBAnYACtM7eZAIECBAgQIAAAQIECBAgQIAAAQIEkgUUoMnA4gkQIECAAAECBAgQIECAAAECBAgQqBNQgNbZm0yAAAECBAgQIECAAAECBAgQIECAQLKAAjQZWDwBAgQIECBAgAABAgQIECBAgAABAnUCCtA6e5MJECBAgAABAgQIECBAgAABAgQIEEgW+AYV8VAU47+kWwAAAABJRU5ErkJggg==)
