import pandas as pd
from biogeme.models import ordered_logit
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable, Elem, log
import biogeme.database as db
import os

# Assume that the data is loaded into a Pandas DataFrame named 'df'.
# You need to replace 'your_dataframe.csv' with your actual data file.

# Load your data
df = pd.read_csv('input/data/2015/persons_cleaned.csv')

database = db.Database('persons_cleaned', df)

globals().update(database.variables)

# Define variables using the Biogeme Variable class
age = Variable('age')
#interv_lang = Variable('interv_lang')
#day_week = Variable('day_week')
dl_car = Variable('dl_car')
dl_moto = Variable('dl_moto')
park_work_car = Variable('park_work_car')
#use_park_work = Variable('use_park_work')
#park_work_cost = Variable('park_work_cost')
park_work_moto = Variable('park_work_moto')
park_work_bike = Variable('park_work_bike')
#work_home = Variable('work_home')
#work_home_perc = Variable('work_home_perc')
height = Variable('height')
weight = Variable('weight')
is_bike_avail = Variable('is_bike_avail')
is_moped_avail = Variable('is_moped_avail')
is_small_moped_avail = Variable('is_small_moped_avail')
is_moto_avail = Variable('is_moto_avail')
is_car_avail = Variable('is_car_avail')
rdist_sum = Variable('rdist_sum')
rdist_sum_Inland = Variable('rdist_sum_Inland')
#travel_time_sum = Variable('travel_time_sum')
#travel_time_sum_inland = Variable('travel_time_sum_inland')
n_stops = Variable('n_stops')
n_trips = Variable('n_trips')
travel_times = Variable('travel_times')
dist_softmodes = Variable('dist_softmodes')
dist_car = Variable('dist_car')
dist_pt = Variable('dist_pt')
dist_other = Variable('dist_other')
n_work_trips = Variable('n_work_trips')
n_edu_trips = Variable('n_edu_trips')
n_shopping_trips = Variable('n_shopping_trips')
n_business_trips = Variable('n_business_trips')
n_leisure_trips = Variable('n_leisure_trips')
n_service_trips = Variable('n_service_trips')
job_x_CH1903 = Variable('job_x_CH1903')
job_y_CH1903 = Variable('job_y_CH1903')
#business_sector_agriculture = Variable('business_sector_agriculture')
#business_sector_industry = Variable('business_sector_industry')
#business_sector_services = Variable('business_sector_services')
work_sched_fully_fixed = Variable('work_sched_fully_fixed')
work_sched_fixed_daily_blocks = Variable('work_sched_fixed_daily_blocks')
work_sched_fixed_weekly_monthly = Variable('work_sched_fixed_weekly_monthly')
work_sched_fully_flex = Variable('work_sched_fully_flex')
#ptsub_ga = Variable('ptsub_ga')
ptsub_regional = Variable('ptsub_regional')
ptsub_parcours = Variable('ptsub_parcours')
ptsub_other = Variable('ptsub_other')
#ptsub_junior = Variable('ptsub_junior')
ptsub_ht = Variable('ptsub_ht')
ptsub_track_sev = Variable('ptsub_track_sev')
ptsub_ga_first = Variable('ptsub_ga_first')
ptsub_ga_second = Variable('ptsub_ga_second')
#ptsub_parcours_first = Variable('ptsub_parcours_first')
#ptsub_parcours_second = Variable('ptsub_parcours_second')
#ptsub_regional_first = Variable('ptsub_regional_first')
#ptsub_regional_second = Variable('ptsub_regional_second')
#pt_emp_subsidies = Variable('pt_emp_subsidies')
can_walk_alone = Variable('can_walk_alone')
#use_wheelchair = Variable('use_wheelchair')
#car_sharing_subscription = Variable('car_sharing_subscription')
single = Variable('single')
married = Variable('married')
widowed = Variable('widowed')
divorced = Variable('divorced')
#unmarried = Variable('unmarried')
registered_partnership = Variable('registered_partnership')
#dissolved_partnership = Variable('dissolved_partnership')
male = Variable('male')
self_employed_with_employees = Variable('self_employed_with_employees')
self_employed_without_employees = Variable('self_employed_without_employees')
employed_by_family = Variable('employed_by_family')
employee_management = Variable('employee_management')
employee_with_supervision = Variable('employee_with_supervision')
employee_without_supervision = Variable('employee_without_supervision')
#trainee = Variable('trainee')
#unemployed = Variable('unemployed')
#student = Variable('student')
#retired = Variable('retired')
#invalid = Variable('invalid')
#stay_at_home_parent = Variable('stay_at_home_parent')
#inactive = Variable('inactive')
compulsory_education = Variable('compulsory_education')
secondary_education = Variable('secondary_education')
tertiary_education = Variable('tertiary_education')
full_time = Variable('full_time')
part_time = Variable('part_time')
swiss = Variable('swiss')
#job_abroad = Variable('job_abroad')
job_location_city = Variable('job_location_city')
job_location_popsize_500000_plus = Variable('job_location_popsize_500000_plus')
job_location_popsize_200000_500000 = Variable('job_location_popsize_200000_500000')
job_location_popsize_100000_200000 = Variable('job_location_popsize_100000_200000')
job_location_popsize_50000_100000 = Variable('job_location_popsize_50000_100000')
job_location_popsize_50000_minus = Variable('job_location_popsize_50000_minus')
job_location_popsize_no_agglo = Variable('job_location_popsize_no_agglo')
hh_x_ch1903 = Variable('hh_x_ch1903')
hh_y_ch1903 = Variable('hh_y_ch1903')
hh_size = Variable('hh_size')
n_cars_hh = Variable('n_cars_hh')
#n_car_park_hh = Variable('n_car_park_hh')
n_moto_hh = Variable('n_moto_hh')
n_s_moto_hh = Variable('n_s_moto_hh')
n_mopeds_hh = Variable('n_mopeds_hh')
n_bike_hh = Variable('n_bike_hh')
n_ebike_s_hh = Variable('n_ebike_s_hh')
n_ebike_f_hh = Variable('n_ebike_f_hh')
hh_location_city = Variable('hh_location_city')
hh_location_popsize_500000_plus = Variable('hh_location_popsize_500000_plus')
hh_location_popsize_200000_500000 = Variable('hh_location_popsize_200000_500000')
hh_location_popsize_50000_100000 = Variable('hh_location_popsize_50000_100000')
hh_location_popsize_100000_200000 = Variable('hh_location_popsize_100000_200000')
hh_location_popsize_50000_minus = Variable('hh_location_popsize_50000_minus')
hh_location_popsize_no_agglo = Variable('hh_location_popsize_no_agglo')
pt_access_A = Variable('pt_access_A')
pt_access_B = Variable('pt_access_B')
pt_access_C = Variable('pt_access_C')
pt_access_D = Variable('pt_access_D')
single_household = Variable('single_household')
couple_without_children = Variable('couple_without_children')
couple_with_children = Variable('couple_with_children')
single_parent = Variable('single_parent')
non_family_household = Variable('non_family_household')
hh_income_2000_minus = Variable('hh_income_2000_minus')
hh_income_2000_4000 = Variable('hh_income_2000_4000')
hh_income_4000_6000 = Variable('hh_income_4000_6000')
hh_income_6000_8000 = Variable('hh_income_6000_8000')
hh_income_8000_10000 = Variable('hh_income_8000_10000')
hh_income_10000_12000 = Variable('hh_income_10000_12000')
hh_income_12000_14000 = Variable('hh_income_12000_14000')
hh_income_14000_16000 = Variable('hh_income_14000_16000')
hh_income_16000_plus = Variable('hh_income_16000_plus')
work_home_crow_dist = Variable('work_home_crow_dist')
car_net_distance = Variable('car_net_distance')
work_accessibility_car = Variable('accsib_car')
work_accessibility_pt = Variable('accsib_pt')
work_accessibility_mul = Variable('accsib_mul')


# Define the alternative-specific constant (ASC) and coefficients
ASC_1 = Beta('ASC_1', 0, None, None, 1)
ASC_2 = 0  # Normalized to 0

#thresholds
tau_1 = Beta('tau_1', -2, None, 0, 0) # first threshold

# positive differences between thresholds
diff_12 = Beta('diff_12', 1, 0, None, 0) # difference between thresholds
diff_23 = Beta('diff_23', 1, 0, None, 0) # difference between thresholds
diff_34 = Beta('diff_34', 1, 0, None, 0) # difference between thresholds
diff_45 = Beta('diff_45', 1, 0, None, 0) # difference between thresholds

# other thresholds
tau_2 = tau_1 + diff_12
tau_3 = tau_2 + diff_23
tau_4 = tau_3 + diff_34
tau_5 = tau_4 + diff_45


# define utility parameters
Beta_age = Beta('Beta_age', 0, None, None, 0)
Beta_dl_car = Beta('Beta_dl_car', 0, None, None, 0)
Beta_dl_moto = Beta('Beta_dl_moto', 0, None, None, 0)
Beta_park_work_car = Beta('Beta_park_work_car', 0, None, None, 0)
Beta_park_work_moto = Beta('Beta_park_work_moto', 0, None, None, 0)
Beta_park_work_bike = Beta('Beta_park_work_bike', 0, None, None, 0)
Beta_height = Beta('Beta_height', 0, None, None, 0)
Beta_weight = Beta('Beta_weight', 0, None, None, 0)
Beta_is_bike_avail = Beta('Beta_is_bike_avail', 0, None, None, 0)
Beta_is_moped_avail = Beta('Beta_is_moped_avail', 0, None, None, 0)
Beta_is_small_moped_avail = Beta('Beta_is_small_moped_avail', 0, None, None, 0)
Beta_is_moto_avail = Beta('Beta_is_moto_avail', 0, None, None, 0)
Beta_is_car_avail = Beta('Beta_is_car_avail', 0, None, None, 0)
Beta_rdist_sum = Beta('Beta_rdist_sum', 0, None, None, 0)
Beta_rdist_sum_Inland = Beta('Beta_rdist_sum_Inland', 0, None, None, 0)
Beta_n_stops = Beta('Beta_n_stops', 0, None, None, 0)
Beta_n_trips = Beta('Beta_n_trips', 0, None, None, 0)
Beta_travel_times = Beta('Beta_travel_times', 0, None, None, 0)
Beta_dist_softmodes = Beta('Beta_dist_softmodes', 0, None, None, 0)
Beta_dist_car = Beta('Beta_dist_car', 0, None, None, 0)
Beta_dist_pt = Beta('Beta_dist_pt', 0, None, None, 0)
Beta_dist_other = Beta('Beta_dist_other', 0, None, None, 0)
Beta_n_work_trips = Beta('Beta_n_work_trips', 0, None, None, 0)
Beta_n_edu_trips = Beta('Beta_n_edu_trips', 0, None, None, 0)
Beta_n_shopping_trips = Beta('Beta_n_shopping_trips', 0, None, None, 0)
Beta_n_business_trips = Beta('Beta_n_business_trips', 0, None, None, 0)
Beta_n_leisure_trips = Beta('Beta_n_leisure_trips', 0, None, None, 0)
Beta_n_service_trips = Beta('Beta_n_service_trips', 0, None, None, 0)
Beta_job_x_CH1903 = Beta('Beta_job_x_CH1903', 0, None, None, 0)
Beta_job_y_CH1903 = Beta('Beta_job_y_CH1903', 0, None, None, 0)
Beta_work_sched_fully_fixed = Beta('Beta_work_sched_fully_fixed', 0, None, None, 0)
Beta_work_sched_fixed_daily_blocks = Beta('Beta_work_sched_fixed_daily_blocks', 0, None, None, 1) # normalised to 0
Beta_work_sched_fixed_weekly_monthly = Beta('Beta_work_sched_fixed_weekly_monthly', 0, None, None, 0)
Beta_work_sched_fully_flex = Beta('Beta_work_sched_fully_flex', 0, None, None, 0)
Beta_ptsub_regional = Beta('Beta_ptsub_regional', 0, None, None, 0)
Beta_ptsub_parcours = Beta('Beta_ptsub_parcours', 0, None, None, 0)
Beta_ptsub_other = Beta('Beta_ptsub_other', 0, None, None, 0)
Beta_ptsub_ht = Beta('Beta_ptsub_ht', 0, None, None, 0)
Beta_ptsub_track_sev = Beta('Beta_ptsub_track_sev', 0, None, None, 0)
Beta_ptsub_ga_first = Beta('Beta_ptsub_ga_first', 0, None, None, 0)
Beta_ptsub_ga_second = Beta('Beta_ptsub_ga_second', 0, None, None, 0)
Beta_can_walk_alone = Beta('Beta_can_walk_alone', 0, None, None, 0)
Beta_single = Beta('Beta_single', 0, None, None, 0)
Beta_married = Beta('Beta_married', 0, None, None, 0)
Beta_widowed = Beta('Beta_widowed', 0, None, None, 0)
Beta_divorced = Beta('Beta_divorced', 0, None, None, 0)
Beta_registered_partnership = Beta('Beta_registered_partnership', 0, None, None, 0)
#Beta_dissolved_partnership = Beta('Beta_dissolved_partnership', 0, None, None, 1) # normalised to 0
Beta_male = Beta('Beta_male', 0, None, None, 0)
Beta_self_employed_with_employees = Beta('Beta_self_employed_with_employees', 0, None, None, 0)
Beta_self_employed_without_employees = Beta('Beta_self_employed_without_employees', 0, None, None, 0)
Beta_employed_by_family = Beta('Beta_employed_by_family', 0, None, None, 0)
Beta_employee_management = Beta('Beta_employee_management', 0, None, None, 0)
Beta_employee_with_supervision = Beta('Beta_employee_with_supervision', 0, None, None, 0)
Beta_employee_without_supervision = Beta('Beta_employee_without_supervision', 0, None, None, 1) # normalised to 0
Beta_compulsory_education = Beta('Beta_compulsory_education', 0, None, None, 1) # normalised to 0
Beta_secondary_education = Beta('Beta_secondary_education', 0, None, None, 0)
Beta_tertiary_education = Beta('Beta_tertiary_education', 0, None, None, 0)
Beta_full_time = Beta('Beta_full_time', 0, None, None, 0)
Beta_part_time = Beta('Beta_part_time', 0, None, None, 0)
Beta_swiss = Beta('Beta_swiss', 0, None, None, 0)
Beta_job_abroad = Beta('Beta_job_abroad', 0, None, None, 0)
Beta_job_location_city = Beta('Beta_job_location_city', 0, None, None, 0)
Beta_job_location_popsize_500000_plus = Beta('Beta_job_location_popsize_500000_plus', 0, None, None, 0)
Beta_job_location_popsize_200000_500000 = Beta('Beta_job_location_popsize_200000_500000', 0, None, None, 0)
Beta_job_location_popsize_100000_200000 = Beta('Beta_job_location_popsize_100000_200000', 0, None, None, 0)
Beta_job_location_popsize_50000_100000 = Beta('Beta_job_location_popsize_50000_100000', 0, None, None, 0)
Beta_job_location_popsize_50000_minus = Beta('Beta_job_location_popsize_50000_minus', 0, None, None, 1) # normalised to 0
Beta_job_location_popsize_no_agglo = Beta('Beta_job_location_popsize_no_agglo', 0, None, None, 0)
Beta_hh_x_ch1903 = Beta('Beta_hh_x_ch1903', 0, None, None, 0)
Beta_hh_y_ch1903 = Beta('Beta_hh_y_ch1903', 0, None, None, 0)
Beta_hh_size = Beta('Beta_hh_size', 0, None, None, 0)
Beta_n_cars_hh = Beta('Beta_n_cars_hh', 0, None, None, 0)
Beta_n_moto_hh = Beta('Beta_n_moto_hh', 0, None, None, 0)
Beta_n_s_moto_hh = Beta('Beta_n_s_moto_hh', 0, None, None, 0)
Beta_n_mopeds_hh = Beta('Beta_n_mopeds_hh', 0, None, None, 0)
Beta_n_bike_hh = Beta('Beta_n_bike_hh', 0, None, None, 0)
Beta_n_ebike_s_hh = Beta('Beta_n_ebike_s_hh', 0, None, None, 0)
Beta_n_ebike_f_hh = Beta('Beta_n_ebike_f_hh', 0, None, None, 0)
Beta_hh_location_city = Beta('Beta_hh_location_city', 0, None, None, 0)
Beta_hh_location_popsize_500000_plus = Beta('Beta_hh_location_popsize_500000_plus', 0, None, None, 0)
Beta_hh_location_popsize_200000_500000 = Beta('Beta_hh_location_popsize_200000_500000', 0, None, None, 0)
Beta_hh_location_popsize_100000_200000 = Beta('Beta_hh_location_popsize_100000_200000', 0, None, None, 0)
Beta_hh_location_popsize_50000_100000 = Beta('Beta_hh_location_popsize_50000_100000', 0, None, None, 0)
Beta_hh_location_popsize_50000_minus = Beta('Beta_hh_location_popsize_50000_minus', 0, None, None, 1) # normalised to 0
Beta_hh_location_popsize_no_agglo = Beta('Beta_hh_location_popsize_no_agglo', 0, None, None, 0)
Beta_pt_access_A = Beta('Beta_pt_access_A', 0, None, None, 0)
Beta_pt_access_B = Beta('Beta_pt_access_B', 0, None, None, 0)
Beta_pt_access_C = Beta('Beta_pt_access_C', 0, None, None, 0)
Beta_pt_access_D = Beta('Beta_pt_access_D', 0, None, None, 0)
Beta_single_household = Beta('Beta_single_household', 0, None, None, 0)
Beta_couple_without_children = Beta('Beta_couple_without_children', 0, None, None, 0)
Beta_couple_with_children = Beta('Beta_couple_with_children', 0, None, None, 0)
Beta_single_parent = Beta('Beta_single_parent', 0, None, None, 0)
Beta_non_family_household = Beta('Beta_non_family_household', 0, None, None, 1) # normalised to 0
Beta_hh_income_2000_minus = Beta('Beta_hh_income_2000_minus', 0, None, None, 1) # normalised to 0
Beta_hh_income_2000_4000 = Beta('Beta_hh_income_2000_4000', 0, None, None, 0)
Beta_hh_income_4000_6000 = Beta('Beta_hh_income_4000_6000', 0, None, None, 0)
Beta_hh_income_6000_8000 = Beta('Beta_hh_income_6000_8000', 0, None, None, 0)
Beta_hh_income_8000_10000 = Beta('Beta_hh_income_8000_10000', 0, None, None, 0)
Beta_hh_income_10000_12000 = Beta('Beta_hh_income_10000_12000', 0, None, None, 0)
Beta_hh_income_12000_14000 = Beta('Beta_hh_income_12000_14000', 0, None, None, 0)
Beta_hh_income_14000_16000 = Beta('Beta_hh_income_14000_16000', 0, None, None, 0)
Beta_hh_income_16000_plus = Beta('Beta_hh_income_16000_plus', 0, None, None, 0)
Beta_home_work_dist = Beta('Beta_home_work_dist', 0, None, None, 0)
Beta_car_net_distance = Beta('Beta_car_net_distance', 0, None, None, 0)
Beta_work_accessibility_car = Beta('Beta_work_accessibility_car', 0, None, None, 0)
Beta_work_accessibility_pt = Beta('Beta_work_accessibility_pt', 0, None, None, 0)
Beta_work_accessibility_mul = Beta('Beta_work_accessibility_mul', 0, None, None, 0)


# Utility functions
V = (
    Beta_age * age +
    Beta_dl_car * dl_car +
    Beta_dl_moto * dl_moto +
    Beta_park_work_car * park_work_car +
    Beta_park_work_moto * park_work_moto +
    Beta_park_work_bike * park_work_bike +
    Beta_height * height +
    Beta_weight * weight +
    Beta_is_bike_avail * is_bike_avail +
    Beta_is_moped_avail * is_moped_avail +
    Beta_is_small_moped_avail * is_small_moped_avail +
    Beta_is_moto_avail * is_moto_avail +
    Beta_is_car_avail * is_car_avail +
    Beta_rdist_sum * rdist_sum +
    Beta_rdist_sum_Inland * rdist_sum_Inland +
    Beta_n_stops * n_stops +
    Beta_n_trips * n_trips +
    Beta_travel_times * travel_times +
    Beta_dist_softmodes * dist_softmodes +
    Beta_dist_car * dist_car +
    Beta_dist_pt * dist_pt +
    Beta_dist_other * dist_other +
    Beta_n_work_trips * n_work_trips +
    Beta_n_edu_trips * n_edu_trips +
    Beta_n_shopping_trips * n_shopping_trips +
    Beta_n_business_trips * n_business_trips +
    Beta_n_leisure_trips * n_leisure_trips +
    Beta_n_service_trips * n_service_trips +
    #Beta_job_abroad * job_abroad +
    Beta_job_x_CH1903 * job_x_CH1903 +
    Beta_job_y_CH1903 * job_y_CH1903 +
    Beta_work_sched_fully_fixed * work_sched_fully_fixed +
    Beta_work_sched_fixed_daily_blocks * work_sched_fixed_daily_blocks +
    Beta_work_sched_fixed_weekly_monthly * work_sched_fixed_weekly_monthly +
    Beta_work_sched_fully_flex * work_sched_fully_flex +
    Beta_ptsub_regional * ptsub_regional +
    Beta_ptsub_parcours * ptsub_parcours +
    Beta_ptsub_other * ptsub_other +
    Beta_ptsub_ht * ptsub_ht +
    Beta_ptsub_track_sev * ptsub_track_sev +
    Beta_ptsub_ga_first * ptsub_ga_first +
    Beta_ptsub_ga_second * ptsub_ga_second +
    Beta_can_walk_alone * can_walk_alone +
    Beta_single * single +
    Beta_married * married +
    Beta_widowed * widowed +
    Beta_divorced * divorced +
    Beta_registered_partnership * registered_partnership +
    #Beta_dissolved_partnership * dissolved_partnership +
    Beta_male * male +
    Beta_self_employed_with_employees * self_employed_with_employees +
    Beta_self_employed_without_employees * self_employed_without_employees +
    Beta_employed_by_family * employed_by_family +
    Beta_employee_management * employee_management +
    Beta_employee_with_supervision * employee_with_supervision +
    Beta_employee_without_supervision * employee_without_supervision +
    Beta_compulsory_education * compulsory_education +
    Beta_secondary_education * secondary_education +
    Beta_tertiary_education * tertiary_education +
    Beta_full_time * full_time +
    Beta_part_time * part_time +
    Beta_swiss * swiss +
    Beta_job_location_city * job_location_city +
    Beta_job_location_popsize_500000_plus * job_location_popsize_500000_plus +
    Beta_job_location_popsize_200000_500000 * job_location_popsize_200000_500000 +
    Beta_job_location_popsize_100000_200000 * job_location_popsize_100000_200000 +
    Beta_job_location_popsize_50000_100000 * job_location_popsize_50000_100000 +
    Beta_job_location_popsize_50000_minus * job_location_popsize_50000_minus +
    Beta_job_location_popsize_no_agglo * job_location_popsize_no_agglo +
    Beta_hh_x_ch1903 * hh_x_ch1903 +
    Beta_hh_y_ch1903 * hh_y_ch1903 +
    Beta_hh_size * hh_size +
    Beta_n_cars_hh * n_cars_hh +
    Beta_n_moto_hh * n_moto_hh +
    Beta_n_s_moto_hh * n_s_moto_hh +
    Beta_n_mopeds_hh * n_mopeds_hh +
    Beta_n_bike_hh * n_bike_hh +
    Beta_n_ebike_s_hh * n_ebike_s_hh +
    Beta_n_ebike_f_hh * n_ebike_f_hh +
    Beta_hh_location_city * hh_location_city +
    Beta_hh_location_popsize_500000_plus * hh_location_popsize_500000_plus +
    Beta_hh_location_popsize_200000_500000 * hh_location_popsize_200000_500000 +
    Beta_hh_location_popsize_100000_200000 * hh_location_popsize_100000_200000 +
    Beta_hh_location_popsize_50000_100000 * hh_location_popsize_50000_100000 +
    Beta_hh_location_popsize_50000_minus * hh_location_popsize_50000_minus +
    Beta_hh_location_popsize_no_agglo * hh_location_popsize_no_agglo +
    Beta_pt_access_A * pt_access_A +
    Beta_pt_access_B * pt_access_B +
    Beta_pt_access_C * pt_access_C +
    Beta_pt_access_D * pt_access_D +
    Beta_single_household * single_household +
    Beta_couple_without_children * couple_without_children +
    Beta_couple_with_children * couple_with_children +
    Beta_single_parent * single_parent +
    Beta_non_family_household * non_family_household +
    Beta_hh_income_2000_minus * hh_income_2000_minus +
    Beta_hh_income_2000_4000 * hh_income_2000_4000 +
    Beta_hh_income_4000_6000 * hh_income_4000_6000 +
    Beta_hh_income_6000_8000 * hh_income_6000_8000 +
    Beta_hh_income_8000_10000 * hh_income_8000_10000 +
    Beta_hh_income_10000_12000 * hh_income_10000_12000 +
    Beta_hh_income_12000_14000 * hh_income_12000_14000 +
    Beta_hh_income_14000_16000 * hh_income_14000_16000 +
    Beta_hh_income_16000_plus * hh_income_16000_plus +
    Beta_home_work_dist * work_home_crow_dist + 
    Beta_car_net_distance * car_net_distance +
    Beta_work_accessibility_car * work_accessibility_car +
    Beta_work_accessibility_pt * work_accessibility_pt +
    Beta_work_accessibility_mul * work_accessibility_mul
)

CHOICE = Variable('work_home_days')

the_proba = ordered_logit(
    continuous_value=V,
    list_of_discrete_values=[0, 1, 2, 3, 4, 5],
    tau_parameter=tau_1,
)

the_chosen_proba = Elem(the_proba, CHOICE)

logprob = log(the_chosen_proba)

current_dir = os.getcwd()

os.chdir('output/data/')

# Create the biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = "intensity_teleworking_ordinal_logit"

# Estimate the parameters
results = biogeme.estimate()

# Print the results
print(results.getEstimatedParameters())

os.chdir(current_dir)
