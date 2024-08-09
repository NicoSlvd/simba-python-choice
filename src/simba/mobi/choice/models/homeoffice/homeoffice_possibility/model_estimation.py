import pandas as pd
from biogeme import models
from biogeme.biogeme import bioLogLogit
from biogeme.expressions import Beta, Variable

# Assume that the data is loaded into a Pandas DataFrame named 'df'.
# You need to replace 'your_dataframe.csv' with your actual data file.

# Load your data
df = pd.read_csv('your_dataframe.csv')

# Define variables using the Biogeme Variable class
age = Variable('age')
interv_lang = Variable('interv_lang')
day_week = Variable('day_week')
dl_car = Variable('dl_car')
dl_moto = Variable('dl_moto')
park_work_car = Variable('park_work_car')
use_park_work = Variable('use_park_work')
park_work_cost = Variable('park_work_cost')
park_work_moto = Variable('park_work_moto')
park_work_bike = Variable('park_work_bike')
work_home = Variable('work_home')
work_home_perc = Variable('work_home_perc')
height = Variable('height')
weight = Variable('weight')
bike_avail = Variable('bike_avail')
moped_avail = Variable('moped_avail')
small_moped_avail = Variable('small_moped_avail')
moto_avail = Variable('moto_avail')
car_avail = Variable('car_avail')
rdist_sum = Variable('rdist_sum')
rdist_sum_Inland = Variable('rdist_sum_Inland')
travel_time_sum = Variable('travel_time_sum')
travel_time_sum_inland = Variable('travel_time_sum_inland')
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
business_sector_agriculture = Variable('business_sector_agriculture')
business_sector_industry = Variable('business_sector_industry')
business_sector_services = Variable('business_sector_services')
fixed_work_sched = Variable('fixed_work_sched')
semi_flexible_work_sched = Variable('semi_flexible_work_sched')
ptsub_ga = Variable('ptsub_ga')
ptsub_regional = Variable('ptsub_regional')
ptsub_parcours = Variable('ptsub_parcours')
ptsub_other = Variable('ptsub_other')
ptsub_junior = Variable('ptsub_junior')
ptsub_ht = Variable('ptsub_ht')
ptsub_track_sev = Variable('ptsub_track_sev')
ptsub_ga_first = Variable('ptsub_ga_first')
ptsub_ga_second = Variable('ptsub_ga_second')
ptsub_parcours_first = Variable('ptsub_parcours_first')
ptsub_parcours_second = Variable('ptsub_parcours_second')
ptsub_regional_first = Variable('ptsub_regional_first')
ptsub_regional_second = Variable('ptsub_regional_second')
pt_emp_subsidies = Variable('pt_emp_subsidies')
can_walk_alone = Variable('can_walk_alone')
use_wheelchair = Variable('use_wheelchair')
car_sharing_subscription = Variable('car_sharing_subscription')
single = Variable('single')
married = Variable('married')
widowed = Variable('widowed')
divorced = Variable('divorced')
unmarried = Variable('unmarried')
registered_partnership = Variable('registered_partnership')
dissolved_partnership = Variable('dissolved_partnership')
male = Variable('male')
self_employed_with_employees = Variable('self_employed_with_employees')
self_employed_without_employees = Variable('self_employed_without_employees')
employed_by_family = Variable('employed_by_family')
employee_management = Variable('employee_management')
trainee = Variable('trainee')
unemployed = Variable('unemployed')
student = Variable('student')
retired = Variable('retired')
invalid = Variable('invalid')
stay_at_home_parent = Variable('stay_at_home_parent')
inactive = Variable('inactive')
compulsory_education = Variable('compulsory_education')
secondary_education = Variable('secondary_education')
tertiary_education = Variable('tertiary_education')
full_time = Variable('full_time')
part_time = Variable('part_time')
swiss = Variable('swiss')
job_location_city = Variable('job_location_city')
job_location_popsize_500000 = Variable('job_location_popsize_500000')
job_location_popsize_250000 = Variable('job_location_popsize_250000')
job_location_popsize_100000 = Variable('job_location_popsize_100000')
job_location_popsize_50000 = Variable('job_location_popsize_50000')
job_location_popsize_1000 = Variable('job_location_popsize_1000')
job_location_popsize_no_agglo = Variable('job_location_popsize_no_agglo')
hh_x_ch1903 = Variable('hh_x_ch1903')
hh_y_ch1903 = Variable('hh_y_ch1903')
hh_size = Variable('hh_size')
n_cars_hh = Variable('n_cars_hh')
n_car_park_hh = Variable('n_car_park_hh')
n_moto_hh = Variable('n_moto_hh')
n_s_moto_hh = Variable('n_s_moto_hh')
n_mopeds_hh = Variable('n_mopeds_hh')
n_bike_hh = Variable('n_bike_hh')
n_ebike_s_hh = Variable('n_ebike_s_hh')
n_ebike_f_hh = Variable('n_ebike_f_hh')
hh_location_city = Variable('hh_location_city')
hh_location_popsize_500000 = Variable('hh_location_popsize_500000')
hh_location_popsize_250000 = Variable('hh_location_popsize_250000')
hh_location_popsize_100000 = Variable('hh_location_popsize_100000')
hh_location_popsize_50000 = Variable('hh_location_popsize_50000')
hh_location_popsize_1000 = Variable('hh_location_popsize_1000')
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
hh_income_l2000 = Variable('hh_income_l2000')
hh_income_2000_4000 = Variable('hh_income_2000_4000')
hh_income_4000_6000 = Variable('hh_income_4000_6000')
hh_income_6000_8000 = Variable('hh_income_6000_8000')
hh_income_8000_10000 = Variable('hh_income_8000_10000')
hh_income_10000_12000 = Variable('hh_income_10000_12000')
hh_income_12000_14000 = Variable('hh_income_12000_14000')
hh_income_14000_16000 = Variable('hh_income_14000_16000')
hh_income_g16000 = Variable('hh_income_g16000')


# Define the alternative-specific constant (ASC) and coefficients
ASC_1 = Beta('ASC_1', 0, None, None, 0)
ASC_2 = 0  # Normalized to 0

B_age = Beta('B_age', 0, None, None, 0)
B_interv_lang = Beta('B_interv_lang', 0, None, None, 0)
B_day_week = Beta('B_day_week', 0, None, None, 0)
B_dl_car = Beta('B_dl_car', 0, None, None, 0)
B_dl_moto = Beta('B_dl_moto', 0, None, None, 0)
B_park_work_car = Beta('B_park_work_car', 0, None, None, 0)
B_use_park_work = Beta('B_use_park_work', 0, None, None, 0)
B_park_work_cost = Beta('B_park_work_cost', 0, None, None, 0)
B_park_work_moto = Beta('B_park_work_moto', 0, None, None, 0)
B_park_work_bike = Beta('B_park_work_bike', 0, None, None, 0)
B_work_home = Beta('B_work_home', 0, None, None, 0)
B_work_home_perc = Beta('B_work_home_perc', 0, None, None, 0)
B_height = Beta('B_height', 0, None, None, 0)
B_weight = Beta('B_weight', 0, None, None, 0)
B_bike_avail = Beta('B_bike_avail', 0, None, None, 0)
B_moped_avail = Beta('B_moped_avail', 0, None, None, 0)
B_small_moped_avail = Beta('B_small_moped_avail', 0, None, None, 0)
B_moto_avail = Beta('B_moto_avail', 0, None, None, 0)
B_car_avail = Beta('B_car_avail', 0, None, None, 0)
B_rdist_sum = Beta('B_rdist_sum', 0, None, None, 0)
B_rdist_sum_Inland = Beta('B_rdist_sum_Inland', 0, None, None, 0)
B_travel_time_sum = Beta('B_travel_time_sum', 0, None, None, 0)
B_travel_time_sum_inland = Beta('B_travel_time_sum_inland', 0, None, None, 0)
B_n_stops = Beta('B_n_stops', 0, None, None, 0)
B_n_trips = Beta('B_n_trips', 0, None, None, 0)
B_travel_times = Beta('B_travel_times', 0, None, None, 0)
B_dist_softmodes = Beta('B_dist_softmodes', 0, None, None, 0)
B_dist_car = Beta('B_dist_car', 0, None, None, 0)
B_dist_pt = Beta('B_dist_pt', 0, None, None, 0)
B_dist_other = Beta('B_dist_other', 0, None, None, 0)
B_n_work_trips = Beta('B_n_work_trips', 0, None, None, 0)
B_n_edu_trips = Beta('B_n_edu_trips', 0, None, None, 0)
B_n_shopping_trips = Beta('B_n_shopping_trips', 0, None, None, 0)
B_n_business_trips = Beta('B_n_business_trips', 0, None, None, 0)
B_n_leisure_trips = Beta('B_n_leisure_trips', 0, None, None, 0)
B_n_service_trips = Beta('B_n_service_trips', 0, None, None, 0)
B_job_x_CH1903 = Beta('B_job_x_CH1903', 0, None, None, 0)
B_job_y_CH1903 = Beta('B_job_y_CH1903', 0, None, None, 0)
B_business_sector_agriculture = Beta('B_business_sector_agriculture', 0, None, None, 0)
B_business_sector_industry = Beta('B_business_sector_industry', 0, None, None, 0)
B_business_sector_services = Beta('B_business_sector_services', 0, None, None, 0)
B_fixed_work_sched = Beta('B_fixed_work_sched', 0, None, None, 0)
B_semi_flexible_work_sched = Beta('B_semi_flexible_work_sched', 0, None, None, 0)
B_ptsub_ga = Beta('B_ptsub_ga', 0, None, None, 0)
B_ptsub_regional = Beta('B_ptsub_regional', 0, None, None, 0)
B_ptsub_parcours = Beta('B_ptsub_parcours', 0, None, None, 0)
B_ptsub_other = Beta('B_ptsub_other', 0, None, None, 0)
B_ptsub_junior = Beta('B_ptsub_junior', 0, None, None, 0)
B_ptsub_ht = Beta('B_ptsub_ht', 0, None, None, 0)
B_ptsub_track_sev = Beta('B_ptsub_track_sev', 0, None, None, 0)
B_ptsub_ga_first = Beta('B_ptsub_ga_first', 0, None, None, 0)
B_ptsub_ga_second = Beta('B_ptsub_ga_second', 0, None, None, 0)
B_ptsub_parcours_first = Beta('B_ptsub_parcours_first', 0, None, None, 0)
B_ptsub_parcours_second = Beta('B_ptsub_parcours_second', 0, None, None, 0)
B_ptsub_regional_first = Beta('B_ptsub_regional_first', 0, None, None, 0)
B_ptsub_regional_second = Beta('B_ptsub_regional_second', 0, None, None, 0)
B_pt_emp_subsidies = Beta('B_pt_emp_subsidies', 0, None, None, 0)
B_can_walk_alone = Beta('B_can_walk_alone', 0, None, None, 0)
B_use_wheelchair = Beta('B_use_wheelchair', 0, None, None, 0)
B_car_sharing_subscription = Beta('B_car_sharing_subscription', 0, None, None, 0)
B_single = Beta('B_single', 0, None, None, 0)
B_married = Beta('B_married', 0, None, None, 0)
B_widowed = Beta('B_widowed', 0, None, None, 0)
B_divorced = Beta('B_divorced', 0, None, None, 0)
B_unmarried = Beta('B_unmarried', 0, None, None, 0)
B_registered_partnership = Beta('B_registered_partnership', 0, None, None, 0)
B_dissolved_partnership = Beta('B_dissolved_partnership', 0, None, None, 0)
B_male = Beta('B_male', 0, None, None, 0)
B_self_employed_with_employees = Beta('B_self_employed_with_employees', 0, None, None, 0)
B_self_employed_without_employees = Beta('B_self_employed_without_employees', 0, None, None, 0)
B_employed_by_family = Beta('B_employed_by_family', 0, None, None, 0)
B_employee_management = Beta('B_employee_management', 0, None, None, 0)
B_trainee = Beta('B_trainee', 0, None, None, 0)
B_unemployed = Beta('B_unemployed', 0, None, None, 0)
B_student = Beta('B_student', 0, None, None, 0)
B_retired = Beta('B_retired', 0, None, None, 0)
B_invalid = Beta('B_invalid', 0, None, None, 0)
B_stay_at_home_parent = Beta('B_stay_at_home_parent', 0, None, None, 0)
B_inactive = Beta('B_inactive', 0, None, None, 0)
B_compulsory_education = Beta('B_compulsory_education', 0, None, None, 0)
B_secondary_education = Beta('B_secondary_education', 0, None, None, 0)
B_tertiary_education = Beta('B_tertiary_education', 0, None, None, 0)
B_full_time = Beta('B_full_time', 0, None, None, 0)
B_part_time = Beta('B_part_time', 0, None, None, 0)
B_swiss = Beta('B_swiss', 0, None, None, 0)
B_job_location_city = Beta('B_job_location_city', 0, None, None, 0)
B_job_location_popsize_500000 = Beta('B_job_location_popsize_500000', 0, None, None, 0)
B_job_location_popsize_250000 = Beta('B_job_location_popsize_250000', 0, None, None, 0)
B_job_location_popsize_100000 = Beta('B_job_location_popsize_100000', 0, None, None, 0)
B_job_location_popsize_50000 = Beta('B_job_location_popsize_50000', 0, None, None, 0)
B_job_location_popsize_1000 = Beta('B_job_location_popsize_1000', 0, None, None, 0)
B_job_location_popsize_no_agglo = Beta('B_job_location_popsize_no_agglo', 0, None, None, 0)
B_hh_x_ch1903 = Beta('B_hh_x_ch1903', 0, None, None, 0)
B_hh_y_ch1903 = Beta('B_hh_y_ch1903', 0, None, None, 0)
B_hh_size = Beta('B_hh_size', 0, None, None, 0)
B_n_cars_hh = Beta('B_n_cars_hh', 0, None, None, 0)
B_n_car_park_hh = Beta('B_n_car_park_hh', 0, None, None, 0)
B_n_moto_hh = Beta('B_n_moto_hh', 0, None, None, 0)
B_n_s_moto_hh = Beta('B_n_s_moto_hh', 0, None, None, 0)
B_n_mopeds_hh = Beta('B_n_mopeds_hh', 0, None, None, 0)
B_n_bike_hh = Beta('B_n_bike_hh', 0, None, None, 0)
B_n_ebike_s_hh = Beta('B_n_ebike_s_hh', 0, None, None, 0)
B_n_ebike_f_hh = Beta('B_n_ebike_f_hh', 0, None, None, 0)
B_hh_location_city = Beta('B_hh_location_city', 0, None, None, 0)
B_hh_location_popsize_500000 = Beta('B_hh_location_popsize_500000', 0, None, None, 0)
B_hh_location_popsize_250000 = Beta('B_hh_location_popsize_250000', 0, None, None, 0)
B_hh_location_popsize_100000 = Beta('B_hh_location_popsize_100000', 0, None, None, 0)
B_hh_location_popsize_50000 = Beta('B_hh_location_popsize_50000', 0, None, None, 0)
B_hh_location_popsize_1000 = Beta('B_hh_location_popsize_1000', 0, None, None, 0)
B_hh_location_popsize_no_agglo = Beta('B_hh_location_popsize_no_agglo', 0, None, None, 0)
B_pt_access_A = Beta('B_pt_access_A', 0, None, None, 0)
B_pt_access_B = Beta('B_pt_access_B', 0, None, None, 0)
B_pt_access_C = Beta('B_pt_access_C', 0, None, None, 0)
B_pt_access_D = Beta('B_pt_access_D', 0, None, None, 0)
B_single_household = Beta('B_single_household', 0, None, None, 0)
B_couple_without_children = Beta('B_couple_without_children', 0, None, None, 0)
B_couple_with_children = Beta('B_couple_with_children', 0, None, None, 0)
B_single_parent = Beta('B_single_parent', 0, None, None, 0)
B_non_family_household = Beta('B_non_family_household', 0, None, None, 0)
B_hh_income_l2000 = Beta('B_hh_income_l2000', 0, None, None, 0)
B_hh_income_2000_4000 = Beta('B_hh_income_2000_4000', 0, None, None, 0)
B_hh_income_4000_6000 = Beta('B_hh_income_4000_6000', 0, None, None, 0)
B_hh_income_6000_8000 = Beta('B_hh_income_6000_8000', 0, None, None, 0)
B_hh_income_8000_10000 = Beta('B_hh_income_8000_10000', 0, None, None, 0)
B_hh_income_10000_12000 = Beta('B_hh_income_10000_12000', 0, None, None, 0)
B_hh_income_12000_14000 = Beta('B_hh_income_12000_14000', 0, None, None, 0)
B_hh_income_14000_16000 = Beta('B_hh_income_14000_16000', 0, None, None, 0)
B_hh_income_g16000 = Beta('B_hh_income_g16000', 0, None, None, 0)


# Define utility functions for the alternatives
V1 = (
    ASC_1
    + B_age * age
    + B_dl_car * dl_car
    + B_dl_moto * dl_moto
    + B_park_work_car * park_work_car
    + B_use_park_work * use_park_work
    + B_park_work_cost * park_work_cost
    + B_park_work_moto * park_work_moto
    + B_park_work_bike * park_work_bike
    + B_work_home * work_home
    + B_work_home_perc * work_home_perc
    + B_height * height
    + B_weight * weight
    + B_bike_avail * bike_avail
    + B_moped_avail * moped_avail
    + B_small_moped_avail * small_moped_avail
    + B_moto_avail * moto_avail
    + B_car_avail * car_avail
    + B_rdist_sum * rdist_sum
    + B_rdist_sum_Inland * rdist_sum_Inland
    + B_travel_time_sum * travel_time_sum
    + B_travel_time_sum_inland * travel_time_sum_inland
    + B_n_stops * n_stops
    + B_n_trips * n_trips
    + B_travel_times * travel_times
    + B_dist_softmodes * dist_softmodes
    + B_dist_car * dist_car
    + B_dist_pt * dist_pt
    + B_dist_other * dist_other
    + B_n_work_trips * n_work_trips
    + B_n_edu_trips * n_edu_trips
    + B_n_shopping_trips * n_shopping_trips
    + B_n_business_trips * n_business_trips
    + B_n_leisure_trips * n_leisure_trips
    + B_n_service_trips * n_service_trips
    + B_job_x_CH1903 * job_x_CH1903
    + B_job_y_CH1903 * job_y_CH1903
    + B_business_sector_agriculture * business_sector_agriculture
    + B_business_sector_industry * business_sector_industry
    + B_business_sector_services * business_sector_services
    + B_fixed_work_sched * fixed_work_sched
    + B_semi_flexible_work_sched * semi_flexible_work_sched
    + B_ptsub_ga * ptsub_ga
    + B_ptsub_regional * ptsub_regional
    + B_ptsub_parcours * ptsub_parcours
    + B_ptsub_other * ptsub_other
    + B_ptsub_junior * ptsub_junior
    + B_ptsub_ht * ptsub_ht
    + B_ptsub_track_sev * ptsub_track_sev
    + B_ptsub_ga_first * ptsub_ga_first
    + B_ptsub_ga_second * ptsub_ga_second
    + B_ptsub_parcours_first * ptsub_parcours_first
    + B_ptsub_parcours_second * ptsub_parcours_second
    + B_ptsub_regional_first * ptsub_regional_first
    + B_ptsub_regional_second * ptsub_regional_second
    + B_pt_emp_subsidies * pt_emp_subsidies
    + B_can_walk_alone * can_walk_alone
    + B_use_wheelchair * use_wheelchair
    + B_car_sharing_subscription * car_sharing_subscription
    + B_single * single
    + B_married * married
    + B_widowed * widowed
    + B_divorced * divorced
    + B_unmarried * unmarried
    + B_registered_partnership * registered_partnership
    + B_dissolved_partnership * dissolved_partnership
    + B_male * male
    + B_self_employed_with_employees * self_employed_with_employees
    + B_self_employed_without_employees * self_employed_without_employees
    + B_employed_by_family * employed_by_family
    + B_employee_management * employee_management
    + B_trainee * trainee
    + B_unemployed * unemployed
    + B_student * student
    + B_retired * retired
    + B_invalid * invalid
    + B_stay_at_home_parent * stay_at_home_parent
    + B_inactive * inactive
    + B_compulsory_education * compulsory_education
    + B_secondary_education * secondary_education
    + B_tertiary_education * tertiary_education
    + B_full_time * full_time
    + B_part_time * part_time
    + B_swiss * swiss
    + B_job_location_city * job_location_city
    + B_job_location_popsize_500000 * job_location_popsize_500000
    + B_job_location_popsize_250000 * job_location_popsize_250000
    + B_job_location_popsize_100000 * job_location_popsize_100000
    + B_job_location_popsize_50000 * job_location_popsize_50000
    + B_job_location_popsize_1000 * job_location_popsize_1000
    + B_job_location_popsize_no_agglo * job_location_popsize_no_agglo
    + B_hh_x_ch1903 * hh_x_ch1903
    + B_hh_y_ch1903 * hh_y_ch1903
    + B_hh_size * hh_size
    + B_n_cars_hh * n_cars_hh
    + B_n_car_park_hh * n_car_park_hh
    + B_n_moto_hh * n_moto_hh
    + B_n_s_moto_hh * n_s_moto_hh
    + B_n_mopeds_hh * n_mopeds_hh
    + B_n_bike_hh * n_bike_hh
    + B_n_ebike_s_hh * n_ebike_s_hh
    + B_n_ebike_f_hh * n_ebike_f_hh
    + B_hh_location_city * hh_location_city
    + B_hh_location_popsize_500000 * hh_location_popsize_500000
    + B_hh_location_popsize_250000 * hh_location_popsize_250000
    + B_hh_location_popsize_100000 * hh_location_popsize_100000
    + B_hh_location_popsize_50000 * hh_location_popsize_50000
    + B_hh_location_popsize_1000 * hh_location_popsize_1000
    + B_hh_location_popsize_no_agglo * hh_location_popsize_no_agglo
    + B_pt_access_A * pt_access_A
    + B_pt_access_B * pt_access_B
    + B_pt_access_C * pt_access_C
    + B_pt_access_D * pt_access_D
    + B_single_household * single_household
    + B_couple_without_children * couple_without_children
    + B_couple_with_children * couple_with_children
    + B_single_parent * single_parent
    + B_non_family_household * non_family_household
    + B_hh_income_l2000 * hh_income_l2000
    + B_hh_income_2000_4000 * hh_income_2000_4000
    + B_hh_income_4000_6000 * hh_income_4000_6000
    + B_hh_income_6000_8000 * hh_income_6000_8000
    + B_hh_income_8000_10000 * hh_income_8000_10000
    + B_hh_income_10000_12000 * hh_income_10000_12000
    + B_hh_income_12000_14000 * hh_income_12000_14000
    + B_hh_income_14000_16000 * hh_income_14000_16000
    + B_hh_income_g16000 * hh_income_g16000
)

V2 = 0  # Alternative 2 is normalized to zero

# Define the dictionary of utility functions
V = {1: V1, 2: V2}

# Define the availability conditions for each alternative
av = {1: 1, 2: 1}  # Assuming both alternatives are always available

# Define the model
logit_model = models.logit(V, av, df['choice'])

# Create the biogeme object
biogeme = bioLogLogit(df, logit_model)

# Estimate the parameters
results = biogeme.estimate()

# Print the results
print(results.getEstimatedParameters())
