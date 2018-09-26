

```python
#imports libraries and csv file
import pandas as pd
import datetime as dt
import numpy as np
import re
df = pd.read_csv('gun-violence-data_01-2013_03-2018.csv', index_col=None)

#displays 500 rows before auto collapsing - used for visual analysis
#pd.set_option('display.max_rows', 5000)
```

## Data Wrangling
This dataset was downloaded from kaggle: https://www.kaggle.com/jameslko/gun-violence-data. Dataset issues were found both visually and programmatically, and are listed below.  The most common issue found in this dataset is that of multiple values within each cell. Since most of the categories had overlap, and one event could have many delineations of each category, I created single categorical columns with binary operators in order to parse data within cells.

## Assess:

#### Data Quality
- Date is an object datatype and n_guns_involved, state_house_district, state_senate_district, and congressional_district are a floats
- There is no total injured/killed column
- There is no total injured, killed, arrested, or unharmed column
- city_or_county and participant_status have extra characters 
- unneccessary columns (and unwanted personal info): participant_age_group, participant_name, address, and incident_url_fields_missing
- participant_age column has extra characters

#### Data Tidyness
- city_or_county involved two kinds of data in one column.
- participant_status has "arrested", "injured", "killed" and "unharmed" in one column - cells have multiple entries.
- gun_stolen column has data for stolen, unknown, and not stolen in one column.
- participant_gender has many entries in each column of both genders
- participant_type has multiple entries of information for victim and suspect/subject in one column
- participant_relationship has many delineations within it
- gun_types have many kinds of data within column
- incident_characteristics has many kind os data in one column
- dataframe has too much information to be in one file. 



age range?

_______________________

The end result of the dataframe will be split into three separate files. Each will including different data, but will retain the incident_id column for the capability to merge or join later. Each column created will be added to a correctly named copy of the original dataframe, and the appropriate columns will be dropped at the end. The dataframes will contain information on the following
1. df_general - general information about the event
2. df_people - information about the people involved
3. df_gun - information about the guns used in the crime

##### df_general:
- incident_id
- date
- state
- city
- county
- incident_url
- source_url
- congressional_district
- state_house_district
- state_senate_district
- latitude
- longitude
- location_description
- notes
- participant name
- sources

##### df_people:
- incident_id
- incident_characteristics
- n_killed
- n_injured
- participant_age
- participant_relationship
- victim
- suspect
- male
- female
- unharmed_arrested
- unharmed
- arrested
- total_killed_injured
- total_involved
- relationship_significant_other
- relationship_mass_shooting_known
- relationship_family
- relationship_friend
- relationship_home_invasion_known
- relationship_coworker
- relationship_aquaintance
- relationship_neighbor
- incident_shot
- incident_driveby
- incident_tsa
- incident_nonshooting
- incident_domestic
- incident_standoff
- incident_gang
- incident_carjacking
- incident_suicide
- incident_murdersuicide
- incident_accident
- incident_homeinvasion
- incident_school
- incident_massshooting
- incident_animal
- incident_roadrage
- incident_abduction
- incident_defensive
- incident_sexcrime
- incident_spree
- incident_hatecrime
- incident_policetarget
- incident_institution
- incident_armed_robbery
- involved_child
- involved_drug_alcohol

##### df_gun:
- incident_id
- gun_handgun
- gun_shotgun
- gun_rifle
- gun_stolen
- n_guns_involved
- ghost_gun

## Clean


```python
#creates copy of df
df_general = df.copy()
df_gun = df.copy()
df_people = df.copy()
```

### Datatypes Fixes
- Change date column to datetime and n_guns_involved, state_house_district, state_senate_district, and congressional_district columns to objects


```python
#changes datatypes
df_general['date'] = pd.to_datetime(df_general['date'])
df_general['state_house_district'] = df_general['state_house_district'].astype(object)
df_general['state_senate_district'] = df_general['state_senate_district'].astype(object)
df_general['congressional_district'] = df_general['congressional_district'].astype(object)
df_gun['n_guns_involved'] = df_gun['n_guns_involved'].astype(object)
```

### Column Fixes
#### Column Fixes - County
- split city_or_county into two columns and get rid of extra characters.


```python
#creates new column - county
df_general['county'] = df_general.loc[df_general['city_or_county'].str.contains('county', case=False), 'city_or_county']
#renames city or country column to city
df_general.rename(columns={'city_or_county': 'city'}, inplace=True)
```


```python
#extracts extra characters
df_general['county'] = df_general['county'].str.lower()
df_general['county'] = df_general['county'].str.replace('\(|\)', '')
df_general['county'] = df_general['county'].str.replace('county county', 'county')
```

#### Column Fixes - participant_type
- Split column into two columns: "victim" and "suspect" and drop original column


```python
#changes str to lower case
df_people['participant_type'] = df_people['participant_type'].str.lower()
#creates new columns for participant_types
df_people['victim'] = df_people.participant_type.str.count('victim').astype(object)
df_people['suspect'] = df_people.participant_type.str.count('Subject-Suspect').astype(object)
df_people = df_people.drop(columns = ['participant_type'])
```

#### Column Fixes - participant_gender
- Split into separate columns ("male" and "female") with count of each gender, and drop original column.


```python
#changes str to lower case
df_people['participant_gender'] = df_people['participant_gender'].str.lower()
#creates new columns for gender
df_people['male'] = df_people.participant_gender.str.count('male').astype(object)
df_people['female'] = df_people.participant_gender.str.count('female').astype(object)
df_people = df_people.drop(columns = ['participant_gender'])
```

#### Column Fixes - participant_status
- Split into columns "unharmed" and "arrested" with count of each type, and srop original column. 
- Creates total_involved and total_killed_injured columns


```python
#changes str to lower case
df_people['participant_status'] = df_people['participant_status'].str.lower()
#makes column for unharmed/arrested
df_people['unharmed_arrested'] = df_people.participant_status.str.count('unharmed, arrested')
#creates columns for singular accounts or unharmed and arrested
df_people['unharmed'] = df_people.participant_status.str.count('unharmed')
df_people['arrested'] = df_people.participant_status.str.count('arrested')
```


```python
#subtracts duplicates
df_people['arrested'] = df_people.arrested - df_people.unharmed_arrested
df_people['unharmed'] = df_people.unharmed - df_people.unharmed_arrested
#drops unneccesary column
df_people.drop(columns=['participant_status'], inplace=True)
```


```python
#adds those killed and injured into total_affected column
df_people['total_killed_injured'] = df_people.n_killed + df_people.n_injured
df_people['total_involved'] = df_people.n_killed + df_people.n_injured + df_people.arrested + df_people.unharmed + df_people.unharmed_arrested
```


```python
#change datatypes
df_people['arrested'] = df_people['arrested'].astype(object)
df_people['unharmed'] = df_people['unharmed'].astype(object)
df_people['unharmed_arrested'] = df_people['unharmed_arrested'].astype(object)
df_people['total_killed_injured'] = df_people['total_killed_injured'].astype(object)
df_people['total_involved'] = df_people['total_involved'].astype(object)
```

#### Column Fixes - gun_type
- Sort into types and then create three new columns for each type.

There were many kinds of guns included in this dataset. Each type was found by visually looking through column. Each type was then researched and ammalgamated into "handgun", "shotgun", and "rifle". There were some types that did not have enough information to sort them into a gun type. Those delineations are listed below, and were not retained in the final version of df_gun.

Gun Types sorted:
- handgun
- shotgun
- rifle (includes semis)
- 223 Rem (AR-15) - rifle
- 7.62 (AK-47) - rifle
- 10mm - handgun
- 32 auto - handgun
- 380 auto - handgun
- 45 auto - handgun
- 25 auto - handgun
- 357 mag - handgun
- 12 gauge - shotgun
- 16 gauge - shotgun
- 20 gauge - shotgun
- 28 gauge - shotgun
- 410 gauge - shotgun
- 40 sw - handgun
- 30-30 win - rifle
- 308 win - rifle
- 300 win - rifle

Gun Types left out of df:
- 9mm - caliber
- 22 LR - caliber
- 30-06 spr - rifle/machine gun
- 44 mag - revolver and rifle and carbines
- 38 spl - revolver, carbines, and pistol - sometimes semi)
- unknown - should be NaN
- other - should be NaN


```python
#changes column case
df_gun['gun_type'] = df_gun['gun_type'].str.lower()
#creates gun type columns to keep
df_gun['gun_handgun'] = df_gun.gun_type.str.count('handgun')
df_gun['gun_shotgun'] = df_gun.gun_type.str.count('shotgun')
df_gun['gun_rifle'] = df_gun.gun_type.str.count('rifle')
#creates interim gun type columns
df_gun['3030win'] = df_gun.gun_type.str.count('30-30 win')
df_gun['300win'] = df_gun.gun_type.str.count('308 win')
df_gun['308win'] = df_gun.gun_type.str.count('308 win')
df_gun['ar'] = df_gun.gun_type.str.count('ar-15')
df_gun['ak'] = df_gun.gun_type.str.count('ak-47')
df_gun['10mm'] = df_gun.gun_type.str.count('10mm')
df_gun['32auto'] = df_gun.gun_type.str.count('32 auto')
df_gun['380auto'] = df_gun.gun_type.str.count('380 auto')
df_gun['45auto'] = df_gun.gun_type.str.count('45 auto')
df_gun['25auto'] = df_gun.gun_type.str.count('25 auto')
df_gun['357mag'] = df_gun.gun_type.str.count('357 mag')
df_gun['40sw'] = df_gun.gun_type.str.count('40 sw')
df_gun['12gauge'] = df_gun.gun_type.str.count('12 gauge')
df_gun['16gauge'] = df_gun.gun_type.str.count('16 gauge')
df_gun['20gauge'] = df_gun.gun_type.str.count('20 gauge')
df_gun['28gauge'] = df_gun.gun_type.str.count('28 gauge')
df_gun['410gauge'] = df_gun.gun_type.str.count('410 gauge')
```


```python
#changes interim columns into aggregated columns
df_gun['gun_shotgun'] = df_gun['gun_shotgun'] + df_gun['410gauge'] + df_gun['28gauge'] + df_gun['20gauge'] +  df_gun['16gauge'] + df_gun['12gauge']
df_gun['gun_rifle'] = df_gun['gun_rifle'] + df_gun['3030win'] + df_gun['300win'] + df_gun['308win'] + df_gun['ar'] + df_gun['ak']
df_gun['gun_handgun'] = df_gun['gun_handgun'] + df_gun['40sw'] + df_gun['357mag'] + df_gun['25auto'] + df_gun['45auto'] + df_gun['380auto'] + df_gun['32auto'] + df_gun['10mm']
```


```python
#drop interim columns
df_gun.drop(columns=['gun_type', '3030win', '300win', '308win', 'ak', 'ar', '10mm', '32auto', '380auto', '45auto', '25auto', '357mag', '40sw', '12gauge', '16gauge', '20gauge', '28gauge', '410gauge'], inplace=True)
```


```python
#changes datatypes 
df_gun['gun_shotgun'] = df_gun['gun_shotgun'].astype(object)
df_gun['gun_rifle'] = df_gun['gun_rifle'].astype(object)
df_gun['gun_handgun'] = df_gun['gun_handgun'].astype(object)
```

#### Column Fixes - gun_stolen
- Splits into separate columns: stolen, not-stolen, and unknown, and drops original column.


```python
# changes column case
df_gun['gun_stolen'] = df_gun['gun_stolen'].str.lower()
#creates interim columns for gun delineations
df_gun['unknown_stolen'] = df_gun.gun_stolen.str.count('unknown')
df_gun['stolen_gun'] = df_gun.gun_stolen.str.count('stolen')
df_gun['not_stolen'] = df_gun.gun_stolen.str.count('not-stolen')
df_gun['gun_stolen'] = df_gun['stolen_gun']
```


```python
#changes datatypes
df_gun['unknown_stolen'] = df_gun['unknown_stolen'].astype(object)
df_gun['gun_stolen'] = df_gun['gun_stolen'].astype(object)
df_gun['not_stolen'] = df_gun['not_stolen'].astype(object)
#drops interim columns
df_gun.drop(columns=['stolen_gun', 'unknown_stolen', 'not_stolen'], inplace=True)
```

#### Column Fixes - participant_relationship
- Splits column into several by type of relationship, and drops original column. Relationships left out that are later delineated by incident_type (below).

Relationship_status in df:
- significant other - current or former
- armed robbery - data also in incident characteristics
- mass shooting - perp knows victims 
- mass shooting - random victims - data also in incident characteristics
- family
- friends
- home invasion - perp knows victims
- home invasion - perp doesn't know victims - data also in incident characteristics
- gang vs gang - data also in incident characteristics
- co-worker
- aquaintance
- neighbor
- drive by - random victims - data also in incident characteristics


```python
#changes case
df_people['participant_relationship'] = df_people['participant_relationship'].str.lower()
#splits relationships into separate colums
df_people['relationship_significant_other'] = df_people.participant_relationship.str.count('significant other').astype(object)
df_people['relationship_mass_shooting_known'] = df_people.participant_relationship.str.count('mass shooting - perp').astype(object)
df_people['relationship_family'] = df_people.participant_relationship.str.count('family').astype(object)
df_people['relationship_friend'] = df_people.participant_relationship.str.count('friend').astype(object)
df_people['relationship_home_invasion_known'] = df_people.participant_relationship.str.count('home invasion - - perp knows').astype(object)
df_people['relationship_coworker'] = df_people.participant_relationship.str.count('co-worker').astype(object)
df_people['relationship_aquaintance'] = df_people.participant_relationship.str.count('aquaintance').astype(object)
df_people['relationship_neighbor'] = df_people.participant_relationship.str.count('neighbor').astype(object)
df_people.drop(columns=['participant_relationship'], inplace=True)
```

#### Column Fixes - incident characteristics
- Splits column into many by type of incident, and drops original column. Some incidents were not added to the final version of the df because of specificity and overlap. There were some incident types that were listed multiple times - extra code was written below to avoid this issue for those columns.

Incidents included in final df:
- Institution/Group/Business
- Armed robbery
- alcohol or drugs
- shot
- drive by
- TSA Action
- Non shooting incident
- Domestic Violence
- standoff
- Gang
- Car-jacking
- suicide
- murder/suicide
- accident
- Home Invasion
- School Incident
- Mass Shooting
- Animal shot/killed
- Road rage
- child involved
- Kidnapping/abductions/hostage
- defensive use
- sex crime
- spree shooting
- hate crime
- police targeted
- ghost gun *(should be in gun_type)


Incidents not included in final df:
- Possession (found at other crimes)
- ATF/LE Confiscation/Raid/Arrest
- Brandishing/flourishing/open carry/lost/found
- Officer Involved Incident
- Officer Involved Shooting - subject/suspect/perpetrator killed 
- officer involved shooting - subject/suspect/perpetrator suicide by cop
- officer involved shooting - subject/suspect/perpetrator suicide at 
- Gun(s) stolen from owner
- Shots fired, no action (reported, no evidence found)
- ShotSpotter
- subject/suspect/perpetrator, one location)
- Implied Weapon
- LOCKDOWN/ALERT ONLY: No GV Incident Occurred Onsite
- Pistol-whipping
- BB/Pellet/Replica gun
- Assault weapon (AR-15, AK-47, and ALL variants defined by law enforcement)
- hunting accident
- playing with gun
- Stolen/Illegally owned gun{s} recovered during arrest/warrant
- Possession of gun by felon or prohibited person
- defensive use - without a gun
- Defensive Use - Victim stops crime
- defensive use - good samaritan/third party
- defensive use - no shots fired
- Defensive Use - Shots fired, no injury/death
- Defensive Use - Crime occurs, victim shoots subject/suspect/perpetrator
- Defensive Use - Stand Your Ground/Castle Doctrine established
- self-Inflicted (not suicide or suicide attempt - NO PERP)
- Cleaning gun
- Guns stolen from law enforcement
- Gun buy back action
- Thought gun was unloaded
- Unlawful purchase/sale
- concealed carry license - perpetrator
- concealed carry license - victim
- non-aggression incident
- criminal act with stolen gun


```python
#changes case
df_people['incident_characteristics'] = df_people['incident_characteristics'].str.lower()
#creates new columns
df_people['incident_shot'] = df_people.incident_characteristics.str.count('shot')
df_people['incident_driveby'] = df_people.incident_characteristics.str.count('drive-by')
df_people['incident_tsa'] = df_people.incident_characteristics.str.count('tsa action')
df_people['incident_nonshooting'] = df_people.incident_characteristics.str.count('non-shooting')
df_people['incident_domestic'] = df_people.incident_characteristics.str.count('domestic violence')
df_people['incident_standoff'] = df_people.incident_characteristics.str.count('standoff')
df_people['incident_gang'] = df_people.incident_characteristics.str.count('gang')
df_people['incident_carjacking'] = df_people.incident_characteristics.str.count('car-jacking')
df_people['incident_suicide'] = df_people.incident_characteristics.str.count('suicide')
df_people['incident_murdersuicide'] = df_people.incident_characteristics.str.count('murder/suicide')
df_people['incident_accident'] = df_people.incident_characteristics.str.count('accident')
df_people['incident_homeinvasion'] = df_people.incident_characteristics.str.count('home invasion')
df_people['incident_school'] = df_people.incident_characteristics.str.count('school')
df_people['incident_massshooting'] = df_people.incident_characteristics.str.count('mass shooting')
df_people['incident_animal'] = df_people.incident_characteristics.str.count('animal shot/killed')
df_people['incident_roadrage'] = df_people.incident_characteristics.str.count('road rage')
df_people['involved_child'] = df_people.incident_characteristics.str.count('child involved')
df_people['incident_abduction'] = df_people.incident_characteristics.str.count('kidnapping')
df_people['incident_defensive'] = df_people.incident_characteristics.str.count('defensive use')
df_people['incident_sexcrime'] = df_people.incident_characteristics.str.count('sex crime')
df_people['incident_spree'] = df_people.incident_characteristics.str.count('spree shooting')
df_people['incident_hatecrime'] = df_people.incident_characteristics.str.count('hate crime')
df_people['incident_policetarget'] = df_people.incident_characteristics.str.count('police targeted')
df_gun['ghostgun'] = df_gun.incident_characteristics.str.count('ghost gun')
```


```python
#deletes duplicates for shot column
df_people['incident_shot'] = df_people.incident_shot.replace(2.0, 1.0)
df_people['incident_shot'] = df_people.incident_shot.replace(3.0, 1.0)
df_people['incident_shot'] = df_people.incident_shot.replace(4.0, 1.0)
df_people['incident_shot'] = df_people.incident_shot.replace(5.0, 1.0)
```


```python
#deletes duplicates for standoff column
df_people['incident_standoff'] = df_people.incident_standoff.replace(2.0, 1.0)
```


```python
#deletes duplicates for suicide column
df_people['incident_suicide'] = df_people.incident_suicide.replace(2.0, 1.0)
df_people['incident_suicide'] = df_people.incident_suicide.replace(3.0, 1.0)
df_people['incident_suicide'] = df_people.incident_suicide.replace(4.0, 1.0)
df_people['incident_suicide'] = df_people.incident_suicide.replace(5.0, 1.0)
```


```python
#deletes duplicates for murdersuicide column
df_people['incident_murdersuicide'] = df_people.incident_murdersuicide.replace(2.0, 1.0)
```


```python
#deletes duplicates for accident column
df_people['incident_accident'] = df_people.incident_accident.replace(2.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(3.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(4.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(5.0, 1.0)
df_people['incident_accident'] = df_people.incident_accident.replace(6.0, 1.0)
```


```python
#deletes duplicates for homeinvasion column
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(2.0, 1.0)
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(3.0, 1.0)
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(4.0, 1.0)
df_people['incident_homeinvasion'] = df_people.incident_homeinvasion.replace(5.0, 1.0)
```


```python
#deletes duplicates for school column
df_people['incident_school'] = df_people.incident_school.replace(2.0, 1.0)
df_people['incident_school'] = df_people.incident_school.replace(3.0, 1.0)
df_people['incident_school'] = df_people.incident_school.replace(4.0, 1.0)
df_people['incident_school'] = df_people.incident_school.replace(5.0, 1.0)
```


```python
#deletes duplicates for defensive column
df_people['incident_defensive'] = df_people.incident_defensive.replace(2.0, 1.0)
df_people['incident_defensive'] = df_people.incident_defensive.replace(3.0, 1.0)
df_people['incident_defensive'] = df_people.incident_defensive.replace(4.0, 1.0)
df_people['incident_defensive'] = df_people.incident_defensive.replace(5.0, 1.0)
```


```python
#aggregates columns with overlap in armed robbery
df_people['gun_shop'] = df_people.incident_characteristics.str.count('gun shop robbery')
df_people['incident_armed_robbery'] = df_people.incident_characteristics.str.count('armed robbery')
#adds columns
df_people['incident_armed_robbery'] = df_people['incident_armed_robbery'] + df_people['gun_shop']
#deletes duplicates
df_people['incident_armed_robbery'] = df_people.incident_armed_robbery.replace(2.0, 1.0)
#drops interim column
df_people.drop(columns=['gun_shop'], inplace=True)
```


```python
#aggregates columns with overlap in drugs/alcohol
df_people['drug_involvement'] = df_people.incident_characteristics.str.count('drug involvement')
df_people['involved_drug_alcohol'] = df_people.incident_characteristics.str.count('alcohol or drugs')
#adds columns
df_people['involved_drug_alcohol'] = df_people['involved_drug_alcohol'] + df_people['drug_involvement']
#replaces duplicates
df_people['involved_drug_alcohol'] = df_people.involved_drug_alcohol.replace(2.0, 1.0)
#drops interim column
df_people.drop(columns=['drug_involvement'], inplace=True)
```


```python
#aggregates columns with overlap in institution
df_people['bar'] = df_people.incident_characteristics.str.count('bar/club')
df_people['gun_range'] = df_people.incident_characteristics.str.count('gun range/gun shop')
df_people['house_party'] = df_people.incident_characteristics.str.count('house party')
df_people['incident_institution'] = df_people.incident_characteristics.str.count('institution/group/business')
#adds columns
df_people['incident_institution'] = df_people['incident_institution'] + df_people['bar'] + df_people['gun_range'] + df_people['house_party']
#replaces duplicates
df_people['incident_institution'] = df_people.incident_institution.replace(2.0, 1.0)
df_people['incident_institution'] = df_people.incident_institution.replace(3.0, 1.0)
#drops interim column
df_people.drop(columns=['bar', 'gun_range', 'house_party'], inplace=True)
```


```python
#changes datatype
df_people['incident_shot'] = df_people['incident_shot'].astype(object)
df_people['incident_driveby'] = df_people['incident_driveby'].astype(object)
df_people['incident_tsa'] = df_people['incident_tsa'].astype(object)
df_people['incident_nonshooting'] = df_people['incident_nonshooting'].astype(object)
df_people['incident_domestic'] = df_people['incident_domestic'].astype(object)
df_people['incident_standoff'] = df_people['incident_standoff'].astype(object)
df_people['incident_gang'] = df_people['incident_gang'].astype(object)
df_people['incident_carjacking'] = df_people['incident_carjacking'].astype(object)
df_people['incident_suicide'] = df_people['incident_suicide'].astype(object)
df_people['incident_murdersuicide'] = df_people['incident_murdersuicide'].astype(object)
df_people['incident_accident'] = df_people['incident_accident'].astype(object)
df_people['incident_homeinvasion'] = df_people['incident_homeinvasion'].astype(object)
df_people['incident_school'] = df_people['incident_school'].astype(object)
df_people['incident_massshooting'] = df_people['incident_massshooting'].astype(object)
df_people['incident_animal'] = df_people['incident_animal'].astype(object)
df_people['incident_roadrage'] = df_people['incident_roadrage'].astype(object)
df_people['involved_child'] = df_people['involved_child'].astype(object)
df_people['incident_abduction'] = df_people['incident_abduction'].astype(object)
df_people['incident_defensive'] = df_people['incident_defensive'].astype(object)
df_people['incident_sexcrime'] = df_people['incident_sexcrime'].astype(object)
df_people['incident_spree'] = df_people['incident_spree'].astype(object)
df_people['incident_hatecrime'] = df_people['incident_hatecrime'].astype(object)
df_people['incident_policetarget'] = df_people['incident_policetarget'].astype(object)
df_people['involved_drug_alcohol'] = df_people['involved_drug_alcohol'].astype(object)
df_gun['ghostgun'] = df_gun['ghostgun'].astype(object)
```

#### Column fixes - participant_age
- Removes extra characters.


```python
#removes extra characters 
df_people['participant_age'] = df_people['participant_age'].str.replace('\d::', '')
```




    0::Adult 18+                                                                                                                                       94671
    0::Adult 18+||1::Adult 18+                                                                                                                         49273
    0::Adult 18+||1::Adult 18+||2::Adult 18+                                                                                                           13893
    0::Teen 12-17                                                                                                                                       7392
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Adult 18+                                                                                              4975
    1::Adult 18+                                                                                                                                        3916
    0::Adult 18+||1::Teen 12-17                                                                                                                         1962
    0::Teen 12-17||1::Adult 18+                                                                                                                         1914
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Adult 18+||4::Adult 18+                                                                                1736
    0::Teen 12-17||1::Teen 12-17                                                                                                                        1673
    0:Adult 18+                                                                                                                                         1252
    0::Child 0-11                                                                                                                                       1088
    1::Adult 18+||2::Adult 18+                                                                                                                           961
    0:Adult 18+|1:Adult 18+                                                                                                                              904
    0::Adult 18+||1::Adult 18+||2::Teen 12-17                                                                                                            642
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Adult 18+||4::Adult 18+||5::Adult 18+                                                                   609
    0::Child 0-11||1::Adult 18+                                                                                                                          577
    0::Teen 12-17||1::Adult 18+||2::Adult 18+                                                                                                            531
    0::Adult 18+||1::Teen 12-17||2::Teen 12-17                                                                                                           427
    0::Adult 18+||1::Teen 12-17||2::Adult 18+                                                                                                            415
    0::Teen 12-17||1::Teen 12-17||2::Teen 12-17                                                                                                          372
    0:Adult 18+|1:Adult 18+|2:Adult 18+                                                                                                                  348
    2::Adult 18+                                                                                                                                         319
    0::Adult 18+||2::Adult 18+                                                                                                                           277
    1::Adult 18+||2::Adult 18+||3::Adult 18+                                                                                                             276
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Adult 18+||4::Adult 18+||5::Adult 18+||6::Adult 18+                                                     229
    0::Teen 12-17||1::Teen 12-17||2::Adult 18+                                                                                                           228
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Teen 12-17                                                                                              213
    0::Child 0-11||1::Child 0-11                                                                                                                         205
    0::Adult 18+||1::Child 0-11||2::Adult 18+                                                                                                            190
                                                                                                                                                       ...  
    0::Adult 18+||1::Adult 18+||2::Child 0-11||3::Child 0-11||4::Child 0-11||5::Child 0-11||6::Child 0-11||7::Adult 18+||8::Adult 18+||9::Adult 18+        1
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Teen 12-17||4::Teen 12-17||5::Teen 12-17||6::Teen 12-17||7::Teen 12-17||8::Adult 18+                      1
    0::Teen 12-17||1::Teen 12-17||2::Teen 12-17||3::Child 0-11                                                                                             1
    0::Teen 12-17||1::Adult 18+||3::Adult 18+||4::Adult 18+||5::Adult 18+||6::Adult 18+                                                                    1
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Adult 18+||4::Child 0-11||5::Child 0-11||6::Child 0-11||7::Adult 18+                                      1
    0::Teen 12-17||1::Teen 12-17||2::Teen 12-17||3::Teen 12-17||4::Teen 12-17||5::Teen 12-17||6::Teen 12-17||7::Adult 18+                                  1
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Child 0-11||4::Child 0-11||5::Child 0-11                                                                  1
    0::Child 0-11||1::Adult 18+||2::Adult 18+||3::Adult 18+||4::Child 0-11                                                                                 1
    0::Teen 12-17||1::Teen 12-17||2::Adult 18+||3::Teen 12-17||4::Teen 12-17||5::Adult 18+||6::Adult 18+                                                   1
    0:Child 0-11|1:Child 0-11|2:Teen 12-17|3:Teen 12-17                                                                                                    1
    0::Child 0-11||1::Child 0-11||2::Child 0-11||3::Child 0-11||4::Adult 18+                                                                               1
    0::Child 0-11||1::Teen 12-17||2::Teen 12-17||3::Adult 18+||4::Adult 18+                                                                                1
    0::Teen 12-17||1::Teen 12-17||2::Adult 18+||3::Adult 18+||4::Adult 18+||5::Adult 18+||6::Teen 12-17||7::Teen 12-17||8::Teen 12-17                      1
    0:Adult 18+|1:Teen 12-17|2:Adult 18+|3:Teen 12-17|4:Adult 18+|5:Adult 18+|6:Teen 12-17                                                                 1
    0::Adult 18+||3::Adult 18+||4::Adult 18+||5::Teen 12-17                                                                                                1
    0::Teen 12-17||2::Teen 12-17||3::Teen 12-17||4::Teen 12-17                                                                                             1
    0::Adult 18+||1::Child 0-11||2::Child 0-11||6::Teen 12-17                                                                                              1
    13::Adult 18+                                                                                                                                          1
    0::Adult 18+||1::Adult 18+||2::Teen 12-17||3::Teen 12-17||4::Adult 18+||5::Adult 18+||6::Teen 12-17||7::Adult 18+                                      1
    0::Adult 18+||1::Adult 18+||2::Adult 18+||7::Adult 18+||8::Adult 18+||9::Adult 18+                                                                     1
    1::Child 0-11||6::Adult 18+||7::Adult 18+                                                                                                              1
    0::Teen 12-17||1::Teen 12-17||2::Teen 12-17||4::Adult 18+                                                                                              1
    3::Adult 18+||4::Adult 18+||5::Adult 18+||6::Teen 12-17||7::Adult 18+||8::Adult 18+||9::Adult 18+||10::Adult 18+||11::Adult 18+                        1
    0::Teen 12-17||1::Child 0-11||2::Teen 12-17||3::Adult 18+                                                                                              1
    0::Adult 18+||1::Child 0-11||2::Adult 18+||3::Teen 12-17||4::Adult 18+                                                                                 1
    0::Teen 12-17||1::Teen 12-17||2::Adult 18+||3::Adult 18+||4::Adult 18+||5::Adult 18+||6::Teen 12-17                                                    1
    1::Child 0-11||2::Adult 18+||3::Adult 18+                                                                                                              1
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Child 0-11||4::Child 0-11||5::Child 0-11||7::Adult 18+                                                    1
    0::Adult 18+||1::Adult 18+||2::Teen 12-17||3::Teen 12-17||4::Teen 12-17||5::Adult 18+||6::Adult 18+||7::Adult 18+                                      1
    0::Adult 18+||1::Adult 18+||2::Adult 18+||3::Adult 18+||4::Teen 12-17||5::Adult 18+||6::Adult 18+||7::Adult 18+||8::Adult 18+                          1
    Name: participant_age_group, Length: 898, dtype: int64



#### Dropping Columns
- Drops unneccessary columns for each df.


```python
#drops unncessary columns in df_general
df_general = df_general[['incident_id', 'date', 'state', 'county', 'city', 'congressional_district', 'incident_url', 'source_url', 'latitude', 'location_description', 'longitude', 'notes', 'sources', 'state_house_district', 'state_senate_district']]
```


```python
#drops unncessary columns in df_people
df_people.drop(columns=['date', 'state', 'address', 'city_or_county', 'source_url', 'incident_url', 'gun_stolen', 'gun_type', 'participant_name', 'participant_age_group', 'congressional_district', 'incident_url_fields_missing', 'incident_characteristics', 'latitude', 'location_description', 'longitude', 'n_guns_involved', 'notes', 'sources', 'state_house_district', 'state_senate_district'], inplace=True)
```


```python
#drops unncessary columns in df_gun
df_gun = df_gun[['incident_id', 'gun_stolen', 'n_guns_involved', 'gun_handgun', 'gun_shotgun', 'gun_rifle', 'ghostgun']]
```

## Saves to CSV


```python
df_general.to_csv('gun_violence_general.csv')
df_people.to_csv('gun_violence_people.csv')
df_gun.to_csv('gun_violence_gun.csv')
```

### References for Wrangling
- https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html
- https://stackoverflow.com/questions/48094854/python-convert-object-to-float
- https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas
- https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html
- https://www.reddit.com/r/learnpython/comments/4zn20y/how_to_convert_sparse_pandas_dataframe_with_nan/?st=jlyfe2zc&sh=7172d3e9
- https://stackoverflow.com/questions/41550746/error-using-astype-when-nan-exists-in-a-dataframe
- https://chrisalbon.com/python/data_wrangling/pandas_replace_values/
- https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
- https://stackoverflow.com/questions/39768547/replace-whole-string-if-it-contains-substring-in-pandas
- https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.contains.html
- https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
- https://stackoverflow.com/questions/22520932/python-remove-all-non-alphabet-chars-from-string/22521156
- https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.any.html
- https://pandas.pydata.org/pandas-docs/version/0.18/generated/pandas.Series.str.contains.html
- https://stackoverflow.com/questions/42082385/pandas-slicing-selecting-with-multiple-conditions-with-or-statement
- https://stackoverflow.com/questions/22086116/how-do-you-filter-pandas-dataframes-by-multiple-columns
- https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean
- https://stackoverflow.com/questions/42152910/pandas-counting-quantity-of-commas-in-character-field
- https://stackoverflow.com/questions/27975069/how-to-filter-rows-containing-a-string-pattern-from-a-pandas-dataframe
- https://stackoverflow.com/questions/36362432/pandas-refer-to-column-name-case-insensitive
- https://chrisalbon.com/python/data_wrangling/pandas_regex_to_create_columns/
- https://stackoverflow.com/questions/41679687/pandas-searching-for-a-character-in-a-dataframe
- https://stackoverflow.com/questions/13682044/pandas-dataframe-remove-unwanted-parts-from-strings-in-a-column
- http://songhuiming.github.io/pages/2017/04/02/jupyter-and-pandas-display/
- https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html
- https://stackoverflow.com/questions/22573121/splice-a-string-based-on-certain-characters
- https://stackoverflow.com/questions/43768023/remove-characters-from-pandas-column
- https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe
- https://stackoverflow.com/questions/45447848/check-for-words-from-list-and-remove-those-words-in-pandas-dataframe-column
- https://www.w3schools.com/python/python_lists.asp
- https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas
- https://stackoverflow.com/questions/3559559/how-to-delete-a-character-from-a-string-using-python
