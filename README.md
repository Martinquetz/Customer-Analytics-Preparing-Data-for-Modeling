

# Customer_._Analytics_Preparing_Data_for_Modeling
Preprocessing datasets in preparation for modeling.


## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources and Description](#data-sources-and-description)
- [Tools Used](#tools-used)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Analysis](#analysis)
- [Results and Findings](#results-and-findings)
- [Limitations and Recommendations](#limitations-and-recommendations)
- [Limitations](#limitations)
- [References](#references)

### **Project Overview** 
<img width="812" height="367" alt="hr-image-small" src="https://github.com/user-attachments/assets/bf3eaea2-0a67-4335-99ff-bad98259ef5e" />

A common problem when creating models to generate business value from data is that the datasets can be so large that it can take days for the model to generate predictions. Ensuring that your dataset is stored as efficiently as possible is crucial for allowing these models to run on a more reasonable timescale without needing to reduce the dataset's size.

You've been hired by a major online data science training provider called *Training Data Ltd.* to clean up one of their largest customer datasets. This dataset will eventually be used to predict whether their students are looking for a new job or not, information that they will then use to direct them to prospective recruiters.


### **Data Sources and Description**
The customer data is stored in a CSV file named `customer_train.csv`, which is a subset of their entire customer dataset. The dataset contains anonymized student information.

#### The Metadata

| Column                   | Description                                                                      |
|------------------------- |--------------------------------------------------------------------------------- |
| `student_id`             | A unique ID for each student.                                                    |
| `city`                   | A code for the city the student lives in.                                        |
| `city_development_index` | A scaled development index for the city.                                         |
| `gender`                 | The student's gender.                                                            |
| `relevant_experience`    | An indicator of the student's work-relevant experience.                          |
| `enrolled_university`    | The type of university course enrolled in (if any).                              |
| `education_level`        | The student's education level.                                                   |
| `major_discipline`       | The educational discipline of the student.                                       |
| `experience`             | The student's total work experience (in years).                                  |
| `company_size`           | The number of employees at the student's current employer.                       |
| `company_type`           | The type of company employing the student.                                       |
| `last_new_job`           | The number of years between the student's current and previous jobs.             |
| `training_hours`         | The number of hours of training completed.                                       |
| `job_change`             | An indicator of whether the student is looking for a new job (`1`) or not (`0`). |


### **Tools Used**
The Python programming language, along with its data analysis libraries such as Pandas and NumPy, is utilized for data manipulation and analysis in this project. Jupyter Notebook was employed as the development environment for coding and documentation.

#### _Import Necessary Libraries_
```py
  # Import necessary libraries
  import pandas as pd
  import numpy as np
```


### **Data Cleaning and Preparation**
The dataset didn't require cleaning because it's known to be clean; therefore, it was used as imported.

#### _Read The Dataset and View it_
```py
  # import the car data
  ds_jobs = pd.read_csv(r"C:\Users\username\customer_train.csv")
  
  # View the dataset
  ds_jobs.head()
```
<img width="1003" height="283" alt="head1" src="https://github.com/user-attachments/assets/cc573d41-2aeb-4436-9b98-4cbec720c3a3" />



### **Analysis**
#### Create a copy of ds_jobs for transforming
```py
    # Create a copy of ds_jobs for transforming
    ds_jobs_transformed = ds_jobs.copy()
```

#### Call the .info() method on the ds_jobs_transformed to examine the data types and memory.
```py
    # Call the .info() method.
    ds_jobs_transformed.info()
```


#### Select only object-type columns.
```py
    # Call the .info() method.
    object_cols = ds_jobs_transformed.select_dtypes(include='object').columns

    # Get value counts for each object column
    value_counts_obj = {col: ds_jobs_transformed[col].value_counts(dropna=True) for col in object_cols}    
```


#### Loop through all object columns in value_counts_obj.
```py
    # Call the .info() method.
    for col, counts in value_counts_obj.items():
        print(f"\nColumn: {col}")
        print(counts)    
```
<img width="705" height="874" alt="ordering_applied" src="https://github.com/user-attachments/assets/67ea19b2-d25f-494c-a958-6436d7ccd9d4" />



#### Converting the "relevant_experience" column to a boolean.
```py
    # Converting to a boolean.
    ds_jobs_transformed['relevant_experience'] = ds_jobs_transformed['relevant_experience'].map({
        'Has relevant experience': True,
        'No relevant experience': False
    }).fillna(False)  


    # view the value count of the new column
    ds_jobs_transformed['relevant_experience'].value_counts()  
```

```py
    relevant_experience
    True     13792
    False     5366
    Name: count, dtype: int64    
```


```py
    # Get value counts for each object column
    for col in ds_jobs_transformed.select_dtypes(include=['int64', 'float64']).columns:
        if ds_jobs_transformed[col].dtype == 'int64':
            ds_jobs_transformed[col] = pd.to_numeric(ds_jobs_transformed[col], downcast='integer')
        elif ds_jobs_transformed[col].dtype == 'float64':
            ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('float32')    
```


```py
    # Select only object-type columns.
    object_cols = ds_jobs_transformed.select_dtypes(include='object').columns

    # Get value counts for each object column
    for col in object_cols:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('category')    
```
 

```py
    # Create ordered_categories.
    ordered_categories = {
        'education_level': ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'],
        'last_new_job': ['never', '1', '2', '3', '4', '>4'],
        'company_size': ['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'],
        'experience': ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'],
        'enrolled_university': ['no_enrollment', 'Part time course', 'Full time course']
    }

   # Loop through and apply the ordering
   for col, categories in ordered_categories.items():
       ds_jobs_transformed[col] = ds_jobs_transformed[col].cat.set_categories(categories, ordered=True)    
```

```py
    print(ds_jobs_transformed['education_level'].cat.categories,
          ds_jobs_transformed['last_new_job'].cat.categories,
          ds_jobs_transformed['company_size'].cat.categories,
          ds_jobs_transformed['experience'].cat.categories)    
```


```py
    ds_jobs_transformed['job_change'].value_counts()    
```

```py
    job_change
    0.0    14381
    1.0     4777
    Name: count, dtype: int64    
```




```py
    # Experience mapping
    experience_map = {
        '<1': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '10': 10,
        '11': 11,
        '12': 12,
        '13': 13,
        '14': 14,
        '15': 15,
        '16': 16,
        '17': 17,
        '18': 18,
        '19': 19,
        '20': 20,
        '>20': 21
     }

    # Company size mapping (using midpoint estimates)
    company_size_map = {
        '<10': 5,
        '10-49': 30,
        '50-99': 75,
        '100-499': 300,
        '500-999': 750,
        '1000-4999': 3000,
        '5000-9999': 7500,
        '10000+': 10000
    }


    ds_jobs_transformed['job_change'] = ds_jobs_transformed['job_change'].map({0.0: False, 1.0: True})    
```



```py
    # Convert categorical columns to string first
    ds_jobs_transformed['experience'] = ds_jobs_transformed['experience'].astype(str)
    ds_jobs_transformed['company_size'] = ds_jobs_transformed['company_size'].astype(str)


    # Apply mapping
    ds_jobs_transformed['experience_num'] = ds_jobs_transformed['experience'].map(experience_map).fillna(0)
    ds_jobs_transformed['company_size_num'] = ds_jobs_transformed['company_size'].map(company_size_map).fillna(0)    
```


```py
    ds_jobs_transformed = ds_jobs_transformed[
        (ds_jobs_transformed['experience_num'] >= 10) & 
        (ds_jobs_transformed['company_size_num'] >= 1000)
    ]
    ds_jobs_transformed.head()    
```
<img width="1006" height="206" alt="head2" src="https://github.com/user-attachments/assets/ecf08f78-78c0-4b89-a3de-1e8b35613201" />



```py
    ds_jobs_transformed.drop(columns=['experience_num', 'company_size_num'], inplace=True)

    # Check the shape of the transformed df
    ds_jobs_transformed.shape    
```
```py
    (2201, 14)
```


```py
    # Converting the types of specific columns
    ds_jobs_transformed['training_hours'] = ds_jobs_transformed['training_hours'].astype('int32')
    ds_jobs_transformed['experience'] = ds_jobs_transformed['training_hours'].astype('category')
    ds_jobs_transformed['company_size'] = ds_jobs_transformed['training_hours'].astype('category')    
```

```py
    # Converting two features to category type
    ds_jobs_transformed['experience'] = ds_jobs_transformed['experience'].cat.set_categories(['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'], ordered=True)

    ds_jobs_transformed['company_size'] = ds_jobs_transformed['company_size'].cat.set_categories(['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'], ordered=True)    
```


```py
    # Check the data types of each feature
    ds_jobs_transformed.dtypes    
```

<img width="280" height="248" alt="model-prepped_dataset" src="https://github.com/user-attachments/assets/e55c699c-82f0-465c-b9df-720df8fcd12e" />




### **Results and Findings**
#### Result Interpretation
The dataset has now been transformed to be well-suited for modeling.


### **Contributors & License**

**Created by:** Martin Unukpo

**Contributions:** We welcome contributions and suggestions to improve the dashboard.

### **Acknowledgments**

A heartfelt thanks to the DataCamp community and all the open-source resources that inspired this project. 

