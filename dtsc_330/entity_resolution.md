Homework 8:

What is entity resolution?

Definition: Entity resolution project is a pipeline that connects information either within the dataset or between the databases by using machine learning. 

Goal: The goal of the project is to identify dublicate or matching records without comparing every pair by hand.

Why it matters: It matters because sometimes there might be misspellings or shortened versions of words in the dataset that relate to the same thing, but if we don't identify it and combine, it will be treated as separate data. 

Entity Resolution Pipeline

	•   Readers: loads data and generates record pairs that will later be evaluated by the model.
	•	Features: creates similarity features between pairs of records such as name similarity, affiliation, and distance.
    •   Reusable Classifier: used to train and predict whether two records match
	•	Training and testing: is the code that runs full pipeline including feature creation, training the model, predicting and evaluating it.

Connecting Each Week's Classwork to Entity Resolution

Week 1:
	•	What we learned: about Github, documentation, and structure; we also briefly talked about features and labels.
	•	Connection to entity resolution: we need features and labels in the entity resolution pipeline.

Week 2: 
	•	What we learned: we worked with two different datasets, and we did data wrangling & merging.
	•	Connection to entity resolution: entitiy resolution depends on clean data, which is why we need to clean the mess in the data in order to work with it.

Week 3: 
	•	What we learned: training classifiers, predicting labels from features, reusable classifier
	•	Connection to entity resolution: in entity resolution model learns patterns from the data.

Week 4: 
	•	What we learned: decision trees, random forest, XGBoost
	•	Connection to entity resolution: we use these models to decide matches in the dataset. 

Week 5: 
	•	What we learned: bias in data, creating features, and simulating training data. 
	•	Connection to entity resolution: it depends on good features, and it also requires training the data. 

Week 6:
	•	What we learned: reducing comparisons and cross-product datasets, also we talked about the blocking problem.
	•	Connection to entity resolution: blocking is important in entity resolution, because it helps to avoid comparing every record with every other record. 

Week 7:
    •	What we learned: creating databases, SQL, creating tables.
	•	Connection to entity resolution: databases are a good place to store data, and entity resolution often works on large datasets. 

Human Activity Recognition Dataset: it was about predicting human activity from sensor data. The core concept we learned from that is that models learn from features to predict labels. It relates to entitiy resolution because instead of predicting "awake vs. asleep", we're predicting "same person vs. different person."
