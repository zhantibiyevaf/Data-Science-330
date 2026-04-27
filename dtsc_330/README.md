# Data-Science-330
Homework 2 Answers:

1. How do you fill in the missing dates from the grants data?
    In class we renamed the 'BUDGET_START' to 'start_at'. I then checked how many values mere missing in that columns. Since 'applicaiton_id' is unique, every row represents one grant. The only way to fill the missing values is to check if the original .csv file has other columns that represent date. There was a column names 'PROJECT_START' which has dates for when the project has started, and I thought that I can fill out the missing values in 'start_at' using values from that column. After filling those values, I rechecked the NA count to confirm that the dates were mostly resolved. There were also 3 missing dates in project_start, but I decided to remove them because most of the other columns related to that row were empty (for example, there was no country, city, etc.) 
2. PI_NAMEs contains multiple names. We can only connect individual people. Can you make it so that we can get "grantees"?
    The 'pi_names' column has multiple names in one cell that are separated by semicolons ";". Using str.split(';'), and removing the unnecessary words like '(contact)', I splitted each cell into a list of names. Then I used explode to create more rows with individual pi_names. Since the application_id is unique, it would be easy to find pi's related to the same project. 
3. The dates for Articles are problematic. Can you fix them?
    Article dates in are problematic in this .xml file because there are different types of dates for one single publication (date completed, date revised, publication date), which also have months either in like numbers of as a string. Some dates don't have days, while others might have all three: year, month, and date. I think to make it easier to work with this file, we should only choose one type of date to focus on and keep for every publication. I decided to focus on publication date, but then noticed that there are two different types of that: electronic and the journal issue code. My idea was to use the electronic date first and go to journal issue dates if needed. However, I was not fully able to implement it in code. 

Homework 3:

Performance:
I used a random_forest model with features [hr_mean, hr_std, acc_mag_mean, acc_mag_std], and it achieved a score of 0.948979 on the test set.

I first grouped the raw heart rate and accelerometer data into 1-minute windows and then computed the mean and standard deviation of heart rate and acceleration magnitude for each window. I used acceleration magnitude because it gives a single measure of overall movement, regardless of direction. Including both the mean and standard deviation helps capture the general level of activity as well as how variable it is within a minute. This makes sense for sleep classification since sleep typically involves less movement and more stable heart rate compared to being awake.

Homework 4:

1. XGBoost achieved 0.9370 accuracy compared to Random Forest at 0.9452, so XGBoost performed worse by 0.0082.

Homework 5:

The simplest idea is to first block both phone books using a quick key, like last name plus first initial, and only compare people inside the same block. That way we don’t have to do an all-to-all comparison, because entries from different blocks are ignored from the start. We can narrow it down even more by requiring something exact like the same phone number or ZIP code before we even look closely at the names or address details.

Homework 9:

In the website, I trained a model that predicts whether an image contains a fork or a key using a small number of labeled images for each class, with different angles and positions. The model worked well on clear images that looked similar to the training data, but struggled when the object was harder to recognize, such as when the key was at a weird angle or not clearly visible, and in those cases it predicted the wrong class. I did upload around 20 images of each object to the training data. I think that even with a small dataset, it can still perform well because it uses a model that already captures visual features like shapes and edges. It only needs a small amount of data to learn the differences between classes.

Homework 13:

For this homework, I built a spelling correction model using TensorFlow. The goal was to take a misspelled word as input and predict the correct version of that word. I used Generative AI for coding.

To create the dataset, I started with a list of correct words and then simulated typos by randomly deleting, inserting, swapping, or replacing characters. This allowed me to generate a larger dataset without manually labeling everything.

This problem is a sequence-to-sequence (seq2seq) problem, because the input is a sequence of characters and the output is another sequence of characters. For example:

- tonue → tongue  
- calendfar → calendar  
- sucessful → successful  

Instead of predicting a single label like in classification, the model has to predict an entire sequence of characters. This is similar to translation, but here it's translating it from a “misspelled word → correct word”.

The model works by looking at each word letter by letter. First, it turns each letter into a number so the computer can understand it. Then it tries to learn patterns in how misspelled words should be fixed. For example, it might learn that "helo" is missing an "l" or that letters are sometimes swapped. After training, it uses these patterns to predict the corrected word one letter at a time.

I think a more advanced approach would be to use a transformer-based model, which uses attention to better understand relationships between characters. This would definitely improve performance, especially on harder spelling errors, or if there are two or more letters misspelled. For this project, I chose a simpler model so I could focus on understanding how the sequence-to-sequence mapping works before moving to more complex architectures.