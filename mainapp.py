import pandas as pd 
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # for creating wordclouds
from collections import Counter   #for counting objects
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure # to create a figure in matplotlib
from PIL import Image
import streamlit as st
from scipy.stats import chi2_contingency

# Images for the app:
image = Image.open('ab_test.png')

# Streamlit web.
st.set_page_config(page_title="A/B Test for MuscleHub",
        page_icon=("🤔"), layout="wide")

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()



st.title("A/B Test for MuscleHub.")
col1, col_mid, col2 = st.columns((1, 0.1, 1))
with col1:
    st.markdown("""
        Currently, when a MuscleHub visitor purchases a membership, they follow the following steps:\n
        - Take a fitness test with a personal trainer.
        - Fill out an application for the gym.
        - Send in their payment for their first month’s membership.
 """)
with col2:
    st.image(image)
st.write('\n')
st.markdown(""""We think that the fitness test intimidates some prospective members, so we has set up an A/B test.

**Visitors are randomly be assigned to one of two groups:**

**Group A** is still asked to take a fitness test with a personal trainer.\n
**Group B** skips the fitness test and proceed directly to the application.\n
The hypothesis is that visitors assigned to Group B will be more likely to eventually purchase a membership to MuscleHub than visitors assigned to Group A. So that the null and alternate hypotheses are as follows:

**Null Hypothesis** = There will no difference between the visitors in Group A that purchase membership and the visitors in Group B that purchase membership.\n
**Alternate Hypothesis** = There will be more visitors in Group B that will purchase membership than visitors in Group A that will purchase membership.\n
The significance threshold we will set as the benchmark to either accept or fail to reject the null hypothesis will be:

**𝛼 = 0.05**
""") 
st.markdown("""---""")  

st.subheader("Dataset")
st.write("Like most businesses, keeps they're data in a SQL database. I have already downloaded the data from her database to a csv file, and will load it using pandas to conduct A/B testing for the MuscleHub Gym.")
col1, col_mid, col2 = st.columns((1, 0.1, 1))
with col1:
    applications = pd.read_csv("applications.csv")
    st.write("Aplications")
    st.dataframe(applications.head())
    purchases = pd.read_csv("purchases.csv")
    st.write("Purchases")
    st.dataframe(purchases.head())
with col2:
    fitness_tests = pd.read_csv("fitness_tests.csv")
    st.write("Fitness tests")
    st.dataframe(fitness_tests.head())
    visits = pd.read_csv("visits.csv")
    st.write("Visits")
    st.dataframe(visits.head())
st.markdown("""---""")  
st.markdown("### Joining all the data")
st.markdown("""It would be helpful to have a single DataFrame with all of this data. \n
A DataFrame containing all of this data. Not all visits in **visits.csv** occurred during the A/B test, only data where `visit_date` is on or after 7-1-17.
""")
visits =  visits[visits["visit_date"] >= "7-1-17"]
df = visits.merge(fitness_tests, on=["first_name", "last_name", "email",  "gender"], how="left").merge(applications, on=["first_name", "last_name", "email", "gender"], how="left").merge(purchases, on=["first_name", "last_name", "email",  "gender"], how="left")
st.markdown("""---""") 
st.markdown("### Visualize the Groups")
st.markdown("Create new ab_test_group variable")
df["AB_test_group"] = df.fitness_test_date.apply(lambda x: "A" if  pd.notnull(x) else "B")
col1, col_mid, col2 = st.columns((1, 0.1, 1))
with col1:
    st.markdown("Obtain value counts of each group")
    st.write(df.AB_test_group.value_counts())
with col2:
    st.markdown("Obtain percentages of each group")
    st.write(df.AB_test_group.value_counts(normalize=True))

fig = px.pie(df, "AB_test_group", title="AB test group")
st.plotly_chart(fig, use_container_width=True)
st.markdown("""---""") 

st.markdown("### Count of applications")
st.markdown("""The sign-up process for MuscleHub has several steps:

1. Take a fitness test with a personal trainer (only Group A).
2. Fill out an application for the gym.
3. Send in their payment for their first month's membership.

Determining the percentage of people in each group who complete Step 2, filling out an application.""")

# Create is_application variable
df["is_application"] = df.application_date.apply(lambda x: "Application"  if pd.notnull(x) else "No_Application")
# Create new app_counts DataFrame
app_counts = df.groupby(["AB_test_group", "is_application"]).first_name.count().reset_index()
app_pivot = pd.pivot_table(app_counts, index="AB_test_group",
                    columns=["is_application"], values='first_name').reset_index()
# Create the total variable
app_pivot['Total'] = app_pivot.Application + app_pivot.No_Application
# Create the percent with application variable
app_pivot["Percent _with_Application"] = app_pivot.Application / app_pivot.Total
st.dataframe(app_pivot)

st.write("It looks like more people from Group B turned in an application.  Why might that be?\nWe need to know if this difference is statistically significant.")
st.markdown("""---""") 
st.markdown("### The statistical significance of applications")
st.markdown("Having calculated the difference in who turned in an application between groups, we should think if this difference is statistically significant.")
st.markdown("P-value")
contingency = [[250, 2254], [325, 2175]]
st.write(chi2_contingency(contingency))
st.write("A p-value of 0.00096 relative to a significance threshold of 0.05 indicates that there is a statistically significant difference between the two groups.")
st.markdown("""---""") 
st.markdown("### Count of memberships from applications")
st.write("Of those who picked up an application, how many purchased a membership?\nDetermine how many potential customers purchased a membership out of those that picked up an application.")
# Create an is_member variable
df["is_member"] = df.purchase_date.apply(lambda x: "member" if pd.notnull(x) else "not_member")
# Create the just_apps DataFrame
just_apps = df[df.is_application == "Application"]
# Create member_count DataFrame
member_count = just_apps.groupby(["AB_test_group", "is_member"]).first_name.count().reset_index()
# Pivot member_count
member_pivot = pd.pivot_table(member_count, index="AB_test_group", columns=["is_member"], values="first_name").reset_index()
# Create the Total variable
member_pivot["total"] = member_pivot.member + member_pivot.not_member
# Create the Percent Purchase variable
member_pivot["percent_purchase"] = member_pivot.member / member_pivot.total
st.write(member_pivot)
st.write("It looks like people who took the fitness test were more likely to purchase a membership if they picked up an application.  Why might that be?")
st.markdown("""---""") 

st.markdown("### The statistical significance of memberships")
st.markdown("""Calculate if the difference between the following groups is statistically significant: \n
- The customers that picked up an application and took a fitness test.\n
- The customers that did not take a fitness test and picked up an application.""")
# Calculate the p-value
contingency = [[200, 50], [250, 75]]
st.write(chi2_contingency(contingency))
st.write("A p-value of 0.432 relative to a significance threshold of 0.05 does not reflect a statistically significant difference between the two groups, and would lead us to fail to reject the null hypothesis.")
st.markdown("""---""") 

st.markdown("### Count of all memberships")
st.write("Previously, we looked at what percentage of people who picked up applications purchased memberships. \nWhat percentage of ALL visitors purchased memberships.?")
# Create final_member_count DataFrame
all_memberships = df.groupby(['AB_test_group', 'is_member']).first_name.count().reset_index()
# Pivot final_member_count
all_memberships_pivot = all_memberships.pivot(index="AB_test_group", columns="is_member", values="first_name").reset_index()
# Create the Total variable
all_memberships_pivot["total"] = all_memberships_pivot.member + all_memberships_pivot.not_member
# Create the Percent Purchase variable
all_memberships_pivot["total_percent"] = all_memberships_pivot.member / all_memberships_pivot.total
st.write(all_memberships_pivot)
st.write("Previously, when we only considered people who had already picked up an application, we saw that there was no significant difference in membership between Group A and Group B.\nNow, when we consider all people who visit MuscleHub, we see that there might be a significant difference in memberships between Group A and Group B.")
st.markdown("""---""") 

st.markdown("### The statistical significance between groups")
st.write("Determine if there is a significant difference in memberships between Group A and Group B.")
# Calculate the p-value
contingency = [[200, 2304], [250, 2250]]
st.write(chi2_contingency(contingency))
st.write("A p-value of 0.0147 relative to a significance threshold of 0.05 indicates that there is a statistically significant difference between the two groups. This informs us that we should not reject the hypothesis that visitors assigned to Group B will be more likely to eventually purchase a membership to MuscleHub than visitors assigned to Group A.\nHowever, it is important to note that when assessing the groups among those customers that filled out an application, those that completed a fitness test (Group A), were more likely to make a purchase than those customers that did not complete a fitness test (Group B).")
st.markdown("""---""")  
st.markdown("### Visualize the results")
st.markdown("""The difference between Group A (people who were given the fitness test) and Group B (people who were not given the fitness test) at each state of the process:\n

- Percent of visitors who apply.
- Percent of applicants who purchase a membership.
- Percent of visitors who purchase a membership.""")

fig = px.bar(app_pivot, y="Percent _with_Application", x="AB_test_group", color="AB_test_group", title='Percent of visitors who apply')
for data in fig['data']:
        data.width = 0.5
fig.update_layout(title_x=0.5, yaxis = dict(tickmode = 'array', tickvals = [0, 0.05, 0.10, 0.15, 0.20], ticktext = ['0%', '5%', '10%', '15%', '20%']), xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 1],
        ticktext =  ["Fitness Test", "No Fitness Test"]), yaxis_title=None, xaxis_title=None)
st.plotly_chart(fig, use_container_width=True)
st.markdown("""---""")  
fig = px.bar(member_pivot, y="percent_purchase", x="AB_test_group", color="AB_test_group", title="Percent of applicants who purchase a membership.")
for data in fig['data']:
        data.width = 0.5
fig.update_layout(title_x=0.5, yaxis = dict(tickmode = 'array', tickvals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], ticktext = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']),  xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 1],
        ticktext =  ["Fitness Test", "No Fitness Test"]), yaxis_title=None, xaxis_title=None)
st.plotly_chart(fig, use_container_width=True)
st.markdown("""---""")  
fig = px.bar(all_memberships_pivot, y="total_percent", x="AB_test_group", color="AB_test_group", title='Percent of visitors who purchase a membership.')
for data in fig['data']:
        data.width = 0.5
fig.update_layout(title_x=0.5, yaxis = dict(tickmode = 'array', tickvals = [0, 0.05, 0.10, 0.15, 0.20], ticktext = ['0%', '5%', '10%', '15%', '20%']), xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 1],
        ticktext =  ["Fitness Test", "No Fitness Test"]), yaxis_title=None, xaxis_title=None)
st.plotly_chart(fig, use_container_width=True)
st.markdown("""---""")  

st.markdown("#### Creating a [wordcloud](https://pypi.org/project/wordcloud/) visualization that can be use to create an ad for the MuscleHub Gym with the data in `interviews.txt`. ")
# Open and read the interviews.txt file
interviews = open(r"interviews.txt", encoding='utf8')
txtContent = interviews.read()
st.write("The Content of text file is : ", txtContent)
# Print the length of the new string
st.write('There are {} words in the total interviews.txt file.'.format(len(txtContent)))
# Create a wordcloud object
wordcloud = WordCloud(width=2500, height=1250).generate(txtContent)

# Display the wordcloud with MatplotLib and save figure
fig, ax = plt.subplots()
figure(num=None, figsize=(20, 16), facecolor='w', edgecolor='k')
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.markdown("#### Thats all for this project!\n #### Thanks for yor time!!!👋")
