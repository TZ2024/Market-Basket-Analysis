# Import necessary libraries
import streamlit as st 
import mlxtend
import plotly

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#add side for uploading file
uploaded_file = st.sidebar.file_uploader("Upload a file")

#default dataset: read the dataset
transactions_df = pd.read_csv('my_transactions.csv')#change the path to the dataset


if uploaded_file is not None:
    transactions_df = pd.read_csv(uploaded_file)
    st.write(transactions_df)
    st.write(transactions_df.columns)
    

#put a download button
st.sidebar.markdown('Download the template') 
st.sidebar.markdown('[The template](https://github.com/TZ2024/Market-Basket-Analysis/blob/main/my_transactions.csv)')#update the link to the dataset
      

#add title to the app
st.title('Market Basket Analysis')


#create pages tabs
tab_intro,tab_encoded,tab_freq,tab_s_value,tab_associa,tab_filter = st.tabs(['Introduction','The encoded dataset','The frequent itemsets','The support value','The association rules','Filter functions'])



    


with tab_intro:
    st.header("Introduction")
    st.markdown("""**Welcome to the Market Basket Analysis Learning App!** 🎉

**What is Market Basket Analysis?**  
Market Basket Analysis is a technique used by businesses to discover which products are frequently purchased together. For example, a grocery store might find that bread and butter are often bought in the same trip. Knowing this, they could place those items closer or run combined promotions.

**How to Use this App:**  
- On the left, you can **upload your own transactions dataset** (or use the default example dataset already loaded).  
- This app will then walk you through the process of analyzing the data to find interesting patterns.  
- Use the tabs above to navigate:
  1. **Introduction:** Overview of the analysis and data.  
  2. **Encoded Dataset:** How raw purchase data is prepared for analysis.  
  3. **Frequent Itemsets:** Item combinations that appear together often.  
  4. **Support Values:** How common those combinations are (their frequency).  
  5. **Association Rules:** “If-then” insights (e.g., "if X is bought, Y is likely bought").  
  6. **Filter Functions:** Explore specific items or adjust thresholds for deeper analysis.

Feel free to just observe the example data or upload your own CSV file to see results for your business case. Let's get started! 🎯

*(Below is a preview of the transaction dataset we'll analyze.)*
""")
    # Display a preview of the dataset (first 5 rows) to give an idea of the data
    st.write("Sample of the transactions dataset:", transactions_df.head())


with tab_encoded:
    st.header("The Encoded Dataset")
    st.markdown("""**Data Encoding - Preparing Data for Analysis**  
In the original data, each transaction is a list of items. To analyze it, we need to convert that into a table of **True/False values** (also seen as 1/0). This process is called *one-hot encoding*. Each row of the table is a transaction, and each column is an item. A value of **True (1)** means the item was purchased in that transaction, and **False (0)** means it was not.""")

    st.markdown("The encoded dataset is used to find frequent item combinations using the Apriori algorithm.", help="Apriori is a classic algorithm for finding frequent itemsets in data mining.")

    # Convert the 'Items' column (raw transactions) into a list of item lists
    transactions = transactions_df['Items'].apply(lambda x: x.split(','))
    # Encode the transactions into a boolean matrix
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transactions_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    # Display the encoded dataset (show a subset if it's too large)
    st.write("Encoded transaction dataset (True = item present, False = item absent):")
    st.dataframe(transactions_encoded.head(10))  # showing first 10 rows as an example


with tab_freq:

    st.header("The Frequent Itemsets")
    st.markdown("""**Frequent Itemsets - Which Items Often Appear Together?**  
“Frequent itemsets” are combinations of items that appear together in many transactions. Here, we've used a minimum support threshold of **40%**. This means we're only looking at item combinations that appear in **at least 40% of all transactions**.

Why 40%? We chose it as an example threshold - it's fairly high, so we focus on the most common patterns. (In practice, analysts might adjust this number to find more or fewer itemsets.) Below you'll see all the itemsets that met this 40% cutoff, along with their **support** value (the percentage of transactions containing that combination).

*For example, if you see an itemset with support 0.6, that means 60% of the transactions include that combination of items.*
""")
    # Find frequent itemsets with minimum support of 0.4 (40%)
    frequent_itemsets = apriori(transactions_encoded, min_support=0.4, use_colnames=True)
    # Sort itemsets by support in descending order for easier reading
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    # Convert itemsets from Python frozenset to a list of item names (for readability)
    frequent_itemsets['Items'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    frequent_itemsets_display = frequent_itemsets[['Items', 'support']].reset_index(drop=True)
    frequent_itemsets_display['support'] = (frequent_itemsets_display['support']*100).round(1).astype(str) + '%'
    # Display the frequent itemsets in a friendly format
    st.markdown("Frequent itemsets (with support ≥ 40% of transactions) [:question:](#### 'Support means the % of transactions containing the items')", unsafe_allow_html=True)
    st.dataframe(frequent_itemsets_display)


with tab_s_value:
    st.header("Support Value Visualization")
    st.markdown("""**Visualizing Support - How Common Are These Itemsets?**  
Now that we have the frequent itemsets, let's look at how their support values are distributed. Below, the **line chart** plots each frequent itemset's support value (on the y-axis). The **histogram** shows how many itemsets fall into different support ranges.

What can we learn from these graphs? If most itemsets just barely passed the 40% support threshold, you'll see many bars near the 0.4 mark in the histogram. If some itemsets are extremely common, you'll see points higher up on the line chart. This visualization helps illustrate the concept of support: a higher line or a bar far to the right means that combination of items appears in more transactions.

*In summary, these charts give you a sense of which item combinations are very frequent versus just moderately frequent.*
""")
    # Create a line chart for support values
    support_values = frequent_itemsets['support'].values
    line_fig = plotly.graph_objs.Figure()
    line_fig.add_trace(plotly.graph_objs.Scatter(x=list(range(len(support_values))), 
                                                 y=support_values, mode='lines+markers', name='Support'))
    line_fig.update_layout(title="Support Values for Frequent Itemsets",
                           xaxis_title="Itemset Index (each point is a frequent itemset)",
                           yaxis_title="Support (fraction of transactions)")
    st.plotly_chart(line_fig)
    
    # Create a histogram for support values distribution
    hist_fig = plotly.graph_objs.Figure()
    hist_fig.add_trace(plotly.graph_objs.Histogram(x=support_values, nbinsx=10, name='Support'))
    hist_fig.update_layout(title="Distribution of Support Values",
                           xaxis_title="Support Range",
                           yaxis_title="Number of Itemsets",
                           bargap=0.1)
    st.plotly_chart(hist_fig)


with tab_associa:
    st.header("Association Rules")
    st.markdown("""**Association Rules - “If X, then Y” Patterns**  
Association rules reveal relationships in the form **IF (antecedent) THEN (consequent)**. In other words, if a transaction contains the antecedent item(s), it is likely to contain the consequent item(s) as well.

For example, a rule could be: **IF Bread THEN Butter**. This would mean whenever Bread is purchased, Butter is also purchased in that transaction a significant portion of the time.

We use three metrics to evaluate each rule:  
- **Support:** The fraction of all transactions that contain *both* the antecedent and the consequent.  
- **Confidence:** Given that the antecedent occurs, the probability that the consequent also occurs. (For the Bread ⇒ Butter rule, if confidence = 0.8, it means 80% of the transactions that have Bread also have Butter.)  
- **Lift:** How much more likely the consequent is purchased when the antecedent is present, compared to random chance. (If lift = 1.2, it means Butter is 1.2 times more likely to be bought when Bread is in the basket, compared to Butter being bought regardless of Bread.)

Below is a list of association rules found (using a minimum confidence of 50%). The table shows the antecedent ⇒ consequent, along with each rule's support, confidence, and lift. After the unsorted list, we show the same rules sorted by **lift** so that the strongest (most “interesting”) rules are at the top.
""")
    # Generate association rules from the frequent itemsets with min confidence 0.5 (50%)
    rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
    # Format antecedents and consequents as readable strings
    rules_df['Antecedent'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_df['Consequent'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
    # Select relevant columns and round confidence and lift for display
    rules_display = rules_df[['Antecedent', 'Consequent', 'support', 'confidence', 'lift']].copy()
    rules_display['confidence'] = rules_display['confidence'].round(2)
    rules_display['lift'] = rules_display['lift'].round(2)
    # Display the unsorted association rules
    st.write("Association rules (min confidence = 50%):")
    st.dataframe(rules_display.reset_index(drop=True))
    # Sort rules by highest lift to highlight the strongest associations
    rules_by_lift = rules_display.sort_values(by='lift', ascending=False).reset_index(drop=True)
    st.write("Association rules sorted by lift (strength of association):")
    st.dataframe(rules_by_lift)


with tab_filter:
    st.header("Filter Functions and Custom Rules")
    st.markdown("""**Interactive Exploration - Filter Rules by Item or Confidence**  
This section lets you interact with the association rules:

- **Filter by Item:** Select a product from the dropdown menu to see only the rules where that product is on the IF side of the rule. (For example, choose "Bread" to find rules that start with Bread ⇒ ... )
- **Adjust Confidence Threshold:** Use the slider to change the minimum confidence. A higher value will show fewer, stronger rules; a lower value will include more rules (even weaker ones).

Try it out: pick an item you're interested in and see what rules involve it. You can also raise or lower the confidence threshold to see how the rule list changes. This helps you understand how stricter or looser criteria affect the associations we find.
""")
    # Dropdown to select an item for filtering antecedents
    all_items = sorted({item for items in rules_df['antecedents'] for item in items})
    # Create a selectbox for the items
    selected_item = st.selectbox(
    'Select an item to filter rules (item in IF part of rule):', 
    all_items, 
    help="Filters to rules where this item is in the antecedent (the IF part of the rule)."
)

    # Filter the rules dataframe to those where the selected item is in the antecedent set
    filtered_by_item = rules_display[rules_display['Antecedent'].str.contains(selected_item)]
    st.write(f"Rules where **{selected_item}** is in the antecedent:")
    st.dataframe(filtered_by_item.reset_index(drop=True))
    
    # Slider to adjust confidence threshold
    min_conf = st.slider(
    'Select minimum confidence:', 
    0.0, 1.0, 0.5, 0.1, 
    help="Lower this to see more rules (including weaker ones), or raise it to see only very strong rules."
)
    # Display the selected confidence threshold
    st.write(f"Showing rules with confidence ≥ {min_conf:.1f}:")
    # Recompute or filter rules based on new confidence threshold
    new_rules_df = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_conf)
    new_rules_df['Antecedent'] = new_rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    new_rules_df['Consequent'] = new_rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
    new_rules_display = new_rules_df[['Antecedent', 'Consequent', 'support', 'confidence', 'lift']].copy()
    new_rules_display['confidence'] = new_rules_display['confidence'].round(2)
    new_rules_display['lift'] = new_rules_display['lift'].round(2)
    st.dataframe(new_rules_display.reset_index(drop=True))
