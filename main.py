import os.path

import pandas as pd
import zipfile
import re
import nltk
from nltk.corpus import stopwords

# Step 1: Data Preprocessing
# Extracting the text data from the given corpus
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

# Reading the contents of each file into a DataFrame
files = []
files_names=os.listdir("dataset/dataset")
for name in files_names:

    with open(f"dataset/dataset/{name}", 'r') as f:
        files.append(f.read())
df = pd.DataFrame({'text': files})

# Cleaning and preprocessing the text data
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = [word for word in text.split() if word not in stop_words]
    return ' '.join(text)
df['clean_text'] = df['text'].apply(clean_text)

# Step 2: Creating Controlled Vocabulary
# Creating a list of relevant terms and concepts
covid_terms = ['covid', 'coronavirus']
property_terms = ['property', 'real estate']
vtl_terms = ['vaccine', 'therapeutic', 'treatment', 'cure']
omicron_terms = ['omicron', 'variant']
not_omicron_terms = ['covid', 'coronavirus', '-omicron']

# Creating a list of synonyms and related terms for each concept
covid_synonyms = ['sars-cov-2', 'covid19', 'covid-19']
property_synonyms = ['housing', 'mortgage', 'rental']
vtl_synonyms = ['vaccination', 'immunization', 'remdesivir', 'monoclonal antibody']
omicron_synonyms = ['b.1.1.529', 'new variant', 'variant of concern']
not_omicron_synonyms = []

# Assigning unique identifiers to each term in our controlled vocabulary
covid_codes = ['CV' + str(i+1) for i in range(len(covid_terms))]
property_codes = ['PR' + str(i+1) for i in range(len(property_terms))]
vtl_codes = ['VL' + str(i+1) for i in range(len(vtl_terms))]
omicron_codes = ['OM' + str(i+1) for i in range(len(omicron_terms))]
not_omicron_codes = ['NO' + str(i+1) for i in range(len(not_omicron_terms))]

# Creating a dictionary to store our controlled vocabulary
controlled_vocab = {}
for term, synonyms, codes in zip([covid_terms, property_terms, vtl_terms, omicron_terms, not_omicron_terms],
                                 [covid_synonyms, property_synonyms, vtl_synonyms, omicron_synonyms, not_omicron_synonyms],
                                 [covid_codes, property_codes, vtl_codes, omicron_codes, not_omicron_codes]):
    for t, s, c in zip(term, synonyms, codes):
        controlled_vocab[t] = {'synonyms': [t] + ([s] if isinstance(s, str) else s), 'codes': [c]}

# Step 3: Creating a Free Text Search Engine
# Defining a function to search for keywords and phrases within the text data
def free_text_search(query, df):
    results = df[df['clean_text'].str.contains(query)]
    return results

# Step 4: Combining Controlled Vocabulary and Free Text Search
def search_documents(query, df, controlled_vocab):
    # Splitting the query into individual terms
    query = " ".join(query)
    query_terms = query.lower().split()
    # Identifying which concepts are present in the query
    concepts_present = {}
    for term in query_terms:
        for concept, values in controlled_vocab.items():
            if term in values['synonyms']:
                if concept not in concepts_present:
                    concepts_present[concept] = []
                concepts_present[concept].append(term)

    # Generating a list of search terms for each concept present in the query
    search_terms = []
    for concept, terms in concepts_present.items():
        codes = controlled_vocab[concept]['codes']
        for code in codes:
            search_terms.extend([f'{code}:{term}' for term in terms])

    # Performing free text search using the generated search terms
    if search_terms:
        query = ' '.join(search_terms)
        results = free_text_search(query, df)
    else:
        results = pd.DataFrame()

    # Adding an extra column to the results DataFrame to indicate which concepts were present in the query
    for concept in concepts_present:
        results[concept] = True

    return results


query = 'Covid-19 and Property'
results = search_documents(query, df, controlled_vocab)
print(f"Found {len(results)} documents matching the query '{query}':")
print(results[['covid-19']])

