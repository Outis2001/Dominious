import re
import openai
import os
from dotenv import load_dotenv,find_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import time
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
import requests
import json

load_dotenv(find_dotenv())
api_key=os.environ.get("OPEN_API_KEY")
llm = OpenAI(api_key=api_key,temperature=0.7)
pine_cone_api_key=os.environ.get("PINE_CONE_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
pc = Pinecone(api_key=pine_cone_api_key)

embeddings=OpenAIEmbeddings(openai_api_key=api_key)
index_name = "dominious"  # change if desired
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

API_URL = "https://t7cpt5aki2ddo8ox.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}



if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


def RAG(user_query):
    results = vector_store.similarity_search(query=user_query,k=20)
    categories = []
    domain_names = []

    for doc in results:
        # Save the category
        categories.append(doc.metadata['category'])

        # Get the raw domain_names field
        raw = doc.metadata['domain_names']

        # Join into one string (if it's a broken list of strings)
        combined = ' '.join(raw)

        # Use regex to find words (filter out extra characters)
        names = re.findall(r"[A-Za-z][A-Za-z0-9]+", combined)

        # Append as a sublist
        domain_names.append(names)

    domain_names = [item for sublist in domain_names for item in sublist]
    short=domain_names[:15]
    result = ','.join(short)
    print("----------------------------Categories----------------------------")
    print(categories)
    print("----------------------------Domain_names----------------------------")
    print(domain_names)
    return result

def preprocess(user_query):
    pass

def generate_domains(user_description: str, sample_domains: str) -> str:
    print("----------------------------User_query----------------------------")
    print(user_description)
    print("----------------------------Sample_domains----------------------------")
    print(sample_domains)
    prompt = f"""
        You are an expert domain name generator. Your task is to create domain name suggestions that closely match the user's input and follow the style and pattern of the sample domain names provided.

        User Input Description:
        "{user_description}"

        Sample Domain Names:
        {sample_domains}

        Instructions:
        - Generate 10 to 15 domain names that fit the user's input description.
        - The names should be short, easy to understand, creative, memorable, and relevant to the input.
        - Use similar word structures and language style as the samples.
        - Avoid overly long or complicated names; keep them concise and simple.
        - Do not repeat exact sample names.
        - Provide only the domain name suggestions without any domain extensions (like .com, .net, .lk).
        - Provide the domain names in a numbered list .

        Suggested Domain Names:
"""
    prompt_1 = f"""
You are "The Brand Architect," a highly innovative and meticulous domain name specialist known for crafting memorable, impactful, and SEO-friendly brand identities. Your goal is to translate abstract user desires into concrete, compelling domain name suggestions.

User Input Description:
"{user_description}"

Sample Domain Names (Analyze these for underlying patterns, wordplay, tone, and brevity):
{sample_domains}

Instructions for The Brand Architect:
- Generate 12 unique domain names.
- Each name must be a perfect blend of relevance, creativity, and conciseness (ideally 1-2 words, max 3).
- Prioritize names that are easy to spell and pronounce.
- Consider modern, trendy, or classic styles as inspired by the samples and user description.
- Explore concepts like:
    - Keywords + action verbs (e.g., "TechFlow")
    - Portmanteaus (blending two words)
    - Evocative imagery or concepts
    - Short, catchy phrases
- Ensure no exact repetition of sample names or overly generic terms.
- Focus on the core essence of the user's need.
- Present the names as a numbered list.

Suggested Domain Names by The Brand Architect:
"""
    prompt_2 = f"""
You are a precision domain name optimizer. Your mission is to generate highly relevant and available-sounding domain names based on user input, strictly adhering to constraints.

User Input Description:
"{user_description}"

Sample Domain Names (Observe their structure, length, and keyword usage):
{sample_domains}

Instructions:
- Provide 10-15 domain name suggestions.
- **Strictly avoid:** hyphens, numbers, obscure abbreviations, misspellings (unless explicitly part of a stylistic pattern in samples), or names longer than 15 characters.
- Each name must be brandable, short, and directly or indirectly related to the user description.
- Prioritize single-word or two-word combinations.
- Do not include any top-level domains (e.g., .com, .net).
- Present the names in a numbered list.

Optimal Domain Names:
"""
    prompt_3 = f"""
You are a semantic domain name expert. Your task is to generate highly optimized domain names by intelligently incorporating keywords and their synonyms, while maintaining brand appeal.

User Input Description:
"{user_description}"

Core Keywords (identified from user description): [e.g., "health", "wellness", "fitness"]
Related Concepts/Synonyms: [e.g., "vitality", "wellbeing", "strength", "active"]

Sample Domain Names (Analyze how keywords are used and combined):
{sample_domains}

Instructions:
- Generate 10-15 domain names.
- Integrate the core keywords or their relevant synonyms creatively.
- Focus on short, impactful names.
- Explore combinations like:
    - Keyword + benefit
    - Synonym + industry term
    - Abstract concept + direct keyword
- Ensure names are unique and not direct repetitions of samples.
- Do not include domain extensions.
- Provide a numbered list.

Keyword-Rich Domain Names:
"""
    prompt_4 = f"""
You are a master of stylistic domain name generation. Your task is to create domain names that perfectly emulate the stylistic nuances (e.g., modern, playful, elegant, minimalist) found in the provided sample domains, while aligning with the user's description.

User Input Description:
"{user_description}"

Sample Domain Names (Analyze their specific style, tone, and common linguistic traits):
{sample_domains}

Instructions:
- Generate 10-15 domain names.
- Each suggestion should strongly reflect the overall *style and tone* observed in the sample domain names.
- Focus on brevity, ease of recall, and brandability.
- Consider rhythmic qualities and phonetic appeal.
- Avoid generic or boilerplate names.
- Provide only the domain names in a numbered list.

Stylistic Domain Name Suggestions:
"""
    prompt_5 = f"""
You are a domain name strategist specializing in audience-centric branding. Generate domain names specifically tailored to resonate with the described target audience, reflecting their preferences and understanding.

User Input Description:
"{user_description}"

Target Audience Profile:
[e.g., "Young tech enthusiasts, aged 18-30, valuing innovation and community."]
(You'd extract/generate this from your RAG system or user input.)

Sample Domain Names (Consider how these might appeal to specific demographics):
{sample_domains}

Instructions:
- Generate 10-15 domain names.
- Names should appeal directly to the specified target audience. Consider their language, values, and online behavior.
- Names should be catchy, memorable, and reflective of the user's input.
- Keep names concise (1-3 words).
- Ensure uniqueness and relevance.
- Provide a numbered list.

Audience-Tailored Domain Names:
"""
    prompt_6 = f"""
You are a logical domain name architect. Your process involves breaking down complex user needs into core concepts and then building intuitive domain names.

User Input Description:
"{user_description}"

Sample Domain Names (Observe how complex ideas are simplified):
{sample_domains}

Thinking Process:
1.  Identify the core purpose/service: [Gemma will infer this from user_description]
2.  Extract key benefits or outcomes: [Gemma will infer this]
3.  Brainstorm related metaphorical or abstract terms: [Gemma will infer this]
4.  Combine these elements concisely to form domain name candidates.

Instructions:
- Generate 10-15 domain names.
- Each name should be a result of a thoughtful combination of core concepts.
- Aim for clarity, brevity, and memorability.
- Avoid hyphenation or numbers.
- Provide only the domain names in a numbered list.

Generated Domain Names (following thought process):
"""
    prompt_7 = f"""
You are a creative domain name ideator, capable of exploring diverse semantic categories to find the perfect name.

User Input Description:
"{user_description}"

Semantic Categories to Explore (based on user description):
-   **Direct/Functional:** Names that clearly state what it is.
-   **Abstract/Conceptual:** Names that evoke an idea or feeling.
-   **Action-Oriented:** Names that imply doing something.
-   **Compound Words:** Blending two existing words.
-   **Portmanteau/Invented:** Blending parts of words or creating new words.

Sample Domain Names (Identify which categories they fall into):
{sample_domains}

Instructions:
- Generate 10-15 domain names, ensuring a mix across the semantic categories where appropriate.
- Focus on short, easy-to-remember, and brandable names.
- Do not repeat sample names.
- Provide names as a numbered list.

Categorized Domain Names:
"""
    prompt_8 = f"""
You are a poetic domain name visionary. Your strength lies in transforming user descriptions into evocative, metaphorical, and abstract domain names that resonate deeply.

User Input Description:
"{user_description}"

Sample Domain Names (Analyze their metaphorical depth or abstract qualities):
{sample_domains}

Instructions:
- Generate 10-15 domain names.
- Each name should be a metaphor, an abstract concept, or evoke a strong image related to the user's input.
- Names should be short, unique, and highly memorable.
- Avoid overly literal or common terms.
- Focus on brandability and future scalability.
- Provide a numbered list.

Metaphorical Domain Names:
"""
    prompt_9 = f"""
You are a strategic domain name generator, focused on highlighting the unique selling proposition (USP) of a business or idea within the domain name itself.

User Input Description:
"{user_description}"

Unique Value Proposition / Core Benefit:
[e.g., "The fastest delivery service," "AI-powered personalized learning," "Sustainable fashion solutions."]
(You'd derive this from the user_description or an additional input field.)

Sample Domain Names (See how they hint at value or benefit):
{sample_domains}

Instructions:
- Generate 10-15 domain names.
- Each name must subtly or directly hint at the unique value proposition or core benefit.
- Names should be concise, impactful, and easy to recall.
- Prioritize names that evoke curiosity or promise a benefit.
- Do not repeat exact sample names.
- Provide a numbered list.

Value-Driven Domain Names:
"""
    prompt_10 = f"""
You are a phonetic domain name artist. You craft names that sound pleasant, are easy to pronounce, and have a specific flow, adhering to strict phonetic and syllable constraints.

User Input Description:
"{user_description}"

Sample Domain Names (Analyze their syllable count, sound patterns, and rhythm):
{sample_domains}

Instructions:
- Generate 10-15 domain names.
- Each name should ideally be 2-3 syllables long.
- Prioritize names with a smooth, flowing pronunciation.
- Avoid names that are difficult to articulate or contain awkward consonant clusters.
- Focus on strong, clear sounds.
- Do not repeat sample names.
- Provide a numbered list.

Phonetically Optimized Domain Names:
"""
    
    response = llm(prompt)
    # print(response)
    return response

def postprocessing(text):
    domain_names = re.findall(r'\d+\.\s+([a-zA-Z0-9]+)', text)
    print("----------------------------Sample_domains----------------------------")
    print(domain_names)
    return domain_names

def final_domains():
    pass


import ast

def domain_details(domain_name: str, prompt: str):
    template = f"""
You are a branding and domain expert.Generate a Python dictionary in the following format. The description should 50-100 words and it should describe how domain name suitable for the user requirements:

{{
    "domainName": "{domain_name}.lk",
    "domainDescription": "...",  # a creative description using the prompt
    "relatedFields": [ ... ]     # 4 to 6 relevant fields
}}

Domain name: {domain_name}
Prompt: {prompt}
"""
    
    response_text = llm.invoke(template)  # ✅ use .invoke() instead of calling directly

    #print(type(response_text))  # should be <class 'str'> if it’s a string

    try:
        result = ast.literal_eval(response_text)
    except Exception as e:
        result = {
            "domainName": f"{domain_name}.lk",
            "domainDescription": "Failed to parse response from LLM.",
            "relatedFields": [],
            "error": str(e)
        }

    return result  # ✅ This is a dict now


def gemma(user_description: str, sample_domains: str) -> str:
    input_text=f"""
        You are an expert domain name generator. Your task is to create domain name suggestions that closely match the user's input and follow the style and pattern of the sample domain names provided.

        User Input Description:
        "{user_description}"

        Sample Domain Names:
        {sample_domains}

        Instructions:
        - Generate 10 to 15 domain names that fit the user's input description.
        - The names should be short, easy to understand, creative, memorable, and relevant to the input.
        - Use similar word structures and language style as the samples.
        - Avoid overly long or complicated names; keep them concise and simple.
        - Do not repeat exact sample names.
        - Provide only the domain name suggestions without any domain extensions (like .com, .net, .lk).
        - Provide the domain names in a numbered list .

        Suggested Domain Names:
"""
    payload = {
        "inputs": input_text,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_k": 50,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    #print(response.json())
    return response.json()

def gemma_post_processing(output):
    text = output[0]['generated_text']
    suggested_text = text.split("Suggested Domain Names:")[-1]
    domain_names = re.findall(r'\d+\.\s*([A-Za-z0-9]+)', suggested_text)
    domain_names = list(dict.fromkeys(domain_names))
    return domain_names


def gemma_decsription(domain_name: str, prompt: str):
    template = f"""
        You are a branding and domain expert.Generate a Python dictionary in the following format. The description should 50-100 words and it should describe how domain name suitable for the user requirements:
        Domain name: {domain_name}
        Prompt: {prompt}
        {{
            "domainName": "{domain_name}.lk",
            "domainDescription": "...",  # a creative description using the prompt
            "relatedFields": [ ... ]     # 4 to 6 relevant fields
        }}

        output:
    """
    payload = {
        "inputs": template,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_k": 50,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    llm_response=response.json()
    return llm_response[0]['generated_text'],domain_name

def gemma_preprocess(llm_output,domain_name):
    try:
        # Try to find a code block with ```json or ```python
        code_block_match = re.search(r'```(?:json|python)?\s*(\{[\s\S]*?\})\s*```', llm_output, re.IGNORECASE)

        if code_block_match:
            json_string = code_block_match.group(1)
        else:
            # Fallback: try to find the last JSON-like block in the output
            all_matches = re.findall(r'\{[\s\S]*?\}', llm_output)
            if not all_matches:
                raise ValueError("No JSON object found.")
            json_string = all_matches[-1]  # Use the last one assuming it's the actual output

        # Try to parse the JSON
        parsed_json = json.loads(json_string)
        return parsed_json

    except Exception:
        return {
            "domainName": f"{domain_name}.lk",
            "domainDescription": "Failed to parse response from LLM.",
            "relatedFields": []
        }
