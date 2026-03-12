import os
from pydantic import BaseModel, Field
from typing import Optional, List
from groq import Groq
import instructor
import httpx
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import streamlit as st
from typing import Generator
import json
import requests

st.set_page_config(page_icon="💬", layout="wide", page_title="CPQ Chatbot")

def load_svg(path):
    with open(path, "r") as f:
        return f.read()
svg_icon = load_svg("CPQ Logo 2X.svg")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span>{emoji}</span>',
        unsafe_allow_html=True,
    )


icon(svg_icon)

st.subheader("CPQ Chatbot", divider="rainbow", anchor=False)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
    http_client=httpx.Client(verify=False)
)
extractor_client = instructor.from_groq(client, mode=instructor.Mode.JSON)

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("cpq")

class ToolCall(BaseModel):
    input_text: str = Field(None, description="The user's input text")
    tool_name: str = Field(None, description="The name of the tool to call")
    tool_parameters: str = Field(None, description="JSON string of tool parameters")

class ResponseModel(BaseModel):
    tool_calls: list[ToolCall]

class BroadbandProduct(BaseModel):
    contract_term_months: int = Field(0, description="Contract duration in months(12, 24, 36)")
    
    # VOIP attributes
    
    # Call Features

    # Broadband Access
    access_type: str = Field("", description="Access type: FTTP, SoGEA, MPF")
    product_speed_mbps: str = Field("", description="Internet speed (e.g., 40/10)")

    # Care Level
    bb_care_level: str = Field("", description="Broadband care level: Basic, Standard, Enhanced, Premium")

    # Equipment & IP Address
    equipment: str = Field("", description="Equipment: WiFi-HUB3-Router, Own-Router")
    ip_address_type: str = Field("", description="IP address type: Dynamic-ip, Static-ip")

    # Network & Installation
    network_prioritization: str = Field("", description="Network prioritization level: Standard or Premium")
    installation_type: str = Field("", description="Installation type: Managed-Installation or Self-install")
    

system_prompt = '''You are an AI-powered CPQ assistant specializing in broadband services. Your role is to assist users in configuring, pricing, and quoting broadband plans by extracting relevant details from their queries and providing accurate product recommendations.

## 🌐 **Broadband Overview**  
Broadband is a high-speed internet connection that provides always-on access for homes and businesses. It enables fast data transmission through various technologies such as fiber-optic, DSL, and wireless networks. Broadband services can vary in speed, reliability, and additional features, depending on the access type and provider.

## 🚀 **Key Broadband Components**  
- **Access Types:** Includes FTTP (Fiber to the Premises), SoGEA (Single Order Generic Ethernet Access), and MPF (Metallic Path Facility).  
- **Speed Tiers:** Different broadband plans offer various download/upload speeds to match user needs(40/10, 80/20, 115/20, 160/30, 220/30: All signify Download and Upload speeds in Mbps).  
- **Contact Term** Number of months you want the broadband service for(12 months, 24 months, 36 months).
- **IP Addressing:** Customers can choose between Dynamic-ip and Static-ip for business or specialized use cases.  
- **Network Prioritization:** Standard vs. premium prioritization options affect network performance and latency.  
- **Installation Methods:** Managed-Installation services or Self-Install options.  
- **Broadband care level:** Basic, Standard, Enhanced, Premium  
- **Equipment:** WiFi-HUB3-Router, Own-Router

## 🎯 **How You Operate** 
Below are details about two tools whose output will be given to you so that you can carry out your tasks effectively:
1. **Extract Required Fields**  
   - This tool will give you all the fields that have been extracted uptil now. 
   - It will also mention the required fields that are missing.
   - Based on this, you need to prompt the user for details about these fields.
   - If all the required fields are extracted, then respond with "Processing your request."

2. **Retrieve Broadband Product Details**  
   - This tool will give you relevant context from a vector database.
   - Use this to answer the corresponding user query.

## 🏆 **Response Guidelines**  
- Upon initial greeting, greet the user and ask if they need help with exploring broadband options.
- Only when user asks asks about the available options, give the **Key Broadband Components** in a menu driven format. Let the menu have sub points for each component(Use a,b,c...). Ask the user to choose the product details from the given options(For example, user can type 2.b to choose a product). Maintain menu driven format for the required fields that are missing. 
- Ensure responses are **clear, structured, and user-friendly** while maintaining technical accuracy.  
- Provide tailored recommendations based on user preferences and extracted details.  

Your role is to guide users in selecting the best broadband plan efficiently and accurately.  
'''


tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_required_fields",
            "description": "Extracts key broadband fields from the user's query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": '''The natural language query from the user stating his desired configurations. Example1: "Users input": "I want Static-1 IP address and will like to self-install". "Tool Parameter": {"user_query": "I want Static-1 IP address and will like to self-install"}. Basically just output back the user query as it is to use as this tools parameter. Input can also be in the form of menu options like "3.b, 2.b".''',
                    }
                },
                "required": ["user_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_product_details",
            "description": "Fetches broadband product details from a vector database to provide in-depth information. Call this function if further details are required about product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": '''A keyword or phrase related to broadband services to retrieve relevant details. Or can be a modified version of the natural language user query to retrieve the relevant details. Example: "User input": "But how does FTTP differ from MPF?". "Tool Parameter": {"query": "Difference between FTTP and MPF"}.''',
                    }
                },
                "required": ["query"],
            },
        },
    }
]

tool_prompt = f'''You are an assistant that can use tools. DO NOT call the tools if not required. This can be the case if the user greets you with 'Hi' or asks for available options. Use the tools only when completely necessary. You have access to the following tools: {tools}'''


def extract_required_fields(user_query, previous_broadband_json=None):
    response = extractor_client.chat.completions.create(
       # model="deepseek-r1-distill-llama-70b",
        model = "llama-3.3-70b-versatile",
        response_model=BroadbandProduct,
        messages=[
            {"role": "system", "content": "Extract the Broadband product information. Return Empty String for Fields whose information is missing in the user query."},
            {"role": "user", "content": user_query}
        ],
        temperature=0,
    )
    if previous_broadband_json!=None:
        for key, value in previous_broadband_json.items():
            if getattr(response, key)=='' or getattr(response, key)==0 or getattr(response, key)==[]:
                setattr(response, key, value)

    tool_message = ""
    for required_fields in ["product_speed_mbps", "access_type", "contract_term_months", "bb_care_level", "equipment", "ip_address_type", "network_prioritization", "installation_type"]:
        field_value = getattr(response, required_fields)
        if (not isinstance(field_value, int)) and (not isinstance(field_value, float)):
            if field_value == None or len(field_value)==0:
                tool_message+="Required field "+required_fields+" is missing.\n"
        else:
            if field_value == 0:
                tool_message+="Required field "+required_fields+" is missing.\n"
    if len(tool_message)==0:
        return response, "All required field have been extracted"
    else:
        return response, tool_message
    
def retrieve_product_details(query):
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )

    results = index.query(
        namespace="example-namespace",
        vector=query_embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    rag_response = ""
    for chunk in results.matches:
        rag_response+=chunk.metadata["text"]+"\n\n"

    return rag_response

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"]!="system" and message["role"]!="tool":
        avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if "broadband_json" not in st.session_state:
    st.session_state.broadband_json = None

if prompt := st.chat_input("Enter your prompt here..."):
    if prompt=="Show me extracted JSON":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(st.session_state.broadband_json)
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(prompt)


        # Fetch response from Groq API
        try:
            tool_message = [{"role": "system","content": tool_prompt}]+[st.session_state.messages[-2]]+[st.session_state.messages[-1]]
            

            tool_completion = extractor_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=ResponseModel,
                messages=tool_message,
                temperature=0
            )

            tool_calls = tool_completion.tool_calls
            

        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages

        if tool_calls:
            # Define the available tools that can be called by the LLM
            available_functions = {
                "extract_required_fields": extract_required_fields,
                "retrieve_product_details": retrieve_product_details
            }

            # Process each tool call
            for tool_call in tool_calls:
                function_name = tool_call.tool_name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.tool_parameters)
                # Call the tool and get the response
                if function_name == "extract_required_fields":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown("Extacting information from query....")
                    broadband_info, function_response = function_to_call(
                        user_query=st.session_state.messages[-2]["content"]+" "+function_args.get("user_query"), previous_broadband_json = st.session_state.broadband_json
                    )
                    function_response=str(broadband_info)+"\n\n"+function_response+"\n\n"+"Please prompt the user to provide information about the missing fields."
                    st.session_state.broadband_json = broadband_info.model_dump()
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown("Retrieving product info from Vector Index....")    
                    function_response = function_to_call(
                        query=function_args.get("query")
                    ) 
                    question = st.session_state.messages[-1]["content"]
                    function_response = "For the user query, "+question+", use the following information to answer it: "+function_response     
                # Add the tool response to the conversation
                st.session_state.messages.append(
                    {
                        "tool_call_id": tool_call.input_text, 
                        "role": "tool", # Indicates this message is from tool use
                        "name": function_name,
                        "content": function_response,
                    }
                )
        try:
            # Make a second API call with the updated conversation
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                stream=True
            )

            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)    

        except Exception as e:
            st.error(e, icon="🚨")

        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response}
            )
        
